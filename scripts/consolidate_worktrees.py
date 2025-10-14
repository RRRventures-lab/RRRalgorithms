#!/usr/bin/env python3
"""
Consolidate worktrees into main src/ directory.
This script merges all worktrees into a unified monorepo structure.
"""

import os
import shutil
from pathlib import Path
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Worktree to destination mapping
WORKTREE_MAPPING = {
    'neural-network': 'src/neural_network',
    'data-pipeline': 'src/data_pipeline_original',  # Keep original separate initially
    'trading-engine': 'src/trading/engine',
    'risk-management': 'src/trading/risk',
    'backtesting': 'src/backtesting',
    'api-integration': 'src/api',
    'quantum-optimization': 'src/quantum',
    'monitoring': 'src/monitoring_original',  # Keep original separate initially
}


def safe_copy_tree(src: Path, dst: Path, skip_patterns=None):
    """
    Safely copy directory tree, skipping certain patterns.
    
    Args:
        src: Source directory
        dst: Destination directory
        skip_patterns: List of directory names to skip
    """
    if skip_patterns is None:
        skip_patterns = ['__pycache__', '.venv', 'venv', '.git', 'node_modules', '.pytest_cache']
    
    # Create destination if it doesn't exist
    dst.mkdir(parents=True, exist_ok=True)
    
    # Copy files and directories
    for item in src.iterdir():
        # Skip patterns
        if item.name in skip_patterns or item.name.startswith('.'):
            logger.debug(f"Skipping {item}")
            continue
        
        dest_item = dst / item.name
        
        if item.is_dir():
            # Recursively copy directory
            safe_copy_tree(item, dest_item, skip_patterns)
        else:
            # Copy file
            shutil.copy2(item, dest_item)
            logger.debug(f"Copied {item} -> {dest_item}")


def consolidate_worktree(worktree_name: str, dest_path: str, base_dir: Path):
    """
    Consolidate a single worktree into main structure.
    
    Args:
        worktree_name: Name of worktree directory
        dest_path: Destination path relative to base_dir
        base_dir: Base directory (RRRalgorithms root)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Consolidating: {worktree_name} -> {dest_path}")
    logger.info(f"{'='*60}")
    
    src_worktree = base_dir / 'worktrees' / worktree_name
    dest = base_dir / dest_path
    
    if not src_worktree.exists():
        logger.warning(f"Worktree not found: {src_worktree}")
        return
    
    # Find src directory in worktree
    src_dir = src_worktree / 'src'
    
    if src_dir.exists():
        logger.info(f"Copying from {src_dir}")
        safe_copy_tree(src_dir, dest)
    else:
        logger.warning(f"No src/ directory found in {worktree_name}")
        # Try copying root files
        logger.info(f"Copying from {src_worktree} (root level)")
        for item in src_worktree.iterdir():
            if item.name in ['src', 'tests', 'scripts', 'config', 'docs']:
                continue  # Will handle these separately
            if item.is_file() and item.suffix == '.py':
                dest_file = dest / item.name
                shutil.copy2(item, dest_file)
    
    # Copy tests
    src_tests = src_worktree / 'tests'
    if src_tests.exists():
        dest_tests = base_dir / 'tests' / worktree_name.replace('-', '_')
        logger.info(f"Copying tests: {src_tests} -> {dest_tests}")
        safe_copy_tree(src_tests, dest_tests)
    
    logger.info(f"✅ Consolidated {worktree_name}")


def create_init_files(base_dir: Path):
    """Create __init__.py files in all directories."""
    logger.info("\nCreating __init__.py files...")
    
    src_dir = base_dir / 'src'
    for dirpath, dirnames, filenames in os.walk(src_dir):
        # Skip certain directories
        dirnames[:] = [d for d in dirnames if d not in ['__pycache__', '.git', 'venv']]
        
        init_file = Path(dirpath) / '__init__.py'
        if not init_file.exists() and any(f.endswith('.py') for f in filenames):
            init_file.touch()
            logger.debug(f"Created {init_file}")


def update_imports_in_directory(directory: Path, old_prefix: str, new_prefix: str):
    """
    Update import statements in all Python files in directory.
    
    Args:
        directory: Directory to process
        old_prefix: Old import prefix (e.g., 'from data_pipeline')
        new_prefix: New import prefix (e.g., 'from src.data_pipeline')
    """
    logger.info(f"Updating imports: {old_prefix} -> {new_prefix}")
    
    count = 0
    for py_file in directory.rglob('*.py'):
        try:
            content = py_file.read_text()
            new_content = content.replace(f'from {old_prefix}', f'from {new_prefix}')
            new_content = new_content.replace(f'import {old_prefix}', f'import {new_prefix}')
            
            if content != new_content:
                py_file.write_text(new_content)
                count += 1
        except Exception as e:
            logger.error(f"Error processing {py_file}: {e}")
    
    logger.info(f"Updated {count} files")


def main():
    """Main consolidation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Consolidate worktrees into main repository')
    parser.add_argument(
        '--base-dir',
        default='/Volumes/Lexar/RRRVentures/RRRalgorithms',
        help='Base directory path'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
    )
    parser.add_argument(
        '--worktree',
        help='Consolidate only specific worktree'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        logger.error(f"Base directory not found: {base_dir}")
        return
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        for worktree, dest in WORKTREE_MAPPING.items():
            src = base_dir / 'worktrees' / worktree
            dst = base_dir / dest
            logger.info(f"Would consolidate: {src} -> {dst}")
        return
    
    # Create necessary directories
    logger.info("Creating directory structure...")
    for dest in WORKTREE_MAPPING.values():
        dest_path = base_dir / dest
        dest_path.mkdir(parents=True, exist_ok=True)
    
    # Consolidate each worktree
    if args.worktree:
        # Consolidate single worktree
        if args.worktree in WORKTREE_MAPPING:
            consolidate_worktree(
                args.worktree,
                WORKTREE_MAPPING[args.worktree],
                base_dir
            )
        else:
            logger.error(f"Unknown worktree: {args.worktree}")
            logger.info(f"Available: {list(WORKTREE_MAPPING.keys())}")
            return
    else:
        # Consolidate all worktrees
        for worktree, dest in WORKTREE_MAPPING.items():
            consolidate_worktree(worktree, dest, base_dir)
    
    # Create __init__.py files
    create_init_files(base_dir)
    
    logger.info("\n" + "="*60)
    logger.info("Consolidation complete!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Fix imports: python scripts/fix_imports.py")
    logger.info("2. Run tests: pytest tests/")
    logger.info("3. Review and commit changes")


if __name__ == '__main__':
    main()

