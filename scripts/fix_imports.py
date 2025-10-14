#!/usr/bin/env python3
"""
Fix imports across codebase for new unified structure.
Replaces Supabase imports with SQLite database client.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple


# Patterns to replace
REPLACEMENTS = [
    # Supabase client imports
    (r'from\s+supabase\s+import\s+create_client', 'from src.database import get_db'),
    (r'from\s+.*supabase_client\s+import\s+SupabaseClient', 'from src.database import SQLiteClient as DatabaseClient'),
    (r'from\s+data_pipeline\.supabase_client\s+import\s+SupabaseClient', 'from src.database import SQLiteClient as DatabaseClient'),
    (r'from\s+data_pipeline\.supabase\.client\s+import\s+SupabaseClient', 'from src.database import SQLiteClient as DatabaseClient'),
    
    # Direct Supabase usage
    (r'SupabaseClient\(\)', 'get_db()'),
    (r'create_client\(.*?\)', 'get_db()'),
    
    # Environment variables
    (r'SUPABASE_URL', 'DATABASE_PATH'),
    (r'SUPABASE_SERVICE_KEY', 'DATABASE_TYPE'),
]


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', 'node_modules', '.pytest_cache']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    return python_files


def replace_imports_in_file(file_path: Path) -> Tuple[bool, int]:
    """
    Replace imports in a single file.
    
    Returns:
        (modified, num_replacements)
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        num_replacements = 0
        
        for pattern, replacement in REPLACEMENTS:
            new_content, count = re.subn(pattern, replacement, content)
            if count > 0:
                content = new_content
                num_replacements += count
        
        # Write back if modified
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True, num_replacements
        
        return False, 0
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0


def main():
    """Main function to fix imports."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix imports for new database architecture')
    parser.add_argument(
        '--directory',
        default='/Volumes/Lexar/RRRVentures/RRRalgorithms/src',
        help='Directory to process'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    
    args = parser.parse_args()
    
    directory = Path(args.directory)
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return
    
    print(f"Scanning for Python files in: {directory}")
    python_files = find_python_files(directory)
    print(f"Found {len(python_files)} Python files\n")
    
    modified_count = 0
    total_replacements = 0
    
    for file_path in python_files:
        if args.dry_run:
            # Just check without modifying
            with open(file_path, 'r') as f:
                content = f.read()
            
            found_patterns = []
            for pattern, _ in REPLACEMENTS:
                if re.search(pattern, content):
                    found_patterns.append(pattern)
            
            if found_patterns:
                print(f"\n{file_path}:")
                for pattern in found_patterns:
                    print(f"  - Would replace: {pattern}")
                modified_count += 1
        else:
            modified, num_replacements = replace_imports_in_file(file_path)
            if modified:
                print(f"✅ {file_path}: {num_replacements} replacements")
                modified_count += 1
                total_replacements += num_replacements
    
    print(f"\n{'=' * 60}")
    if args.dry_run:
        print(f"Would modify {modified_count} files")
    else:
        print(f"Modified {modified_count} files")
        print(f"Total replacements: {total_replacements}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

