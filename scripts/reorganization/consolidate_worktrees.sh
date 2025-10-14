#!/bin/bash
# ============================================================================
# Worktree Consolidation Script
# ============================================================================
# This script consolidates all git worktrees back into the main repository
# to simplify the project structure and reduce overhead.
#
# Author: RRR Ventures
# Date: 2025-10-12
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MAIN_REPO_DIR="$(pwd)"
WORKTREES_DIR="$MAIN_REPO_DIR/worktrees"
BACKUP_DIR="$MAIN_REPO_DIR/backups/worktree-backup-$(date +%Y%m%d-%H%M%S)"
TARGET_DIR="$MAIN_REPO_DIR/src/services"
LOG_FILE="$MAIN_REPO_DIR/consolidation.log"

# Worktrees to consolidate
WORKTREES=(
    "neural-network"
    "data-pipeline"
    "trading-engine"
    "risk-management"
    "backtesting"
    "api-integration"
    "quantum-optimization"
    "monitoring"
)

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

error() {
    log "${RED}ERROR: $1${NC}"
    exit 1
}

success() {
    log "${GREEN}✓ $1${NC}"
}

warning() {
    log "${YELLOW}⚠ $1${NC}"
}

info() {
    log "${BLUE}ℹ $1${NC}"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

pre_flight_checks() {
    info "Running pre-flight checks..."

    # Check if we're in the right directory
    if [ ! -f "$MAIN_REPO_DIR/CLAUDE.md" ]; then
        error "Not in RRRalgorithms root directory. Please run from project root."
    fi

    # Check if worktrees exist
    if [ ! -d "$WORKTREES_DIR" ]; then
        error "Worktrees directory not found at $WORKTREES_DIR"
    fi

    # Check git status
    if ! git diff --quiet; then
        warning "You have uncommitted changes in main repository"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "Aborted by user"
        fi
    fi

    # Create target directories
    mkdir -p "$TARGET_DIR"
    mkdir -p "$BACKUP_DIR"

    success "Pre-flight checks passed"
}

# ============================================================================
# Backup Current State
# ============================================================================

create_backup() {
    info "Creating backup of current state..."

    # Backup worktrees
    for worktree in "${WORKTREES[@]}"; do
        if [ -d "$WORKTREES_DIR/$worktree" ]; then
            info "Backing up $worktree..."
            cp -r "$WORKTREES_DIR/$worktree" "$BACKUP_DIR/" 2>/dev/null || true
        fi
    done

    # Create git bundle for safety
    info "Creating git bundle backup..."
    git bundle create "$BACKUP_DIR/repo-backup.bundle" --all

    success "Backup created at $BACKUP_DIR"
}

# ============================================================================
# Consolidate Worktree
# ============================================================================

consolidate_worktree() {
    local worktree_name=$1
    local worktree_path="$WORKTREES_DIR/$worktree_name"
    local target_path="$TARGET_DIR/${worktree_name//-/_}"  # Convert hyphens to underscores

    info "Consolidating $worktree_name..."

    if [ ! -d "$worktree_path" ]; then
        warning "Worktree $worktree_name not found, skipping"
        return
    fi

    # Check if worktree has uncommitted changes
    cd "$worktree_path"
    if ! git diff --quiet || ! git diff --cached --quiet; then
        warning "Worktree $worktree_name has uncommitted changes"

        # Stash changes
        git stash push -m "Auto-stash during consolidation"
        info "Changes stashed in $worktree_name"
    fi

    # Get the current branch
    local branch=$(git branch --show-current)
    info "Worktree $worktree_name is on branch: $branch"

    cd "$MAIN_REPO_DIR"

    # Create target directory
    mkdir -p "$target_path"

    # Copy source files (excluding .git and other metadata)
    if [ -d "$worktree_path/src" ]; then
        info "Copying src files from $worktree_name..."
        rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
              --exclude='.pytest_cache' --exclude='node_modules' \
              "$worktree_path/src/" "$target_path/" 2>/dev/null || true
    fi

    # Copy tests if they exist
    if [ -d "$worktree_path/tests" ]; then
        local test_target="$MAIN_REPO_DIR/tests/services/${worktree_name//-/_}"
        mkdir -p "$test_target"
        info "Copying tests from $worktree_name..."
        rsync -av --exclude='__pycache__' --exclude='*.pyc' \
              --exclude='.pytest_cache' \
              "$worktree_path/tests/" "$test_target/" 2>/dev/null || true
    fi

    # Copy configuration files if they exist
    if [ -d "$worktree_path/config" ]; then
        local config_target="$MAIN_REPO_DIR/config/services/${worktree_name//-/_}"
        mkdir -p "$config_target"
        info "Copying config from $worktree_name..."
        cp -r "$worktree_path/config/"* "$config_target/" 2>/dev/null || true
    fi

    # Copy documentation if it exists
    if [ -d "$worktree_path/docs" ]; then
        local docs_target="$MAIN_REPO_DIR/docs/services/${worktree_name//-/_}"
        mkdir -p "$docs_target"
        info "Copying docs from $worktree_name..."
        cp -r "$worktree_path/docs/"* "$docs_target/" 2>/dev/null || true
    fi

    success "Consolidated $worktree_name to $target_path"
}

# ============================================================================
# Update Imports
# ============================================================================

update_imports() {
    info "Updating import statements..."

    # Python import updates
    find "$TARGET_DIR" -name "*.py" -type f | while read -r file; do
        # Update worktree-specific imports to new structure
        sed -i.bak -E 's/from worktrees\.([a-z-]+)\./from src.services.\1./g' "$file"
        sed -i.bak -E 's/import worktrees\.([a-z-]+)\./import src.services.\1./g' "$file"

        # Convert hyphenated names to underscores in imports
        sed -i.bak -E 's/\.neural-network\./\.neural_network\./g' "$file"
        sed -i.bak -E 's/\.data-pipeline\./\.data_pipeline\./g' "$file"
        sed -i.bak -E 's/\.trading-engine\./\.trading_engine\./g' "$file"
        sed -i.bak -E 's/\.risk-management\./\.risk_management\./g' "$file"
        sed -i.bak -E 's/\.api-integration\./\.api_integration\./g' "$file"
        sed -i.bak -E 's/\.quantum-optimization\./\.quantum_optimization\./g' "$file"

        # Remove backup files
        rm -f "${file}.bak"
    done

    success "Import statements updated"
}

# ============================================================================
# Remove Worktrees
# ============================================================================

remove_worktrees() {
    info "Removing git worktrees..."

    for worktree in "${WORKTREES[@]}"; do
        if git worktree list | grep -q "$worktree"; then
            info "Removing worktree $worktree..."
            git worktree remove "$WORKTREES_DIR/$worktree" --force 2>/dev/null || true
        fi
    done

    # Prune worktree references
    git worktree prune

    success "Git worktrees removed"
}

# ============================================================================
# Create Service Registry
# ============================================================================

create_service_registry() {
    info "Creating service registry..."

    cat > "$MAIN_REPO_DIR/src/core/service_registry.py" << 'EOF'
"""
Service Registry
================

Central registry for all microservices in the trading system.
Manages service lifecycle and dependencies.

Author: RRR Ventures
Date: 2025-10-12
"""

import logging
from typing import Dict, Optional, Any, List
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ServiceInfo:
    """Service information."""
    name: str
    status: ServiceStatus
    dependencies: List[str]
    config: Dict[str, Any]
    instance: Optional["BaseService"]


class BaseService(ABC):
    """Abstract base class for all services."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.status = ServiceStatus.STOPPED
        self.logger = logging.getLogger(f"service.{name}")

    @abstractmethod
    async def start(self):
        """Start the service."""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the service."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check service health."""
        pass

    async def restart(self):
        """Restart the service."""
        await self.stop()
        await self.start()


class ServiceRegistry:
    """Central registry for all microservices."""

    def __init__(self):
        self._services: Dict[str, ServiceInfo] = {}
        self._startup_order: List[str] = []
        self.logger = logger

    def register(self,
                 name: str,
                 service_class: type,
                 config: Dict[str, Any],
                 dependencies: Optional[List[str]] = None):
        """Register a service."""
        dependencies = dependencies or []

        # Validate dependencies
        for dep in dependencies:
            if dep not in self._services and dep != name:
                self.logger.warning(f"Service {name} depends on unregistered service {dep}")

        # Create service info
        info = ServiceInfo(
            name=name,
            status=ServiceStatus.STOPPED,
            dependencies=dependencies,
            config=config,
            instance=None
        )

        self._services[name] = info
        self._update_startup_order()

        self.logger.info(f"Registered service: {name}")

    def _update_startup_order(self):
        """Update service startup order based on dependencies."""
        # Simple topological sort
        visited = set()
        order = []

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)

            if name in self._services:
                for dep in self._services[name].dependencies:
                    if dep in self._services:
                        visit(dep)
                order.append(name)

        for name in self._services:
            visit(name)

        self._startup_order = order

    async def start_all(self):
        """Start all registered services in dependency order."""
        self.logger.info("Starting all services...")

        for name in self._startup_order:
            await self.start_service(name)

        self.logger.info("All services started")

    async def start_service(self, name: str):
        """Start a specific service."""
        if name not in self._services:
            raise ValueError(f"Service {name} not registered")

        info = self._services[name]

        if info.status == ServiceStatus.RUNNING:
            self.logger.info(f"Service {name} already running")
            return

        # Start dependencies first
        for dep in info.dependencies:
            if dep in self._services:
                await self.start_service(dep)

        # Create and start service instance
        try:
            info.status = ServiceStatus.STARTING

            # Dynamic import based on service name
            module_name = f"src.services.{name.replace('-', '_')}"
            module = __import__(module_name, fromlist=['Service'])
            service_class = getattr(module, 'Service')

            info.instance = service_class(name, info.config)
            await info.instance.start()

            info.status = ServiceStatus.RUNNING
            self.logger.info(f"Service {name} started successfully")

        except Exception as e:
            info.status = ServiceStatus.ERROR
            self.logger.error(f"Failed to start service {name}: {e}")
            raise

    async def stop_all(self):
        """Stop all services in reverse dependency order."""
        self.logger.info("Stopping all services...")

        for name in reversed(self._startup_order):
            await self.stop_service(name)

        self.logger.info("All services stopped")

    async def stop_service(self, name: str):
        """Stop a specific service."""
        if name not in self._services:
            return

        info = self._services[name]

        if info.status != ServiceStatus.RUNNING:
            return

        try:
            info.status = ServiceStatus.STOPPING

            if info.instance:
                await info.instance.stop()

            info.status = ServiceStatus.STOPPED
            self.logger.info(f"Service {name} stopped")

        except Exception as e:
            self.logger.error(f"Error stopping service {name}: {e}")

    def get_service(self, name: str) -> Optional[BaseService]:
        """Get a service instance."""
        if name in self._services:
            return self._services[name].instance
        return None

    def get_status(self, name: Optional[str] = None) -> Dict[str, ServiceStatus]:
        """Get service status."""
        if name:
            if name in self._services:
                return {name: self._services[name].status}
            return {}

        return {name: info.status for name, info in self._services.items()}

    async def health_check(self) -> Dict[str, bool]:
        """Run health checks on all services."""
        results = {}

        for name, info in self._services.items():
            if info.instance and info.status == ServiceStatus.RUNNING:
                try:
                    results[name] = await info.instance.health_check()
                except Exception as e:
                    self.logger.error(f"Health check failed for {name}: {e}")
                    results[name] = False
            else:
                results[name] = False

        return results


# Global registry instance
registry = ServiceRegistry()


def get_registry() -> ServiceRegistry:
    """Get the global service registry."""
    return registry
EOF

    success "Service registry created"
}

# ============================================================================
# Update Configuration
# ============================================================================

update_configuration() {
    info "Updating configuration files..."

    # Update Docker Compose
    if [ -f "$MAIN_REPO_DIR/docker-compose.yml" ]; then
        info "Updating docker-compose.yml..."
        # Backup original
        cp "$MAIN_REPO_DIR/docker-compose.yml" "$BACKUP_DIR/docker-compose.yml.backup"

        # Update paths in docker-compose
        sed -i.bak 's|./worktrees/|./src/services/|g' "$MAIN_REPO_DIR/docker-compose.yml"
        rm -f "$MAIN_REPO_DIR/docker-compose.yml.bak"
    fi

    # Update package.json if exists
    if [ -f "$MAIN_REPO_DIR/package.json" ]; then
        info "Updating package.json..."
        sed -i.bak 's|worktrees/|src/services/|g' "$MAIN_REPO_DIR/package.json"
        rm -f "$MAIN_REPO_DIR/package.json.bak"
    fi

    success "Configuration files updated"
}

# ============================================================================
# Generate Migration Report
# ============================================================================

generate_report() {
    info "Generating migration report..."

    cat > "$MAIN_REPO_DIR/MIGRATION_REPORT.md" << EOF
# Worktree Consolidation Report
Generated: $(date)

## Summary
Successfully consolidated ${#WORKTREES[@]} worktrees into main repository structure.

## Changes Made

### Directory Structure
- Moved worktrees to \`src/services/\`
- Standardized naming (hyphens to underscores)
- Created service registry at \`src/core/service_registry.py\`

### Consolidated Worktrees
$(for worktree in "${WORKTREES[@]}"; do
    echo "- $worktree → src/services/${worktree//-/_}"
done)

### Files Changed
- Total Python files updated: $(find "$TARGET_DIR" -name "*.py" | wc -l)
- Import statements fixed: $(grep -r "from src.services" "$TARGET_DIR" | wc -l)

### Backup Location
\`$BACKUP_DIR\`

## Next Steps
1. Run tests: \`pytest tests/\`
2. Update CI/CD pipelines
3. Remove backup after verification: \`rm -rf $BACKUP_DIR\`

## Rollback Instructions
If needed, restore from backup:
\`\`\`bash
# Restore git bundle
git bundle unbundle $BACKUP_DIR/repo-backup.bundle

# Restore worktrees
cp -r $BACKUP_DIR/* $WORKTREES_DIR/
\`\`\`
EOF

    success "Migration report generated: MIGRATION_REPORT.md"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo "=============================================="
    echo "     Worktree Consolidation Script"
    echo "=============================================="
    echo

    # Initialize log
    echo "Starting consolidation at $(date)" > "$LOG_FILE"

    # Run pre-flight checks
    pre_flight_checks

    # Create backup
    create_backup

    # Consolidate each worktree
    for worktree in "${WORKTREES[@]}"; do
        consolidate_worktree "$worktree"
    done

    # Update imports
    update_imports

    # Create service registry
    create_service_registry

    # Update configuration
    update_configuration

    # Remove worktrees
    read -p "Remove git worktrees? This is irreversible. (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        remove_worktrees
    else
        warning "Skipped worktree removal"
    fi

    # Generate report
    generate_report

    echo
    echo "=============================================="
    success "Consolidation complete!"
    echo "=============================================="
    echo
    info "Review MIGRATION_REPORT.md for details"
    info "Run tests to verify: pytest tests/"
    info "Backup saved at: $BACKUP_DIR"
    echo
}

# Run main function
main "$@"