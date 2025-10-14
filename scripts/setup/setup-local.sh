#!/bin/bash
# =============================================================================
# RRRalgorithms - Local Development Setup Script
# One-command setup for local laptop development
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Determine project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

print_header "RRRalgorithms Local Setup"
echo ""
print_info "Project root: $PROJECT_ROOT"
echo ""

# Check Python version
print_info "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
        print_success "Python $PYTHON_VERSION (✓ 3.11+)"
    else
        print_error "Python 3.11+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.11+"
    exit 1
fi

# Create virtual environment
print_info "Creating virtual environment..."
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    print_info "Virtual environment already exists, skipping creation"
else
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip --quiet
print_success "Pip upgraded"

# Install dependencies
print_info "Installing dependencies..."
print_info "This may take a few minutes..."

if pip install -r requirements-local.txt --quiet; then
    print_success "Dependencies installed (local mode - minimal)"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Ask about optional dependencies
echo ""
read -p "Install development tools (pytest, black, ruff)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Installing dev dependencies..."
    pip install pytest pytest-asyncio pytest-cov pytest-mock black ruff mypy --quiet
    print_success "Dev tools installed"
fi

echo ""
read -p "Install rich library for better terminal output? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    print_info "Installing rich..."
    pip install rich --quiet
    print_success "Rich installed"
fi

# Create .env.local if it doesn't exist
print_info "Setting up environment configuration..."
if [ -f "$PROJECT_ROOT/.env.local" ]; then
    print_info ".env.local already exists, skipping"
else
    if [ -f "$PROJECT_ROOT/config/env.local.template" ]; then
        cp "$PROJECT_ROOT/config/env.local.template" "$PROJECT_ROOT/.env.local"
        print_success "Created .env.local from template"
        print_info "You can edit .env.local to customize settings"
    else
        print_error "Template file not found: config/env.local.template"
    fi
fi

# Create necessary directories
print_info "Creating directories..."
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/models"
print_success "Directories created"

# Initialize database
print_info "Initializing database..."
if python3 scripts/setup/init-local-db.py; then
    print_success "Database initialized with sample data"
else
    print_error "Failed to initialize database"
    exit 1
fi

# Summary
echo ""
print_header "Setup Complete!"
echo ""
print_success "RRRalgorithms is ready for local development"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate virtual environment:"
echo "     ${GREEN}source venv/bin/activate${NC}"
echo ""
echo "  2. Start the trading system:"
echo "     ${GREEN}./scripts/dev/start-local.sh${NC}"
echo ""
echo "  3. Or run Python directly:"
echo "     ${GREEN}python -m src.main${NC}"
echo ""
echo "  4. Run tests:"
echo "     ${GREEN}pytest tests/${NC}"
echo ""
echo "Configuration:"
echo "  • Database: data/local.db"
echo "  • Logs: logs/"
echo "  • Config: config/local.yml"
echo "  • Env vars: .env.local"
echo ""
print_info "For full ML features, run: pip install -r requirements-full.txt"
echo ""

