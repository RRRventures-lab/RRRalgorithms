#!/bin/bash
#
# Transparency Dashboard API Startup Script
# Starts the FastAPI backend server for the transparency dashboard
#

set -e

echo "=================================================="
echo "  RRRalgorithms Transparency Dashboard API"
echo "=================================================="
echo ""

# Check if running from correct directory
if [ ! -f "src/api/main.py" ]; then
    echo "Error: Must run from project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import fastapi" 2>/dev/null || {
    echo "Installing FastAPI..."
    pip install -q fastapi uvicorn
}

# Check if database exists
DB_PATH="./data/transparency.db"
if [ ! -f "$DB_PATH" ]; then
    echo "Database not found. Creating transparency database..."
    python3 scripts/migrate_transparency_schema.py --db-path "$DB_PATH"
fi

# Start the API server
echo ""
echo "Starting API server..."
echo "  - API URL: http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo "  - Health: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
