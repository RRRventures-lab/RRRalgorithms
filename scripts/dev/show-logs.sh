#!/bin/bash
# =============================================================================
# RRRalgorithms - Show Logs
# Tail and filter log files
# =============================================================================

# Determine project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

LOG_DIR="$PROJECT_ROOT/logs"

# Check if logs directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "No logs directory found: $LOG_DIR"
    exit 1
fi

# Check for log files
LOG_FILES=$(find "$LOG_DIR" -name "*.log" 2>/dev/null)

if [ -z "$LOG_FILES" ]; then
    echo "No log files found in $LOG_DIR"
    exit 1
fi

# If specific log file requested
if [ $# -eq 1 ]; then
    LOG_FILE="$LOG_DIR/$1"
    if [ ! -f "$LOG_FILE" ]; then
        LOG_FILE="$LOG_DIR/$1.log"
    fi
    
    if [ -f "$LOG_FILE" ]; then
        echo "Tailing: $LOG_FILE"
        echo "Press Ctrl+C to stop"
        echo ""
        tail -f "$LOG_FILE"
    else
        echo "Log file not found: $1"
        echo ""
        echo "Available logs:"
        ls -1 "$LOG_DIR"/*.log 2>/dev/null | xargs -n1 basename
    fi
else
    # Show all logs
    echo "Available log files:"
    echo ""
    ls -lh "$LOG_DIR"/*.log 2>/dev/null
    echo ""
    echo "To tail a specific log:"
    echo "  $0 <log_file_name>"
    echo ""
    echo "To tail all logs:"
    echo "  tail -f $LOG_DIR/*.log"
fi

