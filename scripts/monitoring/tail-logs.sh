#!/bin/bash
# Tail logs from Docker services for RRRalgorithms
# Usage: ./tail-logs.sh [service_name] [lines]
#   service_name: Optional, specific service to tail (default: all)
#   lines: Optional, number of lines to show (default: 100)

set -e

cd "$(dirname "$0")/../.."

SERVICE="${1:-}"
LINES="${2:-100}"

echo "==================================="
echo "RRRalgorithms Log Monitor"
echo "==================================="

if [ -z "$SERVICE" ]; then
    echo "Tailing logs from ALL services (last $LINES lines)..."
    echo "Press Ctrl+C to stop"
    echo ""
    docker-compose -f docker-compose.yml logs -f --tail="$LINES"
else
    echo "Tailing logs from: $SERVICE (last $LINES lines)..."
    echo "Press Ctrl+C to stop"
    echo ""
    docker-compose -f docker-compose.yml logs -f --tail="$LINES" "$SERVICE"
fi
