#!/bin/bash
# =============================================================================
# Environment Sanitization Script
# =============================================================================
# This script creates a sanitized version of .env files for sharing/committing
# Run this to create .env.example files with placeholder values
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}Environment File Sanitization${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# Function to sanitize a single env file
sanitize_env_file() {
    local source_file=$1
    local dest_file=$2

    if [ ! -f "$source_file" ]; then
        echo -e "${YELLOW}⚠️  Source file not found: $source_file${NC}"
        return 1
    fi

    echo -e "${BLUE}Processing: $source_file${NC}"

    # Create backup
    cp "$source_file" "${source_file}.backup.$(date +%Y%m%d_%H%M%S)"

    # Sanitize: replace actual values with placeholders
    cat "$source_file" | while IFS= read -r line; do
        # Skip empty lines and comments
        if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
            echo "$line"
        # Replace key=value with key=placeholder
        elif [[ "$line" =~ ^([A-Z_]+)=(.+)$ ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"

            # Determine appropriate placeholder based on key name
            if [[ "$key" =~ API_KEY$ ]]; then
                echo "${key}=your_${key,,}_here"
            elif [[ "$key" =~ SECRET$ ]]; then
                echo "${key}=your_${key,,}_here"
            elif [[ "$key" =~ PASSWORD$ ]]; then
                echo "${key}=your_${key,,}_here"
            elif [[ "$key" =~ TOKEN$ ]]; then
                echo "${key}=your_${key,,}_here"
            elif [[ "$key" =~ URL$ ]]; then
                echo "${key}=https://your-service-url-here"
            elif [[ "$key" =~ _KEY$ ]]; then
                echo "${key}=your_${key,,}_here"
            else
                # Keep non-sensitive values or use placeholder
                case "$key" in
                    ENVIRONMENT|LOG_LEVEL|DEBUG|PAPER_TRADING|LIVE_TRADING|ENABLE_*)
                        echo "$line"  # Keep as-is
                        ;;
                    MAX_*|CACHE_TTL|POLYGON_RATE_LIMIT)
                        echo "$line"  # Keep numeric configs
                        ;;
                    PROJECT_ROOT)
                        echo "${key}=/path/to/your/project"
                        ;;
                    *)
                        echo "${key}=your_${key,,}_value"
                        ;;
                esac
            fi
        else
            echo "$line"
        fi
    done > "$dest_file"

    echo -e "${GREEN}✅ Created: $dest_file${NC}"
    echo -e "${GREEN}✅ Backup: ${source_file}.backup.$(date +%Y%m%d_%H%M%S)${NC}"
}

# Main sanitization
if [ -f "config/api-keys/.env" ]; then
    sanitize_env_file "config/api-keys/.env" "config/api-keys/.env.example"
else
    echo -e "${YELLOW}⚠️  No .env file found at config/api-keys/.env${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Sanitization complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}IMPORTANT:${NC}"
echo -e "  - .env.example files are safe to commit"
echo -e "  - Original .env files are backed up"
echo -e "  - NEVER commit real .env files with actual credentials"
echo ""
