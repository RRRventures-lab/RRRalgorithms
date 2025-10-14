#!/bin/bash
# =============================================================================
# Secrets Detection Script
# =============================================================================
# Scans codebase for accidentally committed secrets/API keys
# Run before committing code
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}Scanning for potential secrets in code...${NC}"
echo ""

FOUND_ISSUES=0

# Pattern matching for common secret patterns
PATTERNS=(
    "api[_-]?key['\"]?\s*[:=]\s*['\"][a-zA-Z0-9]{20,}['\"]"
    "secret['\"]?\s*[:=]\s*['\"][a-zA-Z0-9]{20,}['\"]"
    "password['\"]?\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    "token['\"]?\s*[:=]\s*['\"][a-zA-Z0-9]{20,}['\"]"
    "-----BEGIN [A-Z]+ PRIVATE KEY-----"
    "AKIA[0-9A-Z]{16}"  # AWS Access Key
    "sk-[a-zA-Z0-9]{20,}"  # OpenAI/Anthropic keys
    "ghp_[a-zA-Z0-9]{36}"  # GitHub Personal Access Token
    "pplx-[a-zA-Z0-9]+"  # Perplexity keys
)

for pattern in "${PATTERNS[@]}"; do
    results=$(grep -r -n -E -i "$pattern" \
        --include="*.py" \
        --include="*.js" \
        --include="*.ts" \
        --include="*.yml" \
        --include="*.yaml" \
        --include="*.json" \
        --exclude-dir=".git" \
        --exclude-dir="venv" \
        --exclude-dir="node_modules" \
        --exclude-dir=".venv" \
        --exclude="*.pyc" \
        . 2>/dev/null || true)

    if [ ! -z "$results" ]; then
        echo -e "${RED}⚠️  Potential secret found matching pattern: $pattern${NC}"
        echo "$results"
        echo ""
        FOUND_ISSUES=$((FOUND_ISSUES + 1))
    fi
done

# Check for .env files in git
env_files=$(git ls-files | grep "\.env$" || true)
if [ ! -z "$env_files" ]; then
    echo -e "${RED}❌ ERROR: .env files found in git tracking:${NC}"
    echo "$env_files"
    echo ""
    FOUND_ISSUES=$((FOUND_ISSUES + 1))
fi

# Summary
if [ $FOUND_ISSUES -eq 0 ]; then
    echo -e "${GREEN}✅ No obvious secrets detected in code${NC}"
    exit 0
else
    echo -e "${RED}❌ Found $FOUND_ISSUES potential security issue(s)${NC}"
    echo -e "${YELLOW}Please review the findings above and remove any secrets${NC}"
    exit 1
fi
