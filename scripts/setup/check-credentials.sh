#!/bin/bash
# =============================================================================
# Credentials Check Script
# =============================================================================
# This script checks which API keys and credentials are configured
# and provides instructions for getting missing ones
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}API Credentials Status Check${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# Load .env file
if [ ! -f "config/api-keys/.env" ]; then
    echo -e "${RED}❌ Error: config/api-keys/.env not found${NC}"
    echo -e "${YELLOW}Please create it first${NC}"
    exit 1
fi

set -a
source config/api-keys/.env
set +a

# Function to check a credential
check_credential() {
    local name=$1
    local value=$2
    local placeholder=$3
    local instruction=$4

    if [ -z "$value" ] || [ "$value" = "$placeholder" ] || [[ "$value" == *"your_"* ]] || [[ "$value" == *"YOUR_"* ]]; then
        echo -e "${RED}❌ ${name}: Missing or placeholder${NC}"
        if [ ! -z "$instruction" ]; then
            echo -e "${YELLOW}   → ${instruction}${NC}"
        fi
        return 1
    else
        # Show masked value
        if [ ${#value} -gt 20 ]; then
            echo -e "${GREEN}✅ ${name}: ${value:0:20}...${NC}"
        else
            echo -e "${GREEN}✅ ${name}: Configured${NC}"
        fi
        return 0
    fi
}

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Critical APIs (Required for Trading System)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

CRITICAL_MISSING=0

check_credential "Polygon.io API Key" "$POLYGON_API_KEY" "" \
    "Get from: https://polygon.io/dashboard/api-keys"
CRITICAL_MISSING=$((CRITICAL_MISSING + $?))

check_credential "Supabase URL" "$SUPABASE_URL" "" \
    "Get from: https://supabase.com/dashboard/project/_/settings/api"
CRITICAL_MISSING=$((CRITICAL_MISSING + $?))

check_credential "Supabase Anon Key" "$SUPABASE_ANON_KEY" "" \
    "Get from: https://supabase.com/dashboard/project/_/settings/api"
CRITICAL_MISSING=$((CRITICAL_MISSING + $?))

check_credential "Supabase Service Key" "$SUPABASE_SERVICE_KEY" "your_service_role_key_here" \
    "Get from: https://supabase.com/dashboard/project/_/settings/api (look for 'service_role')"
CRITICAL_MISSING=$((CRITICAL_MISSING + $?))

check_credential "Supabase DB URL" "$SUPABASE_DB_URL" "" \
    "Update the password in DATABASE_URL. Get password from: https://supabase.com/dashboard/project/_/settings/database"
CRITICAL_MISSING=$((CRITICAL_MISSING + $?))

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}AI & Analysis APIs${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

AI_MISSING=0

check_credential "Perplexity API Key" "$PERPLEXITY_API_KEY" "" \
    "Get from: https://www.perplexity.ai/settings/api"
AI_MISSING=$((AI_MISSING + $?))

check_credential "Anthropic API Key" "$ANTHROPIC_API_KEY" "" \
    "Get from: https://console.anthropic.com/settings/keys"
AI_MISSING=$((AI_MISSING + $?))

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Optional APIs (Can configure later)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

check_credential "GitHub Token" "$GITHUB_TOKEN" "your_github_personal_access_token" \
    "Generate at: https://github.com/settings/tokens (scopes: repo, read:org)" || true

check_credential "TradingView Webhook Secret" "$TRADINGVIEW_WEBHOOK_SECRET" "your_secure_webhook_secret_here" \
    "Generate a random secure string: openssl rand -hex 32" || true

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ $CRITICAL_MISSING -eq 0 ]; then
    echo -e "${GREEN}✅ All critical credentials configured!${NC}"
    echo -e "${GREEN}You can now run: ./scripts/setup/init-supabase.sh${NC}"
else
    echo -e "${RED}❌ Missing ${CRITICAL_MISSING} critical credential(s)${NC}"
    echo -e "${YELLOW}Please add the missing credentials to config/api-keys/.env${NC}"
    echo -e ""
    echo -e "${BLUE}Quick Guide:${NC}"
    echo -e ""
    echo -e "${YELLOW}1. Supabase Service Role Key:${NC}"
    echo -e "   Go to your Supabase project settings:"
    if [ ! -z "$SUPABASE_URL" ]; then
        PROJECT_ID=$(echo "$SUPABASE_URL" | sed 's/https:\/\/\(.*\)\.supabase\.co/\1/')
        echo -e "   https://supabase.com/dashboard/project/${PROJECT_ID}/settings/api"
    else
        echo -e "   https://supabase.com/dashboard → Your Project → Settings → API"
    fi
    echo -e "   Look for the 'service_role' key (NOT the anon key)"
    echo -e "   Copy and paste it into .env as SUPABASE_SERVICE_KEY"
    echo -e ""
    echo -e "${YELLOW}2. Supabase Database Password:${NC}"
    if [ ! -z "$SUPABASE_URL" ]; then
        PROJECT_ID=$(echo "$SUPABASE_URL" | sed 's/https:\/\/\(.*\)\.supabase\.co/\1/')
        echo -e "   https://supabase.com/dashboard/project/${PROJECT_ID}/settings/database"
    else
        echo -e "   https://supabase.com/dashboard → Your Project → Settings → Database"
    fi
    echo -e "   Find your database password"
    echo -e "   Replace 'YOUR_DB_PASSWORD' in SUPABASE_DB_URL with the real password"
    echo -e ""
fi

if [ $AI_MISSING -gt 0 ]; then
    echo -e "${YELLOW}⚠️  ${AI_MISSING} AI API credential(s) missing (optional but recommended)${NC}"
fi

echo ""
echo -e "${BLUE}After configuring credentials, run:${NC}"
echo -e "  ./scripts/setup/check-credentials.sh  # Verify again"
echo -e "  ./scripts/setup/init-supabase.sh      # Initialize database"
echo ""
