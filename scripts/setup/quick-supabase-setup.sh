#!/bin/bash
# Quick interactive setup for Supabase credentials

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}Quick Supabase Credentials Setup${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# Load current .env
ENV_FILE="config/api-keys/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: $ENV_FILE not found${NC}"
    exit 1
fi

echo -e "${YELLOW}You need 2 pieces of information from your Supabase dashboard:${NC}"
echo ""

# Get Service Role Key
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}1. Get your Supabase Service Role Key${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "   Open this URL in your browser:"
echo -e "   ${GREEN}https://supabase.com/dashboard/project/isqznbvfmjmghxvctguh/settings/api${NC}"
echo ""
echo "   Steps:"
echo "   1. Scroll down to find the 'service_role' key"
echo "   2. Click 'Reveal' button"
echo "   3. Copy the ENTIRE key (starts with eyJ...)"
echo ""
echo -e "${YELLOW}Paste your service_role key here:${NC}"
read -r SERVICE_KEY

if [ -z "$SERVICE_KEY" ]; then
    echo -e "${RED}No key entered. Exiting.${NC}"
    exit 1
fi

echo ""

# Get Database Password
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}2. Get your Database Password${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "   Open this URL in your browser:"
echo -e "   ${GREEN}https://supabase.com/dashboard/project/isqznbvfmjmghxvctguh/settings/database${NC}"
echo ""
echo "   Steps:"
echo "   1. Look for 'Database password' section"
echo "   2. If you don't see it, click 'Reset Database Password'"
echo "   3. Copy the password"
echo ""
echo -e "${YELLOW}Paste your database password here:${NC}"
read -s DB_PASSWORD
echo ""

if [ -z "$DB_PASSWORD" ]; then
    echo -e "${RED}No password entered. Exiting.${NC}"
    exit 1
fi

echo ""

# Update .env file
echo -e "${BLUE}Updating $ENV_FILE...${NC}"

# Backup original
cp "$ENV_FILE" "${ENV_FILE}.backup"

# Update service key
sed -i '' "s|SUPABASE_SERVICE_KEY=.*|SUPABASE_SERVICE_KEY=${SERVICE_KEY}|g" "$ENV_FILE"

# Update database URLs
sed -i '' "s|YOUR_DB_PASSWORD|${DB_PASSWORD}|g" "$ENV_FILE"

echo -e "${GREEN}✅ Credentials updated!${NC}"
echo ""

# Verify
echo -e "${BLUE}Verifying credentials...${NC}"
./scripts/setup/check-credentials.sh

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Next step: Initialize the database${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "Run: ${GREEN}./scripts/setup/init-supabase.sh${NC}"
echo ""
