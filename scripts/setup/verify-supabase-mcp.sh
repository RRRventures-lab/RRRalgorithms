#!/bin/bash
# Quick script to verify Supabase MCP connection

set -a
source config/api-keys/.env
set +a

echo "Testing Supabase connection via psql..."
psql "$SUPABASE_DB_URL" -c "SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'public';"

echo ""
echo "Available tables:"
psql "$SUPABASE_DB_URL" -c "\dt"
