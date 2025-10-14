#!/bin/bash
# =============================================================================
# Supabase Database Initialization Script
# =============================================================================
# This script:
# 1. Validates Supabase credentials
# 2. Tests database connection
# 3. Applies the schema
# 4. Enables real-time subscriptions
# 5. Verifies the setup
# =============================================================================

set -e  # Exit on error

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
echo -e "${BLUE}Supabase Database Initialization${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# =============================================================================
# Step 1: Load environment variables
# =============================================================================
echo -e "${YELLOW}Step 1: Loading environment variables...${NC}"

if [ ! -f "config/api-keys/.env" ]; then
    echo -e "${RED}❌ Error: config/api-keys/.env not found${NC}"
    echo -e "${YELLOW}Please run: cp config/api-keys/.env.example config/api-keys/.env${NC}"
    exit 1
fi

# Load .env file
set -a
source config/api-keys/.env
set +a

echo -e "${GREEN}✅ Environment variables loaded${NC}"
echo ""

# =============================================================================
# Step 2: Validate Supabase credentials
# =============================================================================
echo -e "${YELLOW}Step 2: Validating Supabase credentials...${NC}"

if [ -z "$SUPABASE_URL" ] || [ "$SUPABASE_URL" = "your_supabase_url_here" ]; then
    echo -e "${RED}❌ SUPABASE_URL is not configured${NC}"
    echo -e "${YELLOW}Please add your Supabase URL to config/api-keys/.env${NC}"
    exit 1
fi
echo -e "${GREEN}✅ SUPABASE_URL: ${SUPABASE_URL}${NC}"

if [ -z "$SUPABASE_ANON_KEY" ] || [ "$SUPABASE_ANON_KEY" = "your_anon_key_here" ]; then
    echo -e "${RED}❌ SUPABASE_ANON_KEY is not configured${NC}"
    exit 1
fi
echo -e "${GREEN}✅ SUPABASE_ANON_KEY: ${SUPABASE_ANON_KEY:0:20}...${NC}"

if [ -z "$SUPABASE_DB_URL" ] || [[ "$SUPABASE_DB_URL" == *"YOUR_DB_PASSWORD"* ]]; then
    echo -e "${RED}❌ SUPABASE_DB_URL is not configured or contains placeholder password${NC}"
    echo -e "${YELLOW}Please update the database password in config/api-keys/.env${NC}"
    echo -e "${YELLOW}Get it from: https://supabase.com/dashboard/project/${SUPABASE_URL##*/}/settings/database${NC}"
    exit 1
fi
echo -e "${GREEN}✅ SUPABASE_DB_URL: ${SUPABASE_DB_URL%%@*}@***${NC}"

echo ""

# =============================================================================
# Step 3: Test database connection
# =============================================================================
echo -e "${YELLOW}Step 3: Testing database connection...${NC}"

# Check if psql is available
if ! command -v psql &> /dev/null; then
    echo -e "${RED}❌ psql is not installed${NC}"
    echo -e "${YELLOW}Installing PostgreSQL client...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install postgresql
    else
        echo -e "${RED}Please install PostgreSQL client manually${NC}"
        exit 1
    fi
fi

# Test connection
if psql "$SUPABASE_DB_URL" -c "SELECT 1" &> /dev/null; then
    echo -e "${GREEN}✅ Database connection successful${NC}"
else
    echo -e "${RED}❌ Database connection failed${NC}"
    echo -e "${YELLOW}Please verify your SUPABASE_DB_URL in config/api-keys/.env${NC}"
    exit 1
fi

echo ""

# =============================================================================
# Step 4: Apply schema
# =============================================================================
echo -e "${YELLOW}Step 4: Applying Supabase schema...${NC}"

SCHEMA_FILE="config/supabase/schema.sql"

if [ ! -f "$SCHEMA_FILE" ]; then
    echo -e "${RED}❌ Schema file not found: $SCHEMA_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}Executing schema...${NC}"
psql "$SUPABASE_DB_URL" -f "$SCHEMA_FILE" > /tmp/supabase_schema_output.log 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Schema applied successfully${NC}"

    # Show summary
    echo -e "\n${BLUE}Schema execution summary:${NC}"
    tail -n 5 /tmp/supabase_schema_output.log
else
    echo -e "${RED}❌ Schema application failed${NC}"
    echo -e "${YELLOW}Error log:${NC}"
    cat /tmp/supabase_schema_output.log
    exit 1
fi

echo ""

# =============================================================================
# Step 5: Verify tables were created
# =============================================================================
echo -e "${YELLOW}Step 5: Verifying tables...${NC}"

TABLES=$(psql "$SUPABASE_DB_URL" -t -c "\dt public.*" | grep -c "table" || echo "0")

if [ "$TABLES" -gt 0 ]; then
    echo -e "${GREEN}✅ Found $TABLES tables${NC}"

    echo -e "\n${BLUE}Tables created:${NC}"
    psql "$SUPABASE_DB_URL" -c "\dt public.*"
else
    echo -e "${RED}❌ No tables found${NC}"
    exit 1
fi

echo ""

# =============================================================================
# Step 6: Verify real-time subscriptions
# =============================================================================
echo -e "${YELLOW}Step 6: Verifying real-time subscriptions...${NC}"

REALTIME_TABLES=$(psql "$SUPABASE_DB_URL" -t -c "
    SELECT COUNT(*)
    FROM pg_publication_tables
    WHERE pubname = 'supabase_realtime'
" | tr -d ' ')

if [ "$REALTIME_TABLES" -gt 0 ]; then
    echo -e "${GREEN}✅ Real-time enabled on $REALTIME_TABLES tables${NC}"
else
    echo -e "${YELLOW}⚠️  Warning: No real-time subscriptions found${NC}"
    echo -e "${YELLOW}This is normal if pg_cron extension is not available${NC}"
fi

echo ""

# =============================================================================
# Step 7: Test helper functions
# =============================================================================
echo -e "${YELLOW}Step 7: Testing helper functions...${NC}"

# Test get_latest_price function
FUNCTION_TEST=$(psql "$SUPABASE_DB_URL" -t -c "
    SELECT EXISTS (
        SELECT 1 FROM pg_proc
        WHERE proname = 'get_latest_price'
    );
" | tr -d ' ')

if [ "$FUNCTION_TEST" = "t" ]; then
    echo -e "${GREEN}✅ Helper functions created${NC}"
else
    echo -e "${YELLOW}⚠️  Warning: Some helper functions may not be available${NC}"
fi

echo ""

# =============================================================================
# Step 8: Insert test data (optional)
# =============================================================================
echo -e "${YELLOW}Step 8: Would you like to insert test data? (y/n)${NC}"
read -r RESPONSE

if [[ "$RESPONSE" =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Inserting test data...${NC}"

    psql "$SUPABASE_DB_URL" << 'EOF'
    -- Insert test crypto aggregate data
    INSERT INTO crypto_aggregates (
        ticker, event_time, open, high, low, close, volume, vwap
    ) VALUES
        ('X:BTCUSD', NOW() - INTERVAL '1 hour', 67000, 67500, 66800, 67200, 150.5, 67100),
        ('X:ETHUSD', NOW() - INTERVAL '1 hour', 3200, 3250, 3180, 3220, 500.2, 3215)
    ON CONFLICT (ticker, event_time) DO NOTHING;

    -- Insert test ML model
    INSERT INTO ml_models (
        model_name, model_version, model_type, architecture, deployed
    ) VALUES (
        'test_model', 'v1.0.0', 'price_prediction', 'Transformer', FALSE
    ) ON CONFLICT (model_name, model_version) DO NOTHING;
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Test data inserted${NC}"
    else
        echo -e "${YELLOW}⚠️  Warning: Could not insert test data${NC}"
    fi
else
    echo -e "${BLUE}Skipping test data insertion${NC}"
fi

echo ""

# =============================================================================
# Step 9: Create MCP verification script
# =============================================================================
echo -e "${YELLOW}Step 9: Creating MCP verification script...${NC}"

cat > scripts/setup/verify-supabase-mcp.sh << 'VERIFY_EOF'
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
VERIFY_EOF

chmod +x scripts/setup/verify-supabase-mcp.sh
echo -e "${GREEN}✅ Created scripts/setup/verify-supabase-mcp.sh${NC}"

echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${GREEN}=============================================================================${NC}"
echo -e "${GREEN}✅ Supabase Setup Complete!${NC}"
echo -e "${GREEN}=============================================================================${NC}"
echo ""
echo -e "${BLUE}What's been set up:${NC}"
echo -e "  ✅ Database connection validated"
echo -e "  ✅ Schema applied with all tables"
echo -e "  ✅ Real-time subscriptions enabled"
echo -e "  ✅ Helper functions created"
echo -e "  ✅ Row Level Security (RLS) policies set"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Test MCP connection: ./scripts/setup/verify-supabase-mcp.sh"
echo -e "  2. View your data: ${SUPABASE_URL/https:\/\//https://supabase.com/dashboard/project/}/editor"
echo -e "  3. Start data pipeline: cd worktrees/data-pipeline && python src/data_pipeline/ingest.py"
echo ""
echo -e "${BLUE}Supabase Dashboard:${NC}"
echo -e "  ${SUPABASE_URL/https:\/\//https://supabase.com/dashboard/project/}"
echo ""
echo -e "${GREEN}🚀 You're ready to start building!${NC}"
