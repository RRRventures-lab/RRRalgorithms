#!/bin/bash
# Configure MCP servers for RRRalgorithms
# This script validates MCP configuration and ensures environment variables are set

set -e

echo "==================================="
echo "MCP Server Configuration"
echo "==================================="

cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)

# Check if .env file exists
if [ ! -f "config/api-keys/.env" ]; then
    echo "ERROR: config/api-keys/.env not found"
    echo "Please copy config/api-keys/.env.example to config/api-keys/.env"
    echo "and fill in your API keys"
    exit 1
fi

echo "✅ Found .env file"

# Load environment variables
source config/api-keys/.env

echo "✅ Loaded environment variables"

# Check MCP config file
if [ ! -f "config/mcp-servers/mcp-config.json" ]; then
    echo "ERROR: config/mcp-servers/mcp-config.json not found"
    exit 1
fi

echo "✅ Found MCP configuration"

# Validate required environment variables
REQUIRED_VARS=("SUPABASE_DB_URL" "GITHUB_TOKEN" "POLYGON_API_KEY" "PERPLEXITY_API_KEY" "COINBASE_API_KEY" "COINBASE_API_SECRET")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo "⚠️  WARNING: Missing environment variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Some MCP servers may not function correctly."
    echo "Please add these to config/api-keys/.env"
else
    echo "✅ All required environment variables are set"
fi

# Validate MCP server files exist
echo ""
echo "Checking MCP server implementations..."

if [ -f "worktrees/api-integration/coinbase/mcp_server.py" ]; then
    echo "✅ Coinbase MCP server found"
else
    echo "⚠️  Coinbase MCP server not found"
fi

if [ -f "worktrees/api-integration/polygon/mcp-server.js" ]; then
    echo "✅ Polygon MCP server found"
else
    echo "⚠️  Polygon MCP server not found"
fi

if [ -f "worktrees/api-integration/perplexity/mcp-server.js" ]; then
    echo "✅ Perplexity MCP server found"
else
    echo "⚠️  Perplexity MCP server not found"
fi

echo ""
echo "==================================="
echo "MCP Configuration Complete"
echo "==================================="
echo ""
echo "To use MCP servers with Claude Code:"
echo "1. Ensure your .env file has all required API keys"
echo "2. The mcp-config.json will use these environment variables"
echo "3. Restart Claude Code to load the MCP servers"
echo ""
echo "Project root: $PROJECT_ROOT"
