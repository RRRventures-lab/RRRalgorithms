#!/bin/bash
# Initialize PostgreSQL database with schema

set -e

echo "🗄️  Initializing RRRalgorithms Database..."
echo "=========================================="
echo ""

# Load environment variables
if [ -f "config/api-keys/.env" ]; then
    source config/api-keys/.env
else
    echo "❌ Error: config/api-keys/.env not found"
    echo "   Run: cp config/api-keys/.env.example config/api-keys/.env"
    exit 1
fi

# Check if PostgreSQL is running
if command -v docker &> /dev/null; then
    echo "📍 Checking Docker services..."
    if ! docker ps | grep -q rrr_postgres; then
        echo "⚠️  PostgreSQL container not running. Starting services..."
        docker-compose up -d postgres
        echo "   Waiting 10 seconds for PostgreSQL to start..."
        sleep 10
    fi
    echo "✅ PostgreSQL container running"
else
    echo "⚠️  Docker not found. Assuming PostgreSQL is running locally..."
fi

# Extract database connection details
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-trading_db}"
DB_USER="${DB_USER:-trading_user}"
DB_PASSWORD="${POSTGRES_PASSWORD:-secure_password}"

echo ""
echo "📋 Database Configuration:"
echo "   Host: $DB_HOST"
echo "   Port: $DB_PORT"
echo "   Database: $DB_NAME"
echo "   User: $DB_USER"
echo ""

# Test connection
echo "🔌 Testing database connection..."
export PGPASSWORD="$DB_PASSWORD"

if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c '\q' 2>/dev/null; then
    echo "✅ Database connection successful"
else
    echo "❌ Cannot connect to database"
    echo "   Make sure PostgreSQL is running and credentials are correct"
    exit 1
fi

# Create database if it doesn't exist
echo ""
echo "🏗️  Creating database (if not exists)..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1 || \
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "CREATE DATABASE $DB_NAME"

echo "✅ Database $DB_NAME ready"

# Run schema
echo ""
echo "📝 Applying database schema..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f config/database/schema.sql

if [ $? -eq 0 ]; then
    echo "✅ Schema applied successfully"
else
    echo "❌ Error applying schema"
    exit 1
fi

# Verify tables
echo ""
echo "🔍 Verifying tables..."
TABLE_COUNT=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
echo "   Found $TABLE_COUNT tables"

# List main tables
echo ""
echo "📊 Main tables:"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "\dt" | grep -E "crypto_|orders|positions|market_sentiment"

echo ""
echo "=========================================="
echo "✅ Database initialization complete!"
echo "=========================================="
echo ""
echo "📋 Next steps:"
echo "   1. Test connection with MCP:"
echo "      Use PostgreSQL MCP tool in Claude Code"
echo ""
echo "   2. Start data pipeline:"
echo "      cd worktrees/data-pipeline"
echo "      python -m src.data_pipeline.polygon.ingest"
echo ""
echo "   3. Query data:"
echo "      psql -h $DB_HOST -U $DB_USER -d $DB_NAME"
echo ""
