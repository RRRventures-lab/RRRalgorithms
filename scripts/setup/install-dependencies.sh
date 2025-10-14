#!/bin/bash
# Install all dependencies for RRRalgorithms trading system

set -e  # Exit on error

echo "🚀 RRRalgorithms - Dependency Installation"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect OS
OS="$(uname -s)"
echo "📍 Detected OS: $OS"
echo ""

# ============================================================================
# Check Prerequisites
# ============================================================================

echo "🔍 Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.9+${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✅ Python $PYTHON_VERSION${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found. Please install Node.js 18+${NC}"
    exit 1
fi
NODE_VERSION=$(node --version)
echo -e "${GREEN}✅ Node.js $NODE_VERSION${NC}"

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm not found. Please install npm${NC}"
    exit 1
fi
NPM_VERSION=$(npm --version)
echo -e "${GREEN}✅ npm $NPM_VERSION${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}⚠️  Docker not found. Install Docker for containerized services${NC}"
else
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
    echo -e "${GREEN}✅ Docker $DOCKER_VERSION${NC}"
fi

# Check git
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git not found. Please install git${NC}"
    exit 1
fi
GIT_VERSION=$(git --version | cut -d' ' -f3)
echo -e "${GREEN}✅ Git $GIT_VERSION${NC}"

echo ""

# ============================================================================
# Python Dependencies
# ============================================================================

echo "🐍 Installing Python dependencies..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "   Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "   Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install core Python packages
echo "   Installing core Python packages..."
cat > requirements.txt <<EOF
# Core Data Science
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0

# Deep Learning
torch>=2.0.0
transformers>=4.30.0
huggingface-hub>=0.16.0

# Reinforcement Learning
gymnasium>=0.28.0
stable-baselines3>=2.0.0

# Quantum Computing
qiskit>=0.43.0
pennylane>=0.31.0

# Data Pipeline
redis>=4.5.0
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0

# Time Series
ta-lib>=0.4.0  # Technical analysis library
statsmodels>=0.14.0

# API Clients
requests>=2.31.0
aiohttp>=3.8.0
websockets>=11.0
python-dotenv>=1.0.0

# FastAPI (for webhook server)
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0

# Trading Libraries
ccxt>=4.0.0  # Cryptocurrency exchange integration

# MLOps
mlflow>=2.5.0
optuna>=3.2.0  # Hyperparameter optimization

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# Logging and Monitoring
structlog>=23.1.0
prometheus-client>=0.17.0

# Utilities
python-dateutil>=2.8.0
pytz>=2023.3
tqdm>=4.65.0
EOF

pip install -r requirements.txt

echo -e "${GREEN}✅ Python dependencies installed${NC}"
echo ""

# ============================================================================
# Node.js Dependencies
# ============================================================================

echo "📦 Installing Node.js dependencies..."

# Initialize package.json if it doesn't exist
if [ ! -f "package.json" ]; then
    cat > package.json <<EOF
{
  "name": "rrralgorithms",
  "version": "0.1.0",
  "description": "Advanced cryptocurrency trading algorithm system",
  "type": "module",
  "scripts": {
    "build": "tsc",
    "test": "jest",
    "dev": "nodemon"
  },
  "keywords": ["trading", "cryptocurrency", "ml", "algorithmic-trading"],
  "author": "",
  "license": "PROPRIETARY",
  "dependencies": {
    "@anthropic-ai/sdk": "^0.27.0",
    "@modelcontextprotocol/sdk": "^1.0.0",
    "axios": "^1.6.0",
    "dotenv": "^16.3.0",
    "express": "^4.18.0",
    "ws": "^8.14.0"
  },
  "devDependencies": {
    "@types/node": "^20.8.0",
    "@types/express": "^4.17.0",
    "@types/ws": "^8.5.0",
    "typescript": "^5.2.0",
    "nodemon": "^3.0.0",
    "jest": "^29.7.0",
    "@types/jest": "^29.5.0"
  }
}
EOF
fi

npm install

# Install MCP servers globally
echo "   Installing MCP servers..."
npm install -g @modelcontextprotocol/server-postgres
npm install -g @modelcontextprotocol/server-github

echo -e "${GREEN}✅ Node.js dependencies installed${NC}"
echo ""

# ============================================================================
# Database Setup
# ============================================================================

echo "🗄️  Database setup..."

if command -v docker &> /dev/null; then
    echo "   Setting up PostgreSQL and TimescaleDB with Docker..."

    # Create docker-compose.yml
    cat > docker-compose.yml <<EOF
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg15
    container_name: rrr_postgres
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: \${POSTGRES_PASSWORD:-secure_password}
      POSTGRES_DB: trading_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: rrr_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: rrr_mlflow
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
    volumes:
      - mlflow_data:/mlflow
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: rrr_grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: \${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: rrr_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  mlflow_data:
  grafana_data:
  prometheus_data:
EOF

    echo "   Starting Docker services..."
    docker-compose up -d

    echo -e "${GREEN}✅ Database services started${NC}"
else
    echo -e "${YELLOW}⚠️  Docker not available. Please install PostgreSQL, Redis, and MLflow manually.${NC}"
fi

echo ""

# ============================================================================
# Configuration Files
# ============================================================================

echo "⚙️  Setting up configuration..."

# Copy .env.example to .env if it doesn't exist
if [ ! -f "config/api-keys/.env" ]; then
    echo "   Creating .env file from template..."
    cp config/api-keys/.env.example config/api-keys/.env
    echo -e "${YELLOW}   ⚠️  Please edit config/api-keys/.env with your API keys${NC}"
else
    echo "   .env file already exists"
fi

# Set PROJECT_ROOT in .env
if [ -f "config/api-keys/.env" ]; then
    if ! grep -q "^PROJECT_ROOT=" config/api-keys/.env; then
        echo "PROJECT_ROOT=$(pwd)" >> config/api-keys/.env
        echo "   Added PROJECT_ROOT to .env"
    fi
fi

echo -e "${GREEN}✅ Configuration complete${NC}"
echo ""

# ============================================================================
# MCP Server Configuration
# ============================================================================

echo "🔌 Configuring MCP servers..."

# Update MCP config with PROJECT_ROOT
PROJECT_ROOT=$(pwd)
sed -i.bak "s|\${PROJECT_ROOT}|$PROJECT_ROOT|g" config/mcp-servers/mcp-config.json
rm -f config/mcp-servers/mcp-config.json.bak

echo -e "${GREEN}✅ MCP servers configured${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Edit config/api-keys/.env with your API keys:"
echo "   - POLYGON_API_KEY"
echo "   - PERPLEXITY_API_KEY"
echo "   - TRADINGVIEW_WEBHOOK_SECRET"
echo ""
echo "2. Verify database is running:"
echo "   docker-compose ps"
echo ""
echo "3. Run database migrations:"
echo "   ./scripts/setup/init-databases.sh"
echo ""
echo "4. Create worktrees for parallel development:"
echo "   ./scripts/setup/create-worktrees.sh"
echo ""
echo "5. Start developing in your chosen worktree:"
echo "   cd worktrees/neural-network"
echo ""
echo "6. Read the documentation:"
echo "   - claude.md (workflow guide)"
echo "   - docs/worktrees/WORKTREE_ARCHITECTURE.md"
echo "   - docs/api-specs/API_MCP_INTEGRATION.md"
echo ""
echo "🎉 Happy trading!"
echo ""
