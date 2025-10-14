# Production Deployment

This directory contains configuration files for production/cloud deployment.

## ⚠️ Not for Local Development

For local development, use the native Python setup in the project root:
```bash
./scripts/setup/setup-local.sh
./scripts/dev/start-local.sh
```

## Production Deployment Options

### Docker Compose (Single Server)

For deploying on a single server or VM:

```bash
cd deployment
docker-compose up -d
```

### Docker Compose - Paper Trading

For extended paper trading validation before live trading:

```bash
cd deployment
docker-compose -f docker-compose.paper-trading.yml up -d
```

## Files in This Directory

- `docker-compose.yml` - Full production stack with all services
- `docker-compose.paper-trading.yml` - Paper trading configuration
- `Dockerfiles/` - Individual service Dockerfiles (if separated)

## Requirements

- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 20GB+ disk space

## Configuration

Production configuration is in `config/production.yml` at the project root.

Required environment variables should be set in `.env` file (not tracked in git):

```bash
# Database
DATABASE_HOST=your-db-host
DATABASE_NAME=trading_db
DATABASE_USER=trading_user
DATABASE_PASSWORD=your-secure-password

# Redis
REDIS_HOST=your-redis-host
REDIS_PASSWORD=your-redis-password

# Trading
COINBASE_API_KEY=your-api-key
COINBASE_API_SECRET=your-api-secret
POLYGON_API_KEY=your-polygon-key

# Monitoring
GRAFANA_PASSWORD=your-secure-password
```

## Switching from Local to Production

If you've been developing locally and want to deploy to production:

1. Ensure all tests pass: `pytest tests/`
2. Review production configuration: `config/production.yml`
3. Set required environment variables (see above)
4. Build and start services: `docker-compose up -d`
5. Monitor logs: `docker-compose logs -f`
6. Access Grafana: http://your-server:3000

## Security Notes

- Never commit API keys or passwords
- Use secrets management (HashiCorp Vault, AWS Secrets Manager, etc.)
- Enable SSL/TLS for all external connections
- Rotate API keys regularly
- Start with paper trading for at least 30 days
- Monitor system health 24/7

## Support

For production deployment assistance, refer to:
- `docs/deployment/PRODUCTION.md`
- `DEPLOYMENT_STATUS_REPORT.md`

