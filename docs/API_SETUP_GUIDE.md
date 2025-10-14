# 📡 API Setup Guide

This guide will help you set up the required API connections for the RRRalgorithms trading system.

---

## 🔑 Required API Keys

### 1. Polygon.io (Market Data) - REQUIRED

Polygon provides real-time and historical market data for cryptocurrencies.

**Get your API key:**
1. Sign up at [Polygon.io](https://polygon.io/)
2. Go to [Dashboard > API Keys](https://polygon.io/dashboard/api-keys)
3. Copy your default API key

**Subscription Levels:**
- **Free Tier**: Limited to 5 API calls/minute, delayed data
- **Starter ($29/mo)**: 10,000 API calls/day, real-time data
- **Developer ($99/mo)**: Unlimited API calls, WebSocket access
- **Professional ($399/mo)**: Full features, priority support

**Recommended**: Start with Free tier for testing, upgrade to Starter for paper trading

---

### 2. TradingView (Alerts) - OPTIONAL

TradingView integration for technical analysis alerts.

**Setup:**
1. Create a [TradingView](https://www.tradingview.com/) account
2. Set up webhook alerts pointing to your server
3. Generate a webhook secret for authentication

**Note**: Requires Pro subscription for webhook alerts

---

### 3. Perplexity AI (Sentiment) - OPTIONAL

AI-powered market sentiment analysis.

**Get your API key:**
1. Sign up at [Perplexity AI](https://www.perplexity.ai/)
2. Go to Settings > API
3. Generate an API key

**Pricing**: Pay-per-use, approximately $0.001 per query

---

### 4. Telegram Bot (Alerts) - OPTIONAL

Mobile push notifications for trades and alerts.

**Setup:**
1. Open Telegram and search for @BotFather
2. Send `/newbot` and follow instructions
3. Save the bot token
4. Get your chat ID:
   - Send a message to your bot
   - Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Find your chat_id in the response

---

## ⚙️ Configuration Steps

### Step 1: Create Environment File

```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
cp config/api-keys/.env.template config/api-keys/.env
```

### Step 2: Edit Configuration

```bash
nano config/api-keys/.env
```

Add your API keys:
```env
POLYGON_API_KEY=pk_abcd1234567890
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234567890
# ... etc
```

### Step 3: Test Connections

```bash
# Test Polygon connection
python test_polygon_connection.py

# Test all APIs
python scripts/test_all_apis.py
```

---

## 🧪 Testing Without API Keys

You can test the system without API keys using simulated data:

```bash
# Run with mock data
python src/main.py --use-mock-data

# Run dashboard with demo data
streamlit run src/dashboards/mobile_dashboard.py
```

---

## 🔒 Security Best Practices

1. **Never commit API keys to git**
   - The `.env` file is gitignored
   - Use `.env.template` for examples only

2. **Use environment variables**
   ```python
   import os
   api_key = os.getenv('POLYGON_API_KEY')
   ```

3. **Rotate keys regularly**
   - Change API keys every 90 days
   - Immediately rotate if exposed

4. **Use minimal permissions**
   - Only enable required API endpoints
   - Use read-only keys where possible

5. **Monitor usage**
   - Check API usage dashboards regularly
   - Set up billing alerts

---

## 📊 API Rate Limits

| API | Free Tier | Paid Tier | Our Usage |
|-----|-----------|-----------|-----------|
| Polygon | 5/minute | Unlimited | ~100/minute |
| TradingView | N/A | 1000/day | ~50/day |
| Perplexity | 100/day | Pay-per-use | ~20/day |
| Telegram | 30/second | 30/second | ~1/minute |

**Note**: The system includes automatic rate limiting to prevent exceeding limits.

---

## 🚀 Quick Start

### Minimal Setup (Paper Trading)
1. Get free Polygon.io API key
2. Add to `.env` file
3. Run: `python src/main.py --paper-trading`

### Full Setup (Production)
1. Get all API keys (Polygon Developer tier minimum)
2. Configure `.env` with all keys
3. Set up Telegram bot for alerts
4. Run: `./scripts/launch.sh`

---

## ❓ Troubleshooting

### "API key not found"
- Check `.env` file exists in `config/api-keys/`
- Verify key name matches exactly (case-sensitive)
- Restart application after adding keys

### "Rate limit exceeded"
- Upgrade API subscription
- Reduce request frequency in settings
- Check rate limiter configuration

### "Connection failed"
- Verify internet connection
- Check API service status
- Confirm API key is valid and active

---

## 📞 Support

- **Polygon.io**: [support.polygon.io](https://support.polygon.io/)
- **TradingView**: [www.tradingview.com/support/](https://www.tradingview.com/support/)
- **Telegram**: [@BotSupport](https://t.me/BotSupport)

---

**Last Updated**: 2025-10-12