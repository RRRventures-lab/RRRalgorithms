# 📱 Mobile Access Guide

Complete guide to accessing and controlling your trading system from iPhone/iPad.

---

## 🎯 Access Methods

### Option 1: Tailscale VPN (Recommended)

**Best for:** Access from anywhere in the world  
**Security:** ✅ Encrypted VPN tunnel  
**Setup time:** 15 minutes  
**Cost:** Free  

### Option 2: Home WiFi

**Best for:** Access only at home  
**Security:** ✅ Local network only  
**Setup time:** 0 minutes  
**Cost:** Free  

---

## 🔐 Setup: Tailscale (Worldwide Access)

### Mac Mini Setup

```bash
# Install Tailscale
brew install tailscale

# Start and authenticate
sudo tailscale up

# Opens browser - login with Google/GitHub account
# Approve the device
```

### iPhone Setup

1. **Install Tailscale app** from App Store
2. **Open app**, tap "Log In"
3. **Login** with same account as Mac Mini
4. **Done!** Mac Mini appears in device list

### Access Dashboard

**In Safari or Chrome on iPhone:**
```
http://mac-mini:8501
```

**Or use Tailscale hostname:**
```
http://mac-mini.tail-scale-domain.ts.net:8501
```

### Add to Home Screen (App-Like Experience)

1. Open dashboard in Safari
2. Tap **Share** button (square with arrow)
3. Scroll and tap **"Add to Home Screen"**
4. Name it **"RRR Trading"**
5. Tap **Add**

Now you have an app icon! Tap to access dashboard instantly. 📱✅

---

## 📊 Using the Dashboard

### Overview Tab

**Portfolio Metrics:**
- Large numbers for easy viewing
- Green/red for profit/loss
- Auto-updates every 5 seconds

**Quick Glance:**
- See portfolio value instantly
- Check daily P&L
- Monitor win rate

### Positions Tab

**All Open Positions:**
- Symbol and quantity
- Entry price vs current
- Unrealized P&L per position
- Color-coded profit/loss

**Touch to:**
- View position details
- See more information

### Trades Tab

**Recent Trade History:**
- Last 20 trades
- Buy/sell indicators
- P&L per trade
- Trade statistics

**Filter by:**
- Symbol
- Date range
- Profit/loss

### System Tab

**System Status:**
- Trading system: Running/Stopped
- Current machine: Mac Mini M4
- Storage: Lexar Drive
- Current time

**Controls:**
- ⏸️ **Pause Trading** button
- ▶️ **Resume Trading** button
- 🚨 **Emergency Stop** button (big, red)

**Configuration:**
- Max position size
- Daily loss limit
- Stop loss percentage
- Max concurrent positions

---

## 🔔 Telegram Alerts

### Setup

1. **Create bot:** Talk to @BotFather on Telegram
2. **Get token:** Copy token from BotFather
3. **Get chat ID:** Talk to @userinfobot
4. **Configure:** Add to `config/.env.lexar` on Lexar drive
5. **Restart:** System auto-starts bot

### Alerts You'll Receive

**Trade Executions:**
```
🤖 Trade Executed

Symbol: BTC-USD
Side: BUY
Quantity: 0.5
Price: $50,000.00
💚 P&L: +$250.00

Time: 14:32:15
```

**Daily Summary (8 AM):**
```
📊 Daily Summary - 2025-10-12

💰 Portfolio: $11,250.00
📈 Total P&L: +$1,250.00
📅 Today P&L: +$250.00
🎯 Win Rate: 62.5%

🔄 Trades Today: 8
Status: 🟢 RUNNING
```

**Risk Warnings:**
```
⚠️ Risk Warning

Position size approaching limit
BTC-USD: $2,100 (21% of portfolio)
Limit: $2,000 (20%)

Time: 16:45:22
```

**System Alerts:**
```
🚨 System Alert

Memory usage high: 78.5%

Time: 22:15:30
```

### Commands

Send these messages to your bot:

**Status & Info:**
- `/status` - Current portfolio and status
- `/trades` - Last 5 trades
- `/positions` - All open positions
- `/help` - List all commands

**Controls:**
- `/pause` - Pause trading (stops new orders)
- `/resume` - Resume trading
- `/stop` - Emergency stop (halt everything)

---

## 🎙️ Voice Control (Siri Shortcuts)

### Setup

1. **Open Shortcuts app** on iPhone
2. **Create new shortcut:**
   - Name: "Trading Status"
   - Action: "Get contents of URL"
   - URL: `http://mac-mini:8501/api/status` (if API built)
   - Action: "Show Notification" with result

3. **Add to Siri:**
   - Tap ... → Add to Siri
   - Record: "Hey Siri, trading status"

### Example Shortcuts

**Portfolio Value:**
```
"Hey Siri, what's my portfolio?"
→ Shows current value and P&L
```

**Pause Trading:**
```
"Hey Siri, pause RRR trading"
→ Sends pause command
→ Confirms with notification
```

**Emergency Stop:**
```
"Hey Siri, emergency stop trading"
→ Halts all trading
→ Sends alert
```

---

## 📈 Mobile Best Practices

### Dashboard Usage

**Quick Check (30 seconds):**
1. Open dashboard app
2. Glance at portfolio value
3. Check if green (profit) or red (loss)
4. Done!

**Detailed Review (5 minutes):**
1. Overview tab: Portfolio metrics
2. Positions tab: Check each position
3. Trades tab: Review recent trades
4. System tab: Verify system healthy

**Active Management (as needed):**
1. See risk warning → Review positions
2. See trade alert → Check if expected
3. See system alert → Check health
4. Take action if needed (pause/stop)

### Telegram Usage

**Enable Notifications:**
- Settings → Notifications → Telegram
- Allow: Notifications, Sounds, Badges
- Important: Show previews

**Customize:**
- Mute during sleep hours (optional)
- Create separate group for trading alerts
- Pin important messages

### Battery Management

**Dashboard usage:**
- Uses minimal data
- Mostly text and numbers
- Auto-refresh uses battery

**Tips:**
- Use WiFi when possible (less battery than cellular)
- Close dashboard when not actively monitoring
- Telegram alerts use almost no battery

---

## 🔒 Security Best Practices

### Tailscale

**Security:**
- ✅ End-to-end encrypted
- ✅ Zero-trust network
- ✅ No port forwarding needed
- ✅ Device-to-device authentication

**Best practices:**
- Don't share Tailscale account
- Enable two-factor auth on Tailscale account
- Review connected devices monthly
- Revoke old devices

### Dashboard

**No authentication by default** - Tailscale provides security

**Optional: Add password:**
```python
# In mobile_dashboard.py
import streamlit_authenticator as stauth

# Add authentication
authenticator = stauth.Authenticate(...)
name, authentication_status, username = authenticator.login('Login', 'main')

if not authentication_status:
    st.stop()
```

### Telegram

**Security:**
- ✅ Encrypted messaging
- ✅ Only your chat ID can send commands
- ✅ Bot token is secret

**Best practices:**
- Never share bot token
- Keep chat ID private
- Use Telegram's two-factor auth

---

## 📱 Mobile UI Tips

### Dashboard

**Touch Targets:**
- All buttons are large (3rem height)
- Spaced for fat-finger friendly
- Clear tap feedback

**Navigation:**
- Swipe left/right between tabs
- Pinch to zoom on charts
- Pull down to refresh (manual)

**Dark Mode:**
- Automatically matches iPhone setting
- Easy on eyes at night
- Saves battery

### Telegram

**Quick Commands:**
- Type `/` to see command list
- Use keyboard shortcuts
- Commands work from lock screen notifications

**Notifications:**
- Previews show key info
- Tap to see full details
- Swipe to dismiss

---

## 🎯 Daily Workflow Example

**Morning (2 minutes):**
1. Check Telegram for daily summary
2. Open dashboard app
3. Glance at portfolio value
4. Check if any positions need attention

**During Day (passive):**
- Receive trade alerts on phone
- Get risk warnings if needed
- System runs autonomously

**Evening (5 minutes):**
1. Open dashboard
2. Review day's trades
3. Check win rate
4. Verify system healthy
5. Review tomorrow's strategy (if active management)

**Weekly (15 minutes):**
1. Review weekly performance
2. Check system health
3. Review logs for errors
4. Adjust parameters if needed

---

## 🚀 Advanced Features

### Multiple Devices

Access from:
- iPhone (primary)
- iPad (larger screen)
- MacBook (when traveling)
- Any device on Tailscale

All see same dashboard, real-time!

### Widgets (iOS 14+)

Create Shortcuts widgets:
- Portfolio value on home screen
- Quick status check
- One-tap pause button

### Apple Watch

- Receive Telegram notifications
- Quick status via Siri
- Emergency stop from wrist

---

## ✅ You're Ready!

With this setup, you can:
- 📱 Monitor portfolio from anywhere
- 🔔 Receive instant trade alerts
- 🎛️ Control system remotely
- 📊 View detailed analytics
- 🚨 Emergency stop if needed
- 🗣️ Use voice commands
- 🌍 Access from anywhere (Tailscale)

**Your trading system is now in your pocket!** 🎉

---

**Next:** Read `TROUBLESHOOTING.md` for common issues and solutions.


