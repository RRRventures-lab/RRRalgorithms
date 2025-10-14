# Mac Mini 24/7 Deployment Guide

**Target:** Mac Mini M4 256GB  
**Storage:** Lexar 2TB (portable), 2TB Dock (backups)  
**Access:** iPhone/iPad via Tailscale  
**Mode:** 24/7 autonomous operation  

---

## 🎯 Overview

This guide walks you through deploying RRRalgorithms on Mac Mini M4 for 24/7 autonomous trading with full mobile control from your iPhone.

**Time Required:** 30-60 minutes  
**Difficulty:** Easy (all automated)  

---

## 📋 Pre-Transfer Checklist (On M3 MacBook)

### 1. Verify Everything is on Lexar

```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
./scripts/prepare_for_mac_mini.sh
```

This script checks:
- ✅ All code on Lexar
- ✅ Virtual environment configured
- ✅ Dependencies installed
- ✅ Launch scripts ready
- ✅ Dashboard working
- ✅ Permissions correct

### 2. Optional: Test on MacBook First

```bash
# Test dashboard
./scripts/launch.sh --dashboard

# Access on iPhone (same WiFi)
# Open Safari: http://macbook-name.local:8501
# Add to home screen
```

### 3. Safely Eject

```bash
# macOS will handle this
# Right-click Lexar → Eject
```

**✅ Ready to transfer!**

---

## 🚀 Mac Mini Setup (One-Time, 30-60 min)

### Step 1: Physical Setup

1. **Connect Lexar 2TB** to Mac Mini USB-C port
2. **Connect 2TB Dock** to second USB-C port (for backups)
3. **Connect Ethernet** (more reliable than WiFi)
4. **Power on** Mac Mini

### Step 2: macOS Initial Setup

1. **Language & Region:** Your preferences
2. **User Account:** Create admin account
3. **Apple ID:** Sign in (for App Store, iCloud)
4. **Siri:** Enable (for voice control later)
5. **FileVault:** Enable (for security)

### Step 3: Verify Lexar Mounted

```bash
# Open Terminal
ls -la /Volumes/Lexar/RRRVentures/RRRalgorithms

# Should see your project files
```

### Step 4: Run First-Boot Setup

```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
./scripts/mac_mini_first_boot.sh
```

This automated script:
- ✅ Installs Homebrew
- ✅ Installs Python 3.11+
- ✅ Installs Tailscale
- ✅ Creates virtual environment  
- ✅ Initializes database
- ✅ Sets up auto-start on boot
- ✅ Configures 24/7 settings
- ✅ Tests the system

**Takes:** 15-30 minutes (mostly downloads)

### Step 5: Configure Tailscale

```bash
# Start Tailscale
sudo tailscale up

# Opens browser - login with Google/GitHub
# Approve the Mac Mini device
```

**On iPhone:**
1. Install **Tailscale** from App Store
2. Login with same account
3. Now Mac Mini appears in your Tailscale network!

### Step 6: Start Trading System

```bash
# Manual start (for testing)
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
./scripts/launch.sh
```

**Or let it auto-start:**
```bash
# System will start automatically on next boot
# Or trigger now:
launchctl start com.rrrventures.trading
```

### Step 7: Access from iPhone

**Find Mac Mini hostname:**
```bash
# On Mac Mini
hostname
# e.g., "mac-mini.local" or Tailscale name
```

**On iPhone:**
1. Open Safari or Chrome
2. Go to: `http://mac-mini:8501`
3. Tap Share → Add to Home Screen
4. Name it "RRR Trading"
5. Now it looks like a native app! 📱

---

## 📱 Mobile Access Methods

### Method 1: Tailscale (Recommended - Worldwide Access)

**Security:** ✅ Encrypted VPN tunnel  
**Range:** 🌍 Anywhere in the world  
**Speed:** ⚡ Fast  
**Setup:** 15 minutes  

**Steps:**
1. Mac Mini & iPhone both on Tailscale
2. Access: `http://mac-mini:8501`
3. Works from anywhere!

### Method 2: Local Network (Home Only)

**Security:** ✅ Local network only  
**Range:** 🏠 Same WiFi as Mac Mini  
**Speed:** ⚡⚡ Fastest  
**Setup:** 0 minutes  

**Steps:**
1. Find Mac Mini IP: System Settings → Network
2. Access: `http://192.168.1.x:8501`
3. Works only at home

### Method 3: Port Forwarding (Not Recommended)

**Security:** ⚠️ Exposes to internet  
**Range:** 🌍 Anywhere  
**Setup:** Complex  

**Skip this** - use Tailscale instead for security

---

## 🔔 Telegram Alerts Setup (Optional, 15 min)

### Step 1: Create Bot

1. Open Telegram app
2. Search for "@BotFather"
3. Send: `/newbot`
4. Follow prompts, choose name
5. **Copy the token** (looks like `123456:ABC-DEF...`)

### Step 2: Get Your Chat ID

1. Search for "@userinfobot"
2. Send: `/start`
3. **Copy your ID** (looks like `123456789`)

### Step 3: Configure

On Mac Mini, create/edit:
```bash
# /Volumes/Lexar/RRRVentures/RRRalgorithms/config/.env.lexar
TELEGRAM_BOT_TOKEN=123456:ABC-DEF-your-token-here
TELEGRAM_CHAT_ID=123456789
```

### Step 4: Test

```bash
# Send test message
python -m src.monitoring.telegram_alerts
```

You should receive: "🚀 Trading System Started"

### Step 5: Enable Auto-Start

Telegram bot will start automatically with trading system.

**Commands you can send:**
- `/status` - Portfolio value
- `/trades` - Recent trades
- `/positions` - Open positions
- `/pause` - Pause trading
- `/resume` - Resume trading
- `/stop` - Emergency stop

---

## ⚙️ System Preferences (Mac Mini)

### Energy Saver
```
System Settings → Energy Saver
✅ Prevent automatic sleeping: ON
✅ Wake for network access: ON
✅ Start up automatically after power failure: ON
```

### User & Groups
```
System Settings → Users & Groups
✅ Login Options → Automatic login: Your user (optional)
```

### Software Update
```
System Settings → Software Update
⚠️ Automatic updates: Consider DISABLING
   (Updates might restart Mac Mini unexpectedly)
   Or: Check "Install security updates only"
```

---

## 🔍 Monitoring & Maintenance

### Check System Status

**From iPhone (via Tailscale):**
- Open: `http://mac-mini:8501`
- View: Portfolio, trades, positions, system status

**Via Telegram:**
- Send: `/status`
- Receive: Portfolio value, P&L, system status

**Direct SSH (if needed):**
```bash
# From any device on Tailscale
ssh your-user@mac-mini

# Check if running
launchctl list | grep rrrventures

# View logs
tail -f /Volumes/Lexar/RRRVentures/RRRalgorithms/logs/system/launchd.log
```

### Daily Routine

**Morning:**
- Check Telegram for daily summary (auto-sent at 8 AM)
- Review dashboard on phone
- Check for any alerts

**Throughout Day:**
- Receive trade alerts on phone
- Monitor P&L
- Check for risk warnings

**Evening:**
- Review day's performance
- Check system health
- Verify backups running

---

## 🛠️ Troubleshooting

### Dashboard Not Accessible

**Check 1: Is system running?**
```bash
# SSH to Mac Mini
launchctl list | grep rrrventures

# Should show PID if running
```

**Check 2: Is Streamlit running?**
```bash
ps aux | grep streamlit
# Should show process
```

**Check 3: Firewall?**
```bash
# macOS Firewall settings
System Settings → Network → Firewall
# Allow incoming connections to Python
```

**Check 4: Tailscale connected?**
```bash
tailscale status
# Should show "Connected"
```

### Trading System Not Starting

**Check logs:**
```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
tail -100 logs/system/launchd.log
tail -100 logs/system/launchd.err
```

**Common issues:**
1. Lexar drive not mounted → Wait for mount, check cable
2. Python error → Check venv activated
3. Database locked → Restart system

**Manual restart:**
```bash
launchctl stop com.rrrventures.trading
launchctl start com.rrrventures.trading
```

### Telegram Not Working

**Check configuration:**
```bash
cat config/.env.lexar | grep TELEGRAM
# Should show token and chat ID
```

**Test bot:**
```bash
python -m src.monitoring.telegram_alerts
# Should send test message
```

**Common issues:**
1. Wrong token → Check @BotFather
2. Wrong chat ID → Check @userinfobot
3. Network issue → Check internet connection

### System Performance Issues

**Check memory:**
```bash
# SSH to Mac Mini
htop
# or
top -o mem
```

**Check disk:**
```bash
df -h /Volumes/Lexar
```

**Check logs for errors:**
```bash
tail -f logs/system/launchd.err
```

---

## 🔄 Updates & Maintenance

### Updating Code

**On MacBook (development):**
1. Make changes
2. Test locally
3. Commit to git (optional)
4. Sync to Lexar

**Transfer to Mac Mini:**
```bash
# Option 1: Physical transfer (safest)
1. Eject Lexar from MacBook
2. Connect to Mac Mini
3. System auto-updates

# Option 2: rsync over Tailscale
rsync -av --exclude 'venv' --exclude '*.pyc' \
  /path/to/local/RRRalgorithms/ \
  mac-mini:/Volumes/Lexar/RRRVentures/RRRalgorithms/

# Restart system
ssh mac-mini "launchctl stop com.rrrventures.trading"
```

### Updating Dependencies

```bash
# SSH to Mac Mini
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
source venv/bin/activate
pip install --upgrade -r requirements-local.txt

# Restart
launchctl stop com.rrrventures.trading
```

### Database Maintenance

**Backup database:**
```bash
# Automatic backups to 2TB Dock every 6 hours
# Manual backup:
cp data/database/local.db /Volumes/2TB-Dock/backups/db_$(date +%Y%m%d).db
```

**Clean old logs:**
```bash
# Compress logs older than 30 days
find logs/ -name "*.log" -mtime +30 -exec gzip {} \;

# Move to archive
mv logs/trading/*.gz logs/archive/
```

---

## 📊 Monitoring Dashboard Features

### Overview Tab
- 💰 Portfolio value (big number)
- 📈 Total P&L
- 📅 Daily P&L
- 🎯 Win rate
- 💵 Cash position
- 📊 Position value

### Positions Tab
- List of all open positions
- Quantity, average price
- Current price, unrealized P&L
- Mobile-optimized table

### Trades Tab
- Recent trade history
- Buy/sell indicators
- P&L per trade
- Trade statistics

### System Tab
- 🟢 System status (running/stopped)
- ⏸️ Pause trading button
- ▶️ Resume trading button
- 🚨 Emergency stop button
- Configuration display

**Auto-refreshes every 5 seconds!**

---

## 🎉 Success!

Once set up, you have:
- ✅ 24/7 autonomous trading on Mac Mini
- ✅ Mobile dashboard on phone (anywhere in world)
- ✅ Push notifications via Telegram
- ✅ Remote control from phone
- ✅ Auto-restart on crash
- ✅ Auto-start on boot
- ✅ Complete data on Lexar (portable)
- ✅ Backups on 2TB Dock

**Your trading system is now running autonomously!** 🚀

---

## 📞 Support

**Logs:** `/Volumes/Lexar/RRRVentures/RRRalgorithms/logs/`  
**Config:** `/Volumes/Lexar/RRRVentures/RRRalgorithms/config/`  
**Scripts:** `/Volumes/Lexar/RRRVentures/RRRalgorithms/scripts/`  

**Common Commands:**
```bash
# Check if running
launchctl list | grep rrrventures

# View logs
tail -f /Volumes/Lexar/RRRVentures/RRRalgorithms/logs/system/launchd.log

# Restart
launchctl stop com.rrrventures.trading
launchctl start com.rrrventures.trading

# Manual launch
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
./scripts/launch.sh
```

**Need Help?** Check `TROUBLESHOOTING.md`

---

**Happy Autonomous Trading! 📈🤖📱**


