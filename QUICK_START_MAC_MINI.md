# 🚀 Quick Start - Mac Mini Deployment

**Get your trading system running on Mac Mini in 1 hour!**

---

## 📋 What You Need

### Hardware (You Have)
- ✅ Mac Mini M4 256GB
- ✅ Lexar 2TB drive (with RRRalgorithms)
- ✅ 2TB Dock (for backups)
- ✅ Ethernet cable (recommended)
- ✅ iPhone/iPad

### Before You Start
- ✅ System is on Lexar at: `/Volumes/Lexar/RRRVentures/RRRalgorithms`
- ✅ All deployment scripts created
- ✅ Configuration files ready
- ✅ Documentation complete

---

## ⚡ Ultra-Quick Start (30 minutes)

### On MacBook (Right Now)

**1. Prepare Lexar (5 min):**
```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
./scripts/prepare_for_mac_mini.sh
```

**2. Eject Lexar**

### On Mac Mini (When It Arrives)

**1. Basic macOS Setup (10 min):**
- Power on Mac Mini
- Follow macOS setup wizard
- Create user account
- Connect to internet

**2. Connect Drives:**
- Plug Lexar 2TB into Mac Mini
- Plug 2TB Dock into Mac Mini (for backups)

**3. Run Setup Script (15 min):**
```bash
# Open Terminal
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
./scripts/mac_mini_first_boot.sh
```

This installs everything automatically!

**Done! System is running!** ✅

### On iPhone (5 min)

**1. Install Tailscale:**
- Download from App Store
- Login (same account as Mac Mini)

**2. Access Dashboard:**
```
Safari: http://mac-mini:8501
Add to Home Screen: "RRR Trading"
```

**Done! You can control from phone!** 📱✅

---

## 🎯 Detailed Steps (If You Want More Control)

### Step 1: Prepare on MacBook

```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms

# Run preparation
./scripts/prepare_for_mac_mini.sh

# Output will show:
# ✅ Virtual environment exists
# ✅ All directories created
# ✅ Scripts executable
# ✅ Dependencies installed
# ✅ Ready to transfer!
```

### Step 2: Transfer to Mac Mini

1. **Eject Lexar** from MacBook safely
2. **Plug into Mac Mini** (any USB-C port)
3. **Wait** for mount (appears in Finder)
4. **Verify:** Open Terminal, run `ls /Volumes/Lexar`

### Step 3: Mac Mini First Boot

```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
./scripts/mac_mini_first_boot.sh
```

**What it does:**
1. ✅ Installs Homebrew (if needed)
2. ✅ Installs Python 3.11+
3. ✅ Installs Tailscale
4. ✅ Creates/verifies virtual environment
5. ✅ Initializes database
6. ✅ Sets up auto-start on boot
7. ✅ Configures 24/7 settings
8. ✅ Tests the system

**Takes:** 15-30 minutes (downloads)

### Step 4: Configure Tailscale

```bash
# On Mac Mini (Terminal)
sudo tailscale up

# Opens browser → Login
# Use Google, GitHub, or email
# Approve Mac Mini device
```

**On iPhone:**
1. App Store → Search "Tailscale"
2. Install Tailscale app
3. Open app → "Log In"
4. Use **same account** as Mac Mini
5. Done! Mac Mini appears in device list

### Step 5: Access Dashboard

**On iPhone:**
1. Open Safari or Chrome
2. Type: `http://mac-mini:8501`
3. See dashboard! 📊
4. Tap Share button (square with arrow)
5. Scroll → "Add to Home Screen"
6. Name it "RRR Trading"
7. Tap Add

**Now you have an app icon!** Tap it to access your trading dashboard anytime! 📱

### Step 6: Optional - Telegram Alerts

**Setup (15 min):**

1. **Open Telegram** on phone
2. **Search:** @BotFather
3. **Send:** `/newbot`
4. **Follow prompts,** choose name
5. **Copy token** (123456:ABC-DEF...)

6. **Search:** @userinfobot  
7. **Send:** `/start`
8. **Copy your ID** (123456789)

9. **On Mac Mini,** edit file:
```bash
nano /Volumes/Lexar/RRRVentures/RRRalgorithms/config/.env.lexar

# Add these lines:
TELEGRAM_BOT_TOKEN=your_token_from_botfather
TELEGRAM_CHAT_ID=your_id_from_userinfobot
ENABLE_TELEGRAM_ALERTS=true
```

10. **Restart system:**
```bash
launchctl stop com.rrrventures.trading
launchctl start com.rrrventures.trading
```

11. **Check phone** - Should receive "🚀 Trading System Started" message!

---

## ✅ Verification

### Check Everything is Running

```bash
# On Mac Mini (Terminal):

# 1. Check LaunchAgent
launchctl list | grep rrrventures
# Should show PID and status

# 2. Check processes
ps aux | grep python
# Should show: python (trading)

ps aux | grep streamlit
# Should show: streamlit (dashboard)

# 3. Check dashboard
curl http://localhost:8501
# Should return HTML

# 4. Check Lexar drive
df -h /Volumes/Lexar
# Should show 2TB drive
```

### Test from iPhone

1. **Dashboard:** Open `http://mac-mini:8501` in Safari
   - Should see portfolio, trades, positions
   - Numbers should auto-update
   
2. **Telegram (if configured):** Send `/status` to your bot
   - Should receive portfolio info
   
3. **Controls:** Tap Pause button
   - Should show "Trading paused" message

**All working?** ✅ You're ready!

---

## 🎯 Daily Usage

### Morning (2 minutes)
1. Check Telegram for daily summary (8 AM automatic)
2. Open dashboard app on phone
3. Quick glance at portfolio

### During Day (Passive)
- System runs autonomously
- Receive trade alerts on phone
- Check dashboard if curious
- System monitors itself

### Evening (5 minutes)
1. Review day's trades
2. Check performance
3. Verify system healthy
4. Plan for tomorrow (if active management)

### Weekly (15 minutes)
1. Review weekly performance
2. Check logs for errors
3. Verify backups working
4. Update if needed

---

## 🚨 Emergency Actions

### From iPhone

**Pause Trading:**
1. Open dashboard app
2. Go to System tab
3. Tap "⏸️ PAUSE TRADING"

**Or via Telegram:**
- Send: `/pause`

**Emergency Stop:**
1. Dashboard → System tab → "🚨 EMERGENCY STOP"

**Or via Telegram:**
- Send: `/stop`

### From Anywhere

**SSH Access (if needed):**
```bash
# Via Tailscale from any device
ssh your-user@mac-mini

# Stop trading
launchctl stop com.rrrventures.trading
```

---

## 💡 Pro Tips

### Dashboard

**Add to Home Screen** for app-like experience:
- Looks like native app
- Tap icon to open instantly
- No browser UI clutter
- Works offline (cached)

**Bookmark frequently used:**
- Overview tab (default)
- System tab (controls)

### Telegram

**Enable Notifications:**
- Settings → Notifications → Telegram
- Allow: Lock screen, Notification Center
- Show previews: When Unlocked (for security)

**Mute during sleep:**
- Telegram → Bot chat → 🔔 icon
- Mute from 11 PM - 7 AM

### System Monitoring

**Check dashboard once daily** - 30 seconds
**Read error alerts immediately** - Could indicate issues
**Weekly log review** - Catch patterns

### Backups

**Automatic** to 2TB Dock every 6 hours:
- Database snapshots
- Configuration backups
- Log archives

**Manual backup (monthly):**
```bash
# On Mac Mini
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
tar -czf ../backup_$(date +%Y%m%d).tar.gz .
mv ../backup_*.tar.gz /Volumes/2TB-Dock/manual_backups/
```

---

## 📊 Expected Performance

### Resource Usage (Mac Mini)
- **CPU:** 5-15% (mostly idle)
- **Memory:** 2-3GB (plenty of headroom with 8GB)
- **Disk:** Grows ~100MB/day (logs + data)
- **Network:** Minimal (API calls only)
- **Power:** 6-10W (~$2-3/month electricity)

### Dashboard Performance
- **Load time:** <1 second
- **Refresh:** Every 5 seconds
- **Data usage:** ~1-2MB/hour
- **Battery:** Minimal impact on iPhone

### System Reliability
- **Uptime target:** 99.9% (8 hours downtime/year)
- **Auto-restart:** Within 60 seconds
- **Health checks:** Every 5 minutes
- **Backups:** Every 6 hours

---

## 🎉 You're All Set!

**What you have:**
- ✅ Trading system (verified 87/100)
- ✅ Mobile dashboard
- ✅ Push notifications
- ✅ 24/7 automation
- ✅ Auto-restart
- ✅ Health monitoring
- ✅ Everything on Lexar (portable)

**Next step:**
```bash
./scripts/prepare_for_mac_mini.sh
```

**Then when Mac Mini arrives:**
```bash
./scripts/mac_mini_first_boot.sh
```

**That's it! 🚀**

---

**Questions?** Read the detailed guides in `docs/deployment/`

**Issues?** Check `docs/deployment/TROUBLESHOOTING.md`

**Ready to trade!** 📈💰🎊


