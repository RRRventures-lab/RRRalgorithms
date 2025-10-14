# 🛠️ Troubleshooting Guide - Mac Mini Deployment

Common issues and solutions for 24/7 Mac Mini trading deployment.

---

## 🔍 Quick Diagnostics

### Is Everything Running?

```bash
# SSH to Mac Mini (or use Terminal on Mac Mini)
ssh your-user@mac-mini

# Check trading system
launchctl list | grep rrrventures
# Should show: PID	Status	Label

# Check processes
ps aux | grep "python\|streamlit"
# Should show: python (trading), streamlit (dashboard)

# Check Lexar drive
ls -la /Volumes/Lexar/RRRVentures/RRRalgorithms
# Should list project files
```

**All good?** ✅ System is running!  
**Missing processes?** ⚠️ See troubleshooting below

---

## 🚨 Common Issues

### 1. Dashboard Not Accessible from Phone

**Symptoms:**
- Can't access `http://mac-mini:8501`
- Connection timeout
- Page not loading

**Solution A: Check Tailscale**
```bash
# On Mac Mini
tailscale status
# Should show: "Connected"

# On iPhone
# Open Tailscale app
# Should show Mac Mini as online
```

**Solution B: Check Streamlit Running**
```bash
# On Mac Mini
ps aux | grep streamlit

# If not running
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
source venv/bin/activate
streamlit run src/dashboards/mobile_dashboard.py --server.address 0.0.0.0
```

**Solution C: Check Firewall**
```
Mac Mini → System Settings → Network → Firewall
- If ON: Click "Options" → Allow Python
- Or: Turn OFF (Tailscale already provides security)
```

**Solution D: Find Correct Hostname**
```bash
# On Mac Mini
hostname
# Use this name: http://that-hostname:8501

# Or use Tailscale IP
tailscale ip
# Use: http://100.x.x.x:8501
```

---

### 2. Trading System Not Starting

**Symptoms:**
- No processes running
- LaunchAgent shows "0" PID
- No log files created

**Check Logs:**
```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms

# LaunchAgent logs
tail -100 logs/system/launchd.log
tail -100 logs/system/launchd.err

# System logs
tail -100 logs/system/system.log
```

**Common Causes:**

**Cause A: Lexar Drive Not Mounted**
```bash
# Check if mounted
ls /Volumes/Lexar

# If not mounted:
# - Check USB cable connection
# - Try different USB port
# - Check Disk Utility for drive health
```

**Cause B: Virtual Environment Issue**
```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms

# Recreate venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-local.txt
```

**Cause C: Permission Issues**
```bash
# Fix permissions
chmod +x scripts/*.sh
chmod -R u+w data/ logs/
```

**Cause D: Python Import Errors**
```bash
# Test imports
source venv/bin/activate
python -c "from src.core.database.local_db import get_db; print('✅ Imports OK')"

# If psycopg2 error (expected for local-only mode):
# System will use local_db.py directly, not database.py
```

---

### 3. Telegram Bot Not Responding

**Symptoms:**
- No messages received
- Commands don't work
- Bot shows offline

**Check Configuration:**
```bash
# Verify token and chat ID set
cat /Volumes/Lexar/RRRVentures/RRRalgorithms/config/.env.lexar | grep TELEGRAM

# Should show:
# TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
# TELEGRAM_CHAT_ID=123456789
```

**Test Bot:**
```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
source venv/bin/activate

# Install if needed
pip install python-telegram-bot

# Run bot manually
python -m src.monitoring.telegram_alerts

# Should send "System Started" message
```

**Common Issues:**

**Issue A: Wrong Token**
- Go back to @BotFather
- Send `/mybots` → Select your bot → API Token
- Copy exact token (include everything)

**Issue B: Wrong Chat ID**
- Talk to @userinfobot on Telegram
- Send `/start`
- Copy the ID number (no quotes)

**Issue C: Bot Not Installed**
```bash
source venv/bin/activate
pip install python-telegram-bot
```

---

### 4. High Memory Usage

**Symptoms:**
- System slow
- Memory warning alerts
- Processes being killed

**Check Memory:**
```bash
# Current usage
vm_stat | head -5

# Or use Activity Monitor
open "/System/Applications/Utilities/Activity Monitor.app"
```

**Solutions:**

**Solution A: Reduce Symbol Count**
```yaml
# config/lexar.yml
data_pipeline:
  mock:
    symbols:
      - BTC-USD
      - ETH-USD
      # Remove others temporarily
```

**Solution B: Use Mock Models (Not Full ML)**
```yaml
neural_network:
  mode: mock  # Not "full"
```

**Solution C: Reduce Cache Size**
```yaml
data_pipeline:
  cache_size_mb: 100  # Reduce from default
```

**Solution D: Restart System**
```bash
launchctl stop com.rrrventures.trading
launchctl start com.rrrventures.trading
```

---

### 5. Disk Space Running Out

**Symptoms:**
- Warnings about disk space
- System slow
- Can't save data

**Check Space:**
```bash
# Lexar drive space
df -h /Volumes/Lexar

# If >90% full:
```

**Solutions:**

**Solution A: Clean Old Logs**
```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms

# Compress old logs
find logs/ -name "*.log" -mtime +30 -exec gzip {} \;

# Move to archive
mv logs/trading/*.gz logs/archive/
mv logs/system/*.gz logs/archive/
```

**Solution B: Archive Historical Data**
```bash
# Move old data to 2TB Dock
mv data/historical/2023* /Volumes/2TB-Dock/archive/
```

**Solution C: Clean Cache**
```bash
rm -rf data/cache/*
```

**Solution D: Database Vacuum**
```bash
sqlite3 data/database/local.db "VACUUM;"
# Reclaims deleted space
```

---

### 6. System Keeps Crashing

**Symptoms:**
- Frequent restarts
- Error logs show exceptions
- Unstable operation

**Check Crash Logs:**
```bash
# Recent crashes
tail -100 logs/system/launchd.err

# Look for Python tracebacks
grep "Traceback" logs/system/*.log
```

**Common Causes:**

**Cause A: Import Errors**
```bash
# Missing dependencies
source venv/bin/activate
pip install --upgrade -r requirements-local.txt
```

**Cause B: Database Locked**
```bash
# Stop all processes
launchctl stop com.rrrventures.trading

# Remove database lock
rm data/database/local.db-wal
rm data/database/local.db-shm

# Restart
launchctl start com.rrrventures.trading
```

**Cause C: Lexar Drive Issues**
```bash
# Check drive health
diskutil info /Volumes/Lexar

# Run First Aid
diskutil verifyVolume /Volumes/Lexar
diskutil repairVolume /Volumes/Lexar
```

---

### 7. Can't Access from iPhone

**Symptoms:**
- Dashboard won't load
- Timeout errors
- "Can't connect to server"

**Checklist:**

**1. Same Network?**
- ✅ If using Tailscale: Both on Tailscale
- ✅ If using WiFi: Both on same WiFi

**2. Mac Mini Awake?**
```bash
# On Mac Mini
caffeinate -d
# Keeps display awake
```

**3. Correct URL?**
```bash
# Find Mac Mini address
# On Mac Mini:
ifconfig | grep "inet "
# Or Tailscale:
tailscale ip

# Use: http://that-ip:8501
```

**4. Port 8501 Open?**
```bash
# On Mac Mini
lsof -i :8501
# Should show streamlit process
```

---

### 8. Telegram Alerts Stopped

**Symptoms:**
- No recent alerts
- Commands don't work
- Bot seems dead

**Check Bot Process:**
```bash
ps aux | grep telegram

# If not running:
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
source venv/bin/activate
python -m src.monitoring.telegram_alerts &
```

**Check Logs:**
```bash
tail -50 logs/system/telegram.log
```

**Restart Bot:**
```bash
# Kill old process
pkill -f telegram_alerts

# Start new one
python -m src.monitoring.telegram_alerts &
```

---

### 9. Performance Degradation

**Symptoms:**
- Dashboard slow
- Trading lag
- High CPU usage

**Check System Resources:**
```bash
# CPU usage
top -l 1 | grep "CPU usage"

# Memory
vm_stat | perl -ne '/page size of (\d+)/ and $size=$1; /Pages\s+([^:]+)[^\d]+(\d+)/ and printf("%-16s % 16.2f MB\n", "$1:", $2 * $size / 1048576);'

# Disk I/O
iostat 1 5
```

**Solutions:**

**Restart System:**
```bash
launchctl stop com.rrrventures.trading
launchctl start com.rrrventures.trading
```

**Reduce Load:**
- Decrease symbol count
- Increase update interval
- Use mock instead of full ML

**Check Database Size:**
```bash
ls -lh data/database/local.db
# If >1GB, consider archiving old data
```

---

## 🔧 Advanced Troubleshooting

### View All Logs

```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms

# System logs
tail -f logs/system/*.log

# Trading logs
tail -f logs/trading/*.log

# LaunchAgent logs
tail -f logs/system/launchd.log

# Dashboard logs
tail -f logs/system/dashboard.log
```

### Manual System Control

```bash
# Stop everything
launchctl stop com.rrrventures.trading
pkill -f "python.*src.main"
pkill -f streamlit

# Start manually (for debugging)
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
./scripts/launch.sh

# Or start components separately
./scripts/launch.sh --dashboard  # Dashboard only
./scripts/launch.sh --trading    # Trading only
```

### Reset Everything

```bash
# Nuclear option - complete reset
launchctl unload ~/Library/LaunchAgents/com.rrrventures.trading.plist
rm -rf data/database/local.db
rm -rf data/cache/*
rm logs/system/*

# Reinitialize
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
source venv/bin/activate
python -c "from src.core.database.local_db import get_db; get_db()"

# Restart
launchctl load ~/Library/LaunchAgents/com.rrrventures.trading.plist
```

---

## 📞 Getting Help

### Diagnostic Information to Collect

When asking for help, provide:

```bash
# System info
uname -a
sw_vers

# Python version
python3 --version

# Disk space
df -h /Volumes/Lexar

# Memory
vm_stat

# Processes
ps aux | grep python

# Recent logs
tail -100 logs/system/launchd.err
```

### Log Files Locations

- **System logs:** `logs/system/`
- **Trading logs:** `logs/trading/`
- **Dashboard logs:** `logs/system/dashboard.log`
- **Telegram logs:** `logs/system/telegram.log`
- **LaunchAgent logs:** `logs/system/launchd.log`

---

## ✅ Prevention Tips

### Regular Maintenance

**Daily:**
- Check dashboard once
- Review any error alerts
- Verify system running

**Weekly:**
- Review logs for errors
- Check disk space
- Verify backups working
- Test Telegram alerts

**Monthly:**
- Update Python packages
- Clean old logs
- Archive old data
- Review system performance

### Monitoring Best Practices

- ✅ Enable Telegram alerts
- ✅ Check dashboard daily
- ✅ Review logs weekly
- ✅ Test emergency stop occasionally
- ✅ Keep Mac Mini updated (security only)
- ✅ Monitor disk space
- ✅ Verify backups

---

**Most issues can be fixed with a restart!** 🔄

**If stuck, check logs first.** 📋

**For serious issues, stop trading immediately.** 🚨


