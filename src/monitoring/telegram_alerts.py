from datetime import datetime
from functools import lru_cache
from pathlib import Path
from src.core.database.local_db import get_db
from typing import Optional, Dict, Any
import asyncio
import logging
import os
import time


"""
Telegram Alerts System
======================

Send trading alerts and receive commands via Telegram bot.
Enables mobile control and monitoring from anywhere.

Setup:
1. Talk to @BotFather on Telegram
2. Create new bot, get token
3. Add to config/.env.lexar:
   TELEGRAM_BOT_TOKEN=your_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
4. Run: python -m src.monitoring.telegram_alerts

Author: RRR Ventures
Date: 2025-10-12
"""


try:
    from telegram import Update, Bot
    from telegram.ext import Application, CommandHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    # Define placeholder types when telegram is not available
    Update = Any
    ContextTypes = Any
    Bot = Any
    Application = Any
    CommandHandler = Any
    print("⚠️  python-telegram-bot not installed. Run: pip install python-telegram-bot")


logger = logging.getLogger(__name__)


class TelegramAlerts:
    """
    Telegram bot for alerts and remote control.
    
    Sends push notifications to your phone for:
    - Trade executions
    - System status
    - Errors and warnings
    - Daily summaries
    
    Receives commands:
    - /status - Portfolio status
    - /trades - Recent trades
    - /positions - Open positions
    - /pause - Pause trading
    - /resume - Resume trading
    - /emergency_stop - Stop everything
    """
    
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize Telegram bot.
        
        Args:
            token: Bot token from @BotFather
            chat_id: Your Telegram chat ID
        """
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot not installed")
        
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set")
        
        self.bot = Bot(token=self.token)
        self.app = None
        self.trading_paused = False
    
    async def send_message(self, message: str, parse_mode: str = 'Markdown'):
        """
        Send message to user.
        
        Args:
            message: Message text
            parse_mode: Telegram parse mode (Markdown or HTML)
        """
        try:
            if self.chat_id:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=parse_mode
                )
            else:
                logger.warning("TELEGRAM_CHAT_ID not set, message not sent")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    async def send_trade_alert(self, trade: Dict[str, Any]):
        """Send trade execution alert"""
        pnl = trade.get('pnl', 0)
        pnl_emoji = "💚" if pnl > 0 else "❤️" if pnl < 0 else "💛"
        
        message = f"""
🤖 *Trade Executed*

Symbol: `{trade['symbol']}`
Side: *{trade['side'].upper()}*
Quantity: {trade['quantity']}
Price: ${trade['price']:,.2f}
{pnl_emoji} P&L: ${pnl:+,.2f}

Time: {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    async def send_daily_summary(self):
        """Send daily performance summary"""
        try:
            db = get_db()
            metrics = db.get_latest_portfolio_metrics()
            trades = db.get_trades(limit=100)
            
            trades_today = len([t for t in trades if t['timestamp'] > time.time() - 86400])
            
            message = f"""
📊 *Daily Summary* - {datetime.now().strftime('%Y-%m-%d')}

💰 Portfolio: ${metrics['total_value']:,.2f}
📈 Total P&L: ${metrics['total_pnl']:+,.2f}
📅 Today P&L: ${metrics.get('daily_pnl', 0):+,.2f}
🎯 Win Rate: {metrics.get('win_rate', 0):.1%}

🔄 Trades Today: {trades_today}
📊 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}

Status: 🟢 RUNNING
"""
            await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
    
    async def send_risk_warning(self, warning: str):
        """Send risk limit warning"""
        message = f"""
⚠️ *Risk Warning*

{warning}

Time: {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    async def send_system_alert(self, alert: str, severity: str = "INFO"):
        """Send system alert"""
        emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "🚨",
            "CRITICAL": "🔴"
        }.get(severity, "ℹ️")
        
        message = f"""
{emoji} *System Alert*

{alert}

Time: {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    async def cmd_status(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"):
        """Handle /status command"""
        try:
            db = get_db()
            metrics = db.get_latest_portfolio_metrics()
            
            if metrics:
                message = f"""
📊 *System Status*

💰 Portfolio: ${metrics['total_value']:,.2f}
📈 P&L: ${metrics['total_pnl']:+,.2f}
🎯 Win Rate: {metrics.get('win_rate', 0):.1%}

Status: {'⏸️ PAUSED' if self.trading_paused else '🟢 RUNNING'}
Time: {datetime.now().strftime('%H:%M:%S')}
"""
            else:
                message = "❌ No metrics available"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def cmd_trades(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"):
        """Handle /trades command"""
        try:
            db = get_db()
            trades = db.get_trades(limit=5)
            
            if trades:
                message = "📜 *Recent Trades*\n\n"
                for trade in trades[:5]:
                    pnl = trade.get('pnl', 0)
                    pnl_emoji = "💚" if pnl > 0 else "❤️" if pnl < 0 else "💛"
                    message += f"{pnl_emoji} {trade['symbol']} {trade['side'].upper()} "
                    message += f"{trade['quantity']} @ ${trade['price']:,.2f}\n"
            else:
                message = "No recent trades"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def cmd_positions(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"):
        """Handle /positions command"""
        try:
            db = get_db()
            positions = db.get_positions()
            
            if positions:
                message = "💼 *Open Positions*\n\n"
                for pos in positions:
                    pnl = pos.get('unrealized_pnl', 0)
                    pnl_emoji = "💚" if pnl > 0 else "❤️" if pnl < 0 else "💛"
                    message += f"{pnl_emoji} {pos['symbol']}: {pos['quantity']} units\n"
                    message += f"   P&L: ${pnl:+,.2f}\n\n"
            else:
                message = "No open positions"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def cmd_pause(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"):
        """Handle /pause command"""
        self.trading_paused = True
        await update.message.reply_text("⏸️ *Trading PAUSED*\n\nUse /resume to continue", parse_mode='Markdown')
        logger.info("Trading paused via Telegram")
    
    async def cmd_resume(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"):
        """Handle /resume command"""
        self.trading_paused = False
        await update.message.reply_text("▶️ *Trading RESUMED*", parse_mode='Markdown')
        logger.info("Trading resumed via Telegram")
    
    async def cmd_stop(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"):
        """Handle /stop emergency stop command"""
        await update.message.reply_text(
            "🚨 *EMERGENCY STOP*\n\nAll trading halted!\n\nUse /resume to restart",
            parse_mode='Markdown'
        )
        self.trading_paused = True
        logger.critical("Emergency stop triggered via Telegram")

        # Implement actual trading halt
        try:
            # Write emergency stop flag to database
            db = get_db()
            db.execute(
                "INSERT OR REPLACE INTO system_flags (key, value, timestamp) VALUES (?, ?, ?)",
                ('emergency_stop', 'true', time.time())
            )

            # Kill any running trading processes (gracefully)
            import os
            import signal
            # Find and stop trading processes
            for line in os.popen("ps aux | grep 'src/main.py' | grep -v grep"):
                fields = line.split()
                pid = int(fields[1])
                try:
                    os.kill(pid, signal.SIGTERM)
                    logger.info(f"Sent SIGTERM to trading process {pid}")
                except ProcessLookupError:
                    pass

            await self.send_message("⛔ Trading processes terminated successfully")
        except Exception as e:
            logger.error(f"Failed to halt trading: {e}")
            await self.send_message(f"⚠️ Error halting trading: {e}")
    
    async def cmd_help(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE"):
        """Handle /help command"""
        message = """
📱 *RRRalgorithms Bot Commands*

*Status & Info:*
/status - Portfolio and system status
/trades - Recent trades
/positions - Open positions

*Controls:*
/pause - Pause trading
/resume - Resume trading
/stop - Emergency stop (halts all trading)

*Help:*
/help - This message

Dashboard: Access via Tailscale
"""
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def start_bot(self):
        """Start the Telegram bot"""
        self.app = Application.builder().token(self.token).build()
        
        # Add command handlers
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("trades", self.cmd_trades))
        self.app.add_handler(CommandHandler("positions", self.cmd_positions))
        self.app.add_handler(CommandHandler("pause", self.cmd_pause))
        self.app.add_handler(CommandHandler("resume", self.cmd_resume))
        self.app.add_handler(CommandHandler("stop", self.cmd_stop))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("start", self.cmd_help))
        
        # Start polling
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        
        logger.info("Telegram bot started")
        print("✅ Telegram bot is running...")
        print("   Send /help to your bot for commands")
        
        # Send startup message
        await self.send_message("🚀 *Trading System Started*\n\nSystem is online and ready!")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
    
    async def stop_bot(self):
        """Stop the Telegram bot"""
        if self.app:
            await self.send_message("⏹️ *Trading System Stopped*")
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram bot stopped")


# Convenience functions for quick integration
_global_bot: Optional[TelegramAlerts] = None


@lru_cache(maxsize=128)


def get_telegram_bot() -> Optional[TelegramAlerts]:
    """Get global Telegram bot instance"""
    global _global_bot
    
    if _global_bot is None:
        try:
            _global_bot = TelegramAlerts()
        except Exception as e:
            logger.warning(f"Telegram bot not available: {e}")
            return None
    
    return _global_bot


async def send_trade_alert_async(trade: Dict[str, Any]):
    """Send trade alert (async)"""
    bot = get_telegram_bot()
    if bot:
        await bot.send_trade_alert(trade)


def send_trade_alert(trade: Dict[str, Any]):
    """Send trade alert (sync wrapper)"""
    bot = get_telegram_bot()
    if bot:
        asyncio.run(bot.send_trade_alert(trade))


async def send_daily_summary_async():
    """Send daily summary (async)"""
    bot = get_telegram_bot()
    if bot:
        await bot.send_daily_summary()


def send_daily_summary():
    """Send daily summary (sync wrapper)"""
    bot = get_telegram_bot()
    if bot:
        asyncio.run(bot.send_daily_summary())


if __name__ == "__main__":
    """Run Telegram bot standalone"""
    print("="*70)
    print("RRRalgorithms - Telegram Bot")
    print("="*70)
    print()
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / 'config' / '.env.lexar')
    
    # Check configuration
    if not os.getenv('TELEGRAM_BOT_TOKEN'):
        print("❌ TELEGRAM_BOT_TOKEN not set in config/.env.lexar")
        print()
        print("Setup instructions:")
        print("1. Talk to @BotFather on Telegram")
        print("2. Create new bot with /newbot")
        print("3. Copy the token")
        print("4. Add to config/.env.lexar:")
        print("   TELEGRAM_BOT_TOKEN=your_token_here")
        print("   TELEGRAM_CHAT_ID=your_chat_id_here")
        print()
        sys.exit(1)
    
    # Create and start bot
    try:
        bot = TelegramAlerts()
        print(f"✅ Bot initialized")
        print(f"   Chat ID: {bot.chat_id or 'Not set'}")
        print()
        print("Starting bot...")
        print("Send /help to your bot to see available commands")
        print()
        
        asyncio.run(bot.start_bot())
        
    except KeyboardInterrupt:
        print("\n⏹️  Bot stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

