#!/bin/bash
"""
Monitoring script for automated Bitcoin trading
This script shows the status of automated trading and recent logs
"""

echo "🤖 Bitcoin Auto-Trading Monitor"
echo "================================"
echo ""

# Check if cron job is running
echo "📅 Cron Job Status:"
if crontab -l 2>/dev/null | grep -q "auto_trade_scheduler.py"; then
    echo "✅ Cron job is configured"
    crontab -l | grep "auto_trade_scheduler.py"
else
    echo "❌ Cron job not found"
fi
echo ""

# Check recent logs
echo "📊 Recent Auto-Trade Logs:"
if [ -f "/opt/btc-trading/auto_trade.log" ]; then
    echo "Last 10 entries from auto_trade.log:"
    tail -10 /opt/btc-trading/auto_trade.log
else
    echo "No auto_trade.log file found"
fi
echo ""

# Check cron logs
echo "📋 Recent Cron Logs:"
if [ -f "/opt/btc-trading/cron.log" ]; then
    echo "Last 10 entries from cron.log:"
    tail -10 /opt/btc-trading/cron.log
else
    echo "No cron.log file found"
fi
echo ""

# Check if trading system is running
echo "🔧 Trading System Status:"
if curl -k -s https://henryt-btc.live/health > /dev/null 2>&1; then
    echo "✅ Trading system is running"
else
    echo "❌ Trading system is not responding"
fi
echo ""

# Show next scheduled run
echo "⏰ Next Scheduled Runs:"
echo "The system will run every 15 minutes"
echo "Next runs:"
for i in {1..4}; do
    next_time=$(date -d "+$((i*15)) minutes" "+%H:%M")
    echo "  $next_time"
done
echo ""

echo "📈 To view live logs:"
echo "  tail -f /opt/btc-trading/auto_trade.log"
echo "  tail -f /opt/btc-trading/cron.log"
echo ""
echo "🛑 To stop automation:"
echo "  crontab -e  # Remove the auto_trade line"
echo ""
echo "🎯 Your Bitcoin trading system is now fully automated!"
