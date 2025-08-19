#!/bin/bash
"""
Setup script for automated Bitcoin trading
This script configures cron jobs and installs dependencies for automated trading
"""

echo "🚀 Setting up Automated Bitcoin Trading..."

# Copy the automation script to the server
echo "📁 Copying automation script to server..."
scp auto_trade_scheduler.py root@172.237.119.193:/opt/btc-trading/
scp main_btc.py root@172.237.119.193:/opt/btc-trading/

# Make the script executable
echo "🔧 Making script executable..."
ssh root@172.237.119.193 "chmod +x /opt/btc-trading/auto_trade_scheduler.py"

# Install Python requests if not already installed
echo "📦 Installing Python dependencies..."
ssh root@172.237.119.193 "cd /opt/btc-trading && pip install requests"

# Restart the trading application to include the new endpoint
echo "🔄 Restarting trading application..."
ssh root@172.237.119.193 "cd /opt/btc-trading && docker compose restart"

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 10

# Test the new endpoint
echo "🧪 Testing automated trading endpoint..."
ssh root@172.237.119.193 "curl -k -X POST https://henryt-btc.live/auto_trade_scheduled"

echo ""
echo "✅ Automated trading setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Test the automation script:"
echo "   ssh root@172.237.119.193"
echo "   cd /opt/btc-trading"
echo "   python3 auto_trade_scheduler.py"
echo ""
echo "2. Set up cron job for automatic execution:"
echo "   crontab -e"
echo "   # Add one of these lines:"
echo "   # Every 15 minutes: */15 * * * * /usr/bin/python3 /opt/btc-trading/auto_trade_scheduler.py"
echo "   # Every hour: 0 * * * * /usr/bin/python3 /opt/btc-trading/auto_trade_scheduler.py"
echo "   # Every 4 hours: 0 */4 * * * /usr/bin/python3 /opt/btc-trading/auto_trade_scheduler.py"
echo ""
echo "3. Monitor logs:"
echo "   tail -f /opt/btc-trading/auto_trade.log"
echo ""
echo "🎯 Your trading system will now run automatically!"
