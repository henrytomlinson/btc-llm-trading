#!/bin/bash
"""
Deploy No-Fee Mode features to VPS
This script updates the trading system with dynamic delta floor and no-fee mode settings
"""

echo "🚀 Deploying No-Fee Mode features to VPS..."

# Copy the updated files to the server
echo "📁 Copying updated files to server..."
scp db.py root@172.237.119.193:/opt/btc-trading/
scp main_btc.py root@172.237.119.193:/opt/btc-trading/
scp kraken_trading_btc.py root@172.237.119.193:/opt/btc-trading/
scp auto_trade_scheduler.py root@172.237.119.193:/opt/btc-trading/

# Restart the trading application to apply the new features
echo "🔄 Restarting trading application..."
ssh root@172.237.119.193 "cd /opt/btc-trading && docker compose restart"

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 15

# Test the new settings endpoint
echo "🧪 Testing new settings endpoint..."
ssh root@172.237.119.193 "cd /opt/btc-trading && curl -k https://henryt-btc.live/settings"

# Test the auto-trade scheduler with new settings
echo "🧪 Testing auto-trade scheduler with new settings..."
ssh root@172.237.119.193 "cd /opt/btc-trading && python3 auto_trade_scheduler.py"

echo ""
echo "✅ No-Fee Mode features deployed successfully!"
echo ""
echo "📋 New Features:"
echo "1. ✅ No-Fee Mode: Enabled by default (min_trade_delta_usd = $10.0)"
echo "2. ✅ Dynamic Delta Floor: min_trade_delta_pct = 0.00% (allows micro-deltas)"
echo "3. ✅ Smart Delta Calculation: Uses exchange minimum notional when no fees"
echo "4. ✅ Settings API: Updated to include new parameters"
echo ""
echo "🎯 The trading system will now:"
echo "- Allow smaller trades when no fees are involved"
echo "- Use $10 minimum for no-fee mode (Kraken exchange minimum)"
echo "- Use $30 minimum for fee mode (to account for trading costs)"
echo "- Support micro-delta adjustments (0% minimum percentage)"
echo ""
echo "📊 Check the dashboard at https://henryt-btc.live to see the new settings!"
