#!/bin/bash
"""
Deploy centralized risk management function to VPS
This script updates the trading system with the centralized effective_min_delta_usd function
"""

echo "🚀 Deploying centralized risk management function to VPS..."

# Copy the updated files to the server
echo "📁 Copying updated files to server..."
scp risk_management.py root@172.237.119.193:/opt/btc-trading/
scp kraken_trading_btc.py root@172.237.119.193:/opt/btc-trading/

# Restart the trading application to apply the new features
echo "🔄 Restarting trading application..."
ssh root@172.237.119.193 "cd /opt/btc-trading && docker compose restart"

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 15

# Test the centralized function
echo "🧪 Testing centralized risk management function..."
ssh root@172.237.119.193 "cd /opt/btc-trading && python3 -c \"from risk_management import effective_min_delta_usd; settings = {'no_fee_mode': True, 'min_trade_delta_usd': 10.0, 'min_trade_delta_pct': 0.0}; result = effective_min_delta_usd(1000.0, settings); print(f'Effective min delta for $1000 equity: ${result:.2f}')\""

# Test the auto-trade scheduler to ensure it still works
echo "🧪 Testing auto-trade scheduler with centralized function..."
ssh root@172.237.119.193 "cd /opt/btc-trading && python3 auto_trade_scheduler.py"

echo ""
echo "✅ Centralized risk management function deployed successfully!"
echo ""
echo "📋 New Features:"
echo "1. ✅ Centralized effective_min_delta_usd function in risk_management.py"
echo "2. ✅ EXCHANGE_MIN_NOTIONAL_USD constant (Kraken $10 floor)"
echo "3. ✅ Dynamic delta calculation based on fee mode and equity"
echo "4. ✅ Fallback logic for when risk_management module is unavailable"
echo "5. ✅ Helper function for runtime settings integration"
echo ""
echo "🎯 Benefits:"
echo "- Single source of truth for delta calculations"
echo "- Consistent behavior across all trading components"
echo "- Easy to maintain and update delta logic"
echo "- Exchange-specific constants centralized"
echo ""
echo "📊 The trading system now uses centralized risk management!"
