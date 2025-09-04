#!/usr/bin/env python3
"""Simple script to sell $20 worth of BTC to free up cash for the bot"""

from kraken_trading_btc import KrakenTradingBot, best_quotes, get_pair_meta, round_down

def main():
    try:
        print("🔄 Initializing Kraken trading bot...")
        bot = KrakenTradingBot()
        
        print("📊 Getting current market quotes...")
        qts = best_quotes(bot)
        meta = get_pair_meta(bot)
        
        usd_amount = 20.0
        print(f"💰 Target USD amount: ${usd_amount}")
        
        # Calculate quantity to sell
        qty = round_down(usd_amount / qts["mid"], meta["lot_step"])
        actual_usd = qty * qts["mid"]
        
        print(f"📈 Current BTC price: ${qts['mid']:.2f}")
        print(f"🔢 BTC quantity to sell: {qty:.8f}")
        print(f"💵 Actual USD value: ${actual_usd:.2f}")
        
        # Place the sell order
        print("\n🚀 Placing sell order...")
        ok, res = bot.place_limit_post_only("sell", qty, qts["mid"])
        
        if ok:
            print("✅ Sell order placed successfully!")
            print(f"📋 Order details: {res}")
        else:
            print("❌ Failed to place sell order")
            print(f"📋 Error details: {res}")
            
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
