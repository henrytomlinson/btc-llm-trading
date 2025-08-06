#!/usr/bin/env python3
"""
Bitcoin Alpaca Trading Bot
Specialized for Bitcoin trading with compact LLM integration.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import requests

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlpacaTradingBot:
    """Bitcoin-focused Alpaca trading bot"""
    
    def __init__(self):
        """Initialize the Bitcoin trading bot"""
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not found")
        
        # Initialize Alpaca clients
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        # Trading configuration for Bitcoin
        self.position_size = float(os.getenv('POSITION_SIZE', '1000'))  # Default $1000 per trade
        self.max_positions = int(os.getenv('MAX_POSITIONS', '3'))  # Maximum concurrent positions
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '0.05'))  # 5% stop loss
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', '0.10'))  # 10% take profit
        
        # Bitcoin-related symbols for paper trading
        # Using Bitcoin-related stocks/ETFs that are available on Alpaca
        self.btc_symbols = {
            "GBTC": "Grayscale Bitcoin Trust",  # Bitcoin ETF
            "BITO": "ProShares Bitcoin Strategy ETF",  # Bitcoin futures ETF
            "ARKB": "ARK 21Shares Bitcoin ETF",  # Bitcoin ETF
            "IBIT": "iShares Bitcoin Trust"  # Bitcoin ETF
        }
        self.primary_btc_symbol = "GBTC"  # Default to GBTC
        self.min_trade_amount = 10.0  # Minimum trade amount in USD
        
        logger.info("Bitcoin Alpaca trading bot initialized successfully")
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            account = self.trading_client.get_account()
            return {
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "daytrade_count": account.daytrade_count
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        try:
            positions = self.trading_client.get_all_positions()
            position_dict = {}
            
            for position in positions:
                if position.symbol == "BTC/USD":
                    position_dict["BTC"] = {
                        "quantity": float(position.qty),
                        "market_value": float(position.market_value),
                        "unrealized_pl": float(position.unrealized_pl),
                        "side": position.side
                    }
            
            return position_dict
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_market_data(self, symbol: str = "GBTC", bars: int = 1) -> Optional[Dict]:
        """Get Bitcoin-related stock market data"""
        try:
            # Get recent bars for Bitcoin-related stock
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=datetime.now() - timedelta(days=1),
                end=datetime.now()
            )
            
            bars_data = self.data_client.get_stock_bars(request_params)
            
            if bars_data and len(bars_data) > 0:
                latest_bar = bars_data[-1]
                return {
                    'open': float(latest_bar.o),
                    'high': float(latest_bar.h),
                    'low': float(latest_bar.l),
                    'close': float(latest_bar.c),
                    'volume': float(latest_bar.v),
                    'timestamp': latest_bar.t
                }
            else:
                # Fallback to estimated Bitcoin data
                return {
                    'open': 45000.0,
                    'high': 46000.0,
                    'low': 44000.0,
                    'close': 45000.0,
                    'volume': 1000000.0,
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            # Return estimated Bitcoin data as fallback
            return {
                'open': 45000.0,
                'high': 46000.0,
                'low': 44000.0,
                'close': 45000.0,
                'volume': 1000000.0,
                'timestamp': datetime.now()
            }
    
    def calculate_position_size(self, dollar_amount: float, signal_strength: float = 1.0) -> int:
        """Calculate Bitcoin position size based on dollar amount"""
        try:
            # Get current Bitcoin price
            market_data = self.get_market_data()
            if market_data:
                current_price = market_data['close']
            else:
                # Fallback Bitcoin price
                current_price = 45000.0
            
            # Calculate Bitcoin amount (in BTC, not shares)
            btc_amount = dollar_amount / current_price
            
            # Ensure minimum trade size
            if btc_amount * current_price < self.min_trade_amount:
                btc_amount = self.min_trade_amount / current_price
            
            return btc_amount
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def place_buy_order(self, symbol: str, dollar_amount: float, signal_strength: float = 1.0) -> Dict:
        """Place a Bitcoin-related stock buy order"""
        try:
            # Use GBTC as the default Bitcoin proxy
            if symbol == "BTC/USD":
                symbol = self.primary_btc_symbol
            
            # Check if we already have a position
            positions = self.get_positions()
            if symbol in positions:
                return {'status': 'skipped', 'reason': 'existing_position'}
            
            # Check position limits
            if len(positions) >= self.max_positions:
                return {'status': 'skipped', 'reason': 'max_positions'}
            
            # Get current market price
            market_data = self.get_market_data(symbol)
            if not market_data:
                return {'status': 'error', 'reason': 'unable_to_get_market_data'}
            
            current_price = market_data['close']
            
            # Calculate shares to buy
            shares_to_buy = dollar_amount / current_price
            
            # Ensure minimum order value (Alpaca requires at least $1)
            # Also ensure we have at least 1 share
            if shares_to_buy < 1.0:
                shares_to_buy = 1.0
                dollar_amount = shares_to_buy * current_price
            
            # Round to 2 decimal places for shares
            shares_to_buy = round(shares_to_buy, 2)
            
            # Get account info to check available cash
            account_info = self.get_account_info()
            available_cash = account_info.get('cash', 0)
            
            if dollar_amount > available_cash:
                return {'status': 'error', 'reason': 'insufficient_cash'}
            
            # Create market order for Bitcoin-related stock
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=shares_to_buy,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_data)
            
            logger.info(f"Bitcoin proxy buy order placed for ${dollar_amount:.2f} ({shares_to_buy:.2f} shares of {symbol})")
            
            return {
                'status': 'success',
                'order_id': order.id,
                'symbol': symbol,
                'quantity': shares_to_buy,
                'dollar_amount': dollar_amount,
                'actual_amount': shares_to_buy * current_price,
                'side': 'buy',
                'timestamp': datetime.now().isoformat(),
                'price_per_share': current_price
            }
            
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def place_sell_order(self, symbol: str, dollar_amount: float) -> Dict:
        """Place a Bitcoin-related stock sell order"""
        try:
            # Use GBTC as the default Bitcoin proxy
            if symbol == "BTC/USD":
                symbol = self.primary_btc_symbol
            
            # Check if we have a position to sell
            positions = self.get_positions()
            if symbol not in positions:
                return {'status': 'error', 'reason': 'no_position'}
            
            position = positions[symbol]
            available_shares = position['quantity']
            
            # Calculate shares to sell
            market_data = self.get_market_data(symbol)
            current_price = market_data['close'] if market_data else 50.0
            shares_to_sell = dollar_amount / current_price
            
            # Ensure we don't sell more than we have
            if shares_to_sell > available_shares:
                shares_to_sell = available_shares
            
            if shares_to_sell <= 0:
                return {'status': 'error', 'reason': 'insufficient_shares'}
            
            # Create market order for Bitcoin-related stock
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=shares_to_sell,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_data)
            
            logger.info(f"Bitcoin proxy sell order placed for ${dollar_amount:.2f} ({shares_to_sell:.2f} shares of {symbol})")
            
            return {
                'status': 'success',
                'order_id': order.id,
                'symbol': symbol,
                'quantity': shares_to_sell,
                'dollar_amount': dollar_amount,
                'side': 'sell',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error placing sell order: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def close_all_positions(self) -> Dict:
        """Close all Bitcoin positions"""
        try:
            positions = self.get_positions()
            closed_positions = []
            
            for symbol, position in positions.items():
                if symbol == "BTC":
                    # Create sell order for all Bitcoin
                    order_data = MarketOrderRequest(
                        symbol="BTC/USD",
                        qty=position['quantity'],
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    order = self.trading_client.submit_order(order_data)
                    closed_positions.append({
                        'symbol': symbol,
                        'quantity': position['quantity'],
                        'order_id': order.id
                    })
            
            return {
                'status': 'success',
                'closed_positions': closed_positions,
                'message': f"Closed {len(closed_positions)} positions"
            }
            
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def get_trading_summary(self) -> Dict:
        """Get comprehensive trading summary"""
        try:
            account_info = self.get_account_info()
            positions = self.get_positions()
            
            total_positions = len(positions)
            total_value = sum(pos['market_value'] for pos in positions.values())
            total_unrealized_pl = sum(pos['unrealized_pl'] for pos in positions.values())
            
            return {
                "account": account_info,
                "positions": positions,
                "summary": {
                    "total_positions": total_positions,
                    "total_value": total_value,
                    "total_unrealized_pl": total_unrealized_pl,
                    "max_positions": self.max_positions,
                    "position_size": self.position_size
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting trading summary: {e}")
            return {}

if __name__ == "__main__":
    # Test the Bitcoin trading bot
    bot = AlpacaTradingBot()
    print("Bitcoin Trading Bot Test:")
    print(f"Account Info: {bot.get_account_info()}")
    print(f"Positions: {bot.get_positions()}")
    print(f"Market Data: {bot.get_market_data()}") 