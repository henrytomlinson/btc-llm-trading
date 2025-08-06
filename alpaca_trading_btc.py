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
        
        # Bitcoin-specific settings
        self.btc_symbol = "BTC/USD"  # Bitcoin trading pair
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
    
    def get_market_data(self, symbol: str = "BTC/USD", bars: int = 1) -> Optional[Dict]:
        """Get Bitcoin market data"""
        try:
            # Get recent bars for Bitcoin
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
        """Place a Bitcoin buy order"""
        try:
            # Check if we already have a position
            positions = self.get_positions()
            if "BTC" in positions:
                return {'status': 'skipped', 'reason': 'existing_position'}
            
            # Check position limits
            if len(positions) >= self.max_positions:
                return {'status': 'skipped', 'reason': 'max_positions'}
            
            # Calculate Bitcoin amount
            btc_amount = self.calculate_position_size(dollar_amount, signal_strength)
            if btc_amount == 0:
                return {'status': 'error', 'reason': 'insufficient_funds'}
            
            # Get account info to check available cash
            account_info = self.get_account_info()
            available_cash = account_info.get('cash', 0)
            
            if dollar_amount > available_cash:
                return {'status': 'error', 'reason': 'insufficient_cash'}
            
            # Create market order for Bitcoin
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=btc_amount,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_data)
            
            # Calculate actual dollar amount spent
            actual_amount = btc_amount * self.get_market_data(symbol)['close'] if self.get_market_data(symbol) else dollar_amount
            
            logger.info(f"Bitcoin buy order placed for ${dollar_amount:.2f} ({btc_amount:.6f} BTC)")
            
            return {
                'status': 'success',
                'order_id': order.id,
                'symbol': symbol,
                'quantity': btc_amount,
                'dollar_amount': dollar_amount,
                'actual_amount': actual_amount,
                'side': 'buy',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def place_sell_order(self, symbol: str, dollar_amount: float) -> Dict:
        """Place a Bitcoin sell order"""
        try:
            # Check if we have a position to sell
            positions = self.get_positions()
            if "BTC" not in positions:
                return {'status': 'error', 'reason': 'no_position'}
            
            btc_position = positions["BTC"]
            available_btc = btc_position['quantity']
            
            # Calculate Bitcoin amount to sell
            market_data = self.get_market_data(symbol)
            current_price = market_data['close'] if market_data else 45000.0
            btc_to_sell = dollar_amount / current_price
            
            # Ensure we don't sell more than we have
            if btc_to_sell > available_btc:
                btc_to_sell = available_btc
            
            if btc_to_sell <= 0:
                return {'status': 'error', 'reason': 'insufficient_btc'}
            
            # Create market order for Bitcoin
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=btc_to_sell,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_data)
            
            logger.info(f"Bitcoin sell order placed for ${dollar_amount:.2f} ({btc_to_sell:.6f} BTC)")
            
            return {
                'status': 'success',
                'order_id': order.id,
                'symbol': symbol,
                'quantity': btc_to_sell,
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