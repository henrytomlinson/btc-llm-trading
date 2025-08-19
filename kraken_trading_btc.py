#!/usr/bin/env python3
"""
Kraken Bitcoin Trading Bot
Specialized for Bitcoin trading with Kraken API (futures demo environment).
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import requests
import time
import hmac
import hashlib
import base64
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KrakenTradingBot:
    """Bitcoin-focused Kraken trading bot using real spot trading"""
    
    def __init__(self):
        """Initialize the Kraken trading bot"""
        self.api_key = os.getenv('KRAKEN_API_KEY')
        self.secret_key = os.getenv('KRAKEN_PRIVATE_KEY')  # Fixed to match Kubernetes secret
        
        # Use real Kraken API for spot trading
        self.base_url = "https://api.kraken.com"
        self.api_url = f"{self.base_url}/0"
        
        # Trading configuration for Bitcoin
        self.position_size = float(os.getenv('POSITION_SIZE', '1000'))  # Default $1000 per trade
        self.max_positions = int(os.getenv('MAX_POSITIONS', '3'))  # Maximum concurrent positions
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', '0.05'))  # 5% stop loss
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', '0.10'))  # 10% take profit
        
        # Bitcoin spot trading symbol
        self.btc_symbol = "XXBTZUSD"  # Bitcoin/USD pair on Kraken
        self.btc_gbp_symbol = "XXBTZGBP"  # Bitcoin/GBP pair on Kraken
        self.min_trade_amount = 10.0  # Minimum trade amount in USD
        
        # Check if credentials are available
        if not self.api_key or not self.secret_key:
            logger.warning("Kraken API credentials not found - using demo mode only")
            self.demo_mode = True
        else:
            self.demo_mode = False
        
        logger.info("Kraken Bitcoin trading bot initialized successfully")
    
    def _generate_signature(self, endpoint: str, data: Dict, nonce: str) -> str:
        """Generate Kraken API signature for real API"""
        try:
            # Create the signature string according to Kraken API docs
            post_data = urllib.parse.urlencode(data)
            encoded = (str(nonce) + post_data).encode()
            message = endpoint.encode() + hashlib.sha256(encoded).digest()
            
            # Create the signature
            signature = hmac.new(
                base64.b64decode(self.secret_key),
                message,
                hashlib.sha512
            )
            sig_digest = base64.b64encode(signature.digest())
            
            return sig_digest.decode()
            
        except Exception as e:
            logger.error(f"Error generating signature: {e}")
            return ""
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None, private: bool = False, signature_path: str = None) -> Dict:
        """Make API request to Kraken"""
        try:
            url = f"{self.api_url}{endpoint}"
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent': 'KrakenBot/1.0'
            }
            
            if private and not self.demo_mode:
                # Add authentication for private endpoints
                nonce = str(int(time.time() * 1000))
                data = data or {}
                data['nonce'] = nonce
                
                # Use signature_path if provided, otherwise use endpoint
                sig_path = signature_path if signature_path else endpoint
                signature = self._generate_signature(sig_path, data, nonce)
                headers['API-Key'] = self.api_key
                headers['API-Sign'] = signature
            
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=data)
            else:
                response = requests.post(url, headers=headers, data=data)
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error making request to {endpoint}: {e}")
            return {"error": str(e)}
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            if self.demo_mode:
                # Return demo account info
                return {
                    "cash": 100000.0,  # Demo balance
                    "buying_power": 100000.0,
                    "portfolio_value": 100000.0,
                    "equity": 100000.0,
                    "demo_mode": True
                }
            
            # Get real account info from Kraken
            endpoint = "/private/Balance"
            result = self._make_request('POST', endpoint, {}, private=True, signature_path="/0/private/Balance")
            
            if 'error' in result and result['error']:
                return {"error": result['error']}
            
            # Parse account balances
            balances = result.get('result', {})
            
            # Calculate USD and BTC balances
            usd_balance = float(balances.get('ZUSD', '0'))
            btc_balance = float(balances.get('XXBT', '0'))
            
            # Get current BTC price for portfolio value
            market_data = self.get_market_data()
            btc_price = market_data.get('close', 45000.0) if market_data else 45000.0
            btc_value = btc_balance * btc_price
            
            return {
                "cash": usd_balance,
                "buying_power": usd_balance,
                "portfolio_value": usd_balance + btc_value,
                "equity": usd_balance + btc_value,
                "demo_mode": False,
                "balances": balances
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {"error": str(e)}
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        try:
            if self.demo_mode:
                # Return demo positions
                return {
                    "BTC": {
                        "quantity": 0.0,
                        "market_value": 0.0,
                        "unrealized_pl": 0.0,
                        "side": "long"
                    }
                }
            
            # Get real positions from Kraken
            endpoint = "/private/OpenPositions"
            result = self._make_request('POST', endpoint, {}, private=True)
            
            if 'error' in result and result['error']:
                return {"error": result['error']}
            
            positions = {}
            open_positions = result.get('result', {})
            
            # Parse open positions
            for pos_id, position in open_positions.items():
                if position.get('pair') == self.btc_symbol:
                    positions["BTC"] = {
                        "quantity": float(position.get('vol', 0)),
                        "market_value": float(position.get('vol', 0)) * float(position.get('cost', 0)),
                        "unrealized_pl": float(position.get('net', 0)),
                        "side": position.get('type', 'long')
                    }
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {"error": str(e)}
    
    def get_market_data(self, symbol: str = "XXBTZUSD") -> Optional[Dict]:
        """Get current market data for Bitcoin"""
        try:
            if self.demo_mode:
                return self._get_fallback_market_data()
            
            # Get real market data from Kraken
            endpoint = "/public/Ticker"
            params = {'pair': symbol}
            result = self._make_request('GET', endpoint, params, private=False)
            
            if 'error' in result and result['error']:
                logger.error(f"Error getting market data: {result['error']}")
                return self._get_fallback_market_data()
            
            # Parse the ticker data
            ticker_data = result.get('result', {}).get(symbol, {})
            if not ticker_data:
                return self._get_fallback_market_data()
            
            # Extract relevant data
            current_price = float(ticker_data.get('c', [0, 0])[0])  # Current price
            volume_24h = float(ticker_data.get('v', [0, 0])[1])  # 24h volume
            high_24h = float(ticker_data.get('h', [0, 0])[1])  # 24h high
            low_24h = float(ticker_data.get('l', [0, 0])[1])  # 24h low
            
            # Calculate 24h change
            open_24h = float(ticker_data.get('o', 0))  # Opening price is a string, not array
            change_24h = ((current_price - open_24h) / open_24h) * 100 if open_24h > 0 else 0
            
            return {
                'symbol': symbol,
                'close': current_price,
                'open': open_24h,
                'high': high_24h,
                'low': low_24h,
                'volume': volume_24h,
                'change_24h': change_24h,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return self._get_fallback_market_data()
    
    def _get_fallback_market_data(self) -> Dict:
        """Get fallback market data"""
        return {
            'open': 45000.0,
            'high': 46000.0,
            'low': 44000.0,
            'close': 45000.0,
            'volume': 1000000.0,
            'timestamp': datetime.now()
        }
    
    def calculate_position_size(self, dollar_amount: float, signal_strength: float = 1.0) -> float:
        """Calculate Bitcoin position size based on dollar amount"""
        try:
            # Get current Bitcoin price
            market_data = self.get_market_data()
            if market_data:
                current_price = market_data['close']
            else:
                current_price = 45000.0
            
            # Calculate Bitcoin amount
            btc_amount = dollar_amount / current_price
            
            # Ensure minimum trade size
            if btc_amount * current_price < self.min_trade_amount:
                btc_amount = self.min_trade_amount / current_price
            
            return btc_amount
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def place_buy_order(self, symbol: str, dollar_amount: float, signal_strength: float = 1.0) -> Dict:
        """Place a Bitcoin spot buy order"""
        try:
            if self.demo_mode:
                # Simulate buy order in demo mode
                market_data = self.get_market_data()
                current_price = market_data['close'] if market_data else 45000.0
                btc_amount = dollar_amount / current_price
                
                logger.info(f"Demo buy order: ${dollar_amount:.2f} ({btc_amount:.6f} BTC) at ${current_price:.2f}")
                
                return {
                    'status': 'success',
                    'order_id': f"demo_buy_{int(time.time())}",
                    'symbol': symbol,
                    'quantity': btc_amount,
                    'dollar_amount': dollar_amount,
                    'actual_amount': btc_amount * current_price,
                    'side': 'buy',
                    'timestamp': datetime.now().isoformat(),
                    'price_per_btc': current_price,
                    'demo_mode': True
                }
            
            # Get current market price to calculate BTC amount
            market_data = self.get_market_data()
            if not market_data:
                return {'status': 'error', 'reason': 'Unable to get market data'}
            
            current_price = market_data['close']
            btc_amount = dollar_amount / current_price
            
            # Place real spot order using Kraken's AddOrder endpoint
            endpoint = "/private/AddOrder"
            order_data = {
                'pair': self.btc_symbol,
                'type': 'buy',
                'ordertype': 'market',
                'volume': str(btc_amount)  # Amount in BTC
            }
            
            result = self._make_request('POST', endpoint, data=order_data, private=True, signature_path="/0/private/AddOrder")
            
            if 'error' in result and result['error']:
                return {'status': 'error', 'reason': result['error']}
            
            # Parse the response
            order_info = result.get('result', {})
            txid = order_info.get('txid', [None])[0] if order_info.get('txid') else None
            
            return {
                'status': 'success',
                'order_id': txid,
                'symbol': symbol,
                'quantity': btc_amount,
                'dollar_amount': dollar_amount,
                'side': 'buy',
                'timestamp': datetime.now().isoformat(),
                'price_per_btc': current_price,
                'demo_mode': False
            }
            
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def place_sell_order(self, symbol: str, dollar_amount: float) -> Dict:
        """Place a Bitcoin spot sell order"""
        try:
            if self.demo_mode:
                # Simulate sell order in demo mode
                market_data = self.get_market_data()
                current_price = market_data['close'] if market_data else 45000.0
                btc_amount = dollar_amount / current_price
                
                logger.info(f"Demo sell order: ${dollar_amount:.2f} ({btc_amount:.6f} BTC) at ${current_price:.2f}")
                
                return {
                    'status': 'success',
                    'order_id': f"demo_sell_{int(time.time())}",
                    'symbol': symbol,
                    'quantity': btc_amount,
                    'dollar_amount': dollar_amount,
                    'side': 'sell',
                    'timestamp': datetime.now().isoformat(),
                    'price_per_btc': current_price,
                    'demo_mode': True
                }
            
            # Get current market price to calculate BTC amount
            market_data = self.get_market_data()
            if not market_data:
                return {'status': 'error', 'reason': 'Unable to get market data'}
            
            current_price = market_data['close']
            btc_amount = dollar_amount / current_price
            
            # Place real spot order using Kraken's AddOrder endpoint
            endpoint = "/private/AddOrder"
            order_data = {
                'pair': self.btc_symbol,
                'type': 'sell',
                'ordertype': 'market',
                'volume': str(btc_amount)  # Amount in BTC
            }
            
            result = self._make_request('POST', endpoint, data=order_data, private=True, signature_path="/0/private/AddOrder")
            
            if 'error' in result and result['error']:
                return {'status': 'error', 'reason': result['error']}
            
            # Parse the response
            order_info = result.get('result', {})
            txid = order_info.get('txid', [None])[0] if order_info.get('txid') else None
            
            return {
                'status': 'success',
                'order_id': txid,
                'symbol': symbol,
                'quantity': btc_amount,
                'dollar_amount': dollar_amount,
                'side': 'sell',
                'timestamp': datetime.now().isoformat(),
                'price_per_btc': current_price,
                'demo_mode': False
            }
            
        except Exception as e:
            logger.error(f"Error placing sell order: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def place_gbp_buy_order(self, gbp_amount: float) -> Dict:
        """Place a Bitcoin buy order using GBP"""
        try:
            if self.demo_mode:
                # Return demo order result
                return {
                    'status': 'success',
                    'order_id': f"demo_gbp_{int(time.time())}",
                    'symbol': self.btc_gbp_symbol,
                    'quantity': gbp_amount / 45000.0,  # Demo BTC amount
                    'gbp_amount': gbp_amount,
                    'side': 'buy',
                    'timestamp': datetime.now().isoformat(),
                    'demo_mode': True
                }
            
            # Get current market price to calculate BTC amount
            market_data = self.get_market_data(self.btc_gbp_symbol)
            if not market_data:
                return {'status': 'error', 'reason': 'Unable to get market data'}
            
            current_price = market_data['close']
            btc_amount = gbp_amount / current_price
            
            # Place real spot order using Kraken's AddOrder endpoint
            endpoint = "/private/AddOrder"
            order_data = {
                'pair': self.btc_gbp_symbol,
                'type': 'buy',
                'ordertype': 'market',
                'volume': str(btc_amount)  # Amount in BTC
            }
            
            result = self._make_request('POST', endpoint, data=order_data, private=True, signature_path="/0/private/AddOrder")
            
            if 'error' in result and result['error']:
                return {'status': 'error', 'reason': result['error']}
            
            # Parse the response
            order_info = result.get('result', {})
            txid = order_info.get('txid', [None])[0] if order_info.get('txid') else None
            
            return {
                'status': 'success',
                'order_id': txid,
                'symbol': self.btc_gbp_symbol,
                'quantity': btc_amount,
                'gbp_amount': gbp_amount,
                'side': 'buy',
                'timestamp': datetime.now().isoformat(),
                'price_per_btc_gbp': current_price,
                'demo_mode': False
            }
            
        except Exception as e:
            logger.error(f"Error placing GBP buy order: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def close_all_positions(self) -> Dict:
        """Close all Bitcoin positions"""
        try:
            positions = self.get_positions()
            closed_positions = []
            
            for symbol, position in positions.items():
                if symbol == "BTC" and position['quantity'] > 0:
                    # Close position
                    if position['side'] == 'long':
                        result = self.place_sell_order(symbol, position['market_value'])
                    else:
                        result = self.place_buy_order(symbol, position['market_value'])
                    
                    if result['status'] == 'success':
                        closed_positions.append({
                            'symbol': symbol,
                            'quantity': position['quantity'],
                            'order_id': result.get('order_id', '')
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
            
            total_positions = len(positions) if not isinstance(positions, dict) or 'error' not in positions else 0
            total_value = sum(pos['market_value'] for pos in positions.values()) if isinstance(positions, dict) and 'error' not in positions else 0
            total_unrealized_pl = sum(pos['unrealized_pl'] for pos in positions.values()) if isinstance(positions, dict) and 'error' not in positions else 0
            
            return {
                "account": account_info,
                "positions": positions,
                "summary": {
                    "total_positions": total_positions,
                    "total_value": total_value,
                    "total_unrealized_pl": total_unrealized_pl,
                    "max_positions": self.max_positions,
                    "position_size": self.position_size,
                    "demo_mode": self.demo_mode
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting trading summary: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Test the Kraken trading bot
    bot = KrakenTradingBot()
    print("Kraken Bitcoin Trading Bot Test:")
    print(f"Account Info: {bot.get_account_info()}")
    print(f"Positions: {bot.get_positions()}")
    print(f"Market Data: {bot.get_market_data()}")
    print(f"Trading Summary: {bot.get_trading_summary()}")
