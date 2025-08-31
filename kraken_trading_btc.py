#!/usr/bin/env python3
"""
Kraken Bitcoin Trading Bot
Specialized for Bitcoin trading with Kraken API (futures demo environment).
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List, Tuple
import requests
import time
import hmac
import hashlib
import base64
import urllib.parse
import uuid
import random
from math import floor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Trading constants
KRAKEN_PAIR = "XXBTZUSD"
QUOTE_CCY   = "ZUSD"
FALLBACK_LOT_STEP = 1e-5
FALLBACK_TICK     = 0.10
FALLBACK_OB_DEPTH = 10
EXCHANGE_MIN_NOTIONAL_USD = 10.0

# Kraken public API endpoint
K_PUBLIC = "https://api.kraken.com/0/public"

# --- Pair metadata + rounding helpers ---
from functools import lru_cache
import math

@lru_cache(maxsize=16)
def get_pair_meta(api) -> dict:
    """
    Fetches tick/lot and minimums once and caches.
    Returns: {'pair': 'XXBTZUSD', 'tick': float, 'lot_step': float, 'ordermin': float, 'price_dec': int, 'lot_dec': int}
    """
    try:
        info = api._make_request_with_retry('GET', '/public/AssetPairs', data={"pair": KRAKEN_PAIR})
        d = next(iter(info["result"].values()))
        tick = 10 ** (-int(d.get("pair_decimals", 2)))
        lot_step = 10 ** (-int(d.get("lot_decimals", 8)))
        ordermin = float(d.get("ordermin", lot_step))
        return {
            "pair": KRAKEN_PAIR,
            "tick": tick,
            "lot_step": lot_step,
            "ordermin": ordermin,
            "price_dec": int(d.get("pair_decimals", 2)),
            "lot_dec": int(d.get("lot_decimals", 8)),
        }
    except Exception as e:
        logger.warning(f"Failed to get pair metadata, using fallbacks: {e}")
        return {
            "pair": KRAKEN_PAIR,
            "tick": FALLBACK_TICK,
            "lot_step": FALLBACK_LOT_STEP,
            "ordermin": FALLBACK_LOT_STEP,
            "price_dec": 2,
            "lot_dec": 8,
        }

def round_down(v: float, step: float) -> float:
    return math.floor(v / step) * step

def round_up(v: float, step: float) -> float:
    return math.ceil(v / step) * step

def clamp_qty_to_limits(qty: float, price: float, meta: dict, min_notional_usd: float) -> float:
    # Kraken also enforces ordermin (in BTC). Respect ordermin and min_notional
    min_qty = max(meta["ordermin"], (min_notional_usd / max(price, 1e-9)))
    q = max(qty, min_qty)
    return max(round_down(q, meta["lot_step"]), 0.0)

def get_funds_sanitized(api, price: float, side: str, meta: dict, safety_buf: float = 0.995) -> float:
    """Return max BTC you can trade after buffers and rounding."""
    try:
        bal = api._make_request_with_retry('POST', '/private/Balance', private=True, signature_path="/0/private/Balance")
        bal_result = bal["result"]
        usd = float(bal_result.get(QUOTE_CCY, 0.0))
        btc = float(bal_result.get("XXBT", 0.0))  # Kraken asset code for BTC
        if side == "buy":
            max_qty = (usd * safety_buf) / price
        else:
            max_qty = btc * safety_buf
        return max(round_down(max_qty, meta["lot_step"]), 0.0)
    except Exception as e:
        logger.warning(f"Failed to get funds, using fallback: {e}")
        return 0.0

def best_quotes(api) -> dict:
    """
    Return top-of-book for the configured pair using Kraken 'Depth' endpoint.
    Handles any key name Kraken returns (XXBTZUSD / XBTUSD).
    """
    try:
        ob = api._make_request_with_retry('GET', '/public/Depth', data={"pair": KRAKEN_PAIR, "count": 1})
        if ob.get("error"):
            raise RuntimeError(f"Kraken Depth error: {ob['error']}")
        result = ob.get("result", {})
        if not result:
            raise RuntimeError("Empty result from Kraken Depth")
        pair_key = next(iter(result.keys()))  # don't assume exact key name
        bids = result[pair_key].get("bids", [])
        asks = result[pair_key].get("asks", [])
        if not bids or not asks:
            raise RuntimeError("Empty book from Kraken Depth")
        bid = float(bids[0][0])
        ask = float(asks[0][0])
        mid = 0.5 * (bid + ask)
        return {"bid": bid, "ask": ask, "mid": mid}
    except Exception as e:
        logger.warning(f"Failed to get best quotes, using fallback: {e}")
        return {"bid": 100000.0, "ask": 100001.0, "mid": 100000.5}


# --- Fill watchdog: cancel & reprice stale post-only orders ---
def wait_for_fill_or_reprice(api, txid: str, side: str, meta: dict,
                             max_open_mins: int = 3, reprice_ticks: int = 2) -> dict:
    deadline = time.time() + max_open_mins*60
    last_status = None
    while time.time() < deadline:
        # Query order
        try:
            q = api._make_request_with_retry('POST', '/private/QueryOrders', data={"txid": txid}, private=True, signature_path="/0/private/QueryOrders")
            if q.get("error"): break
            od = q["result"][txid]
            status = od.get("status")              # open|closed|canceled
            last_status = status
            if status == "closed":                 # filled
                return {"status":"filled", "txid": txid, "descr": od.get("descr")}
        except Exception as e:
            logger.warning(f"Failed to query order {txid}: {e}")
            break
        # brief sleep
        time.sleep(5)

    # Not filled in time: cancel + reprice further from crossing
    try:
        api._make_request_with_retry('POST', '/private/CancelOrder', data={"txid": txid}, private=True, signature_path="/0/private/CancelOrder")
        book = best_quotes(api)
        tick = meta["tick"]
        if side == "buy":
            px = round_down(book["bid"] - reprice_ticks*tick, tick)
        else:
            px = round_up(book["ask"] + reprice_ticks*tick, tick)

        # Re-submit with same qty as before
        vol = od.get("vol") or od.get("vol_exec") or "0"
        payload = {
            "pair": meta["pair"], "type": "buy" if side=="buy" else "sell",
            "ordertype": "limit", "price": f"{px:.{meta['price_dec']}f}",
            "volume": vol, "oflags": "post", "timeinforce":"GTC",
            "userref": int(time.time())
        }
        
        # Log the reprice attempt
        logger.info("WATCHDOG reprice order_id=%s ticks=%d", txid, reprice_ticks)
        
        r2 = api._make_request_with_retry('POST', '/private/AddOrder', data=payload, private=True, signature_path="/0/private/AddOrder")
        if r2.get("error"):
            logger.warning("REPRICE_FAIL %s", r2["error"])
            return {"status":"reprice_error", "detail": r2["error"]}
        return {"status":"repriced", "result": r2.get("result")}
    except Exception as e:
        logger.warning(f"Failed to reprice order {txid}: {e}")
        return {"status":"reprice_error", "detail": str(e)}

# Asset code normalization for Kraken
_ASSET_MAP = {
    "XXBT": "BTC", "XBT": "BTC", "BTC": "BTC",
    "ZUSD": "USD", "USD": "USD",
    "ZEUR": "EUR", "EUR": "EUR",
    "ZGBP": "GBP", "GBP": "GBP",
}

# Helper functions
def _now_iso(): 
    return datetime.now(timezone.utc).isoformat()

def _round_qty(q, step): 
    return floor(float(q) / step) * step

def _round_price(p, tick): 
    return round(float(p) / tick) * tick if tick > 0 else float(p)

def _slip_cap(best, side, bps): 
    m = (1 + bps/10_000) if side == "buy" else (1 - bps/10_000)
    return best * m

def _parse_ok(resp: dict) -> bool:
    if not isinstance(resp, dict): 
        return False
    return resp.get("status") == "success" or bool(resp.get("txid"))

def _normalize_asset(symbol: str) -> str:
    return _ASSET_MAP.get(symbol, symbol)

def get_balances_normalized() -> dict[str, float]:
    """Get normalized balances from Kraken"""
    try:
        # Create a temporary bot instance to get account info
        bot = KrakenTradingBot()
        account_info = bot.get_account_info()
        raw = account_info.get('balances', {})
    except Exception as e:
        logger.warning(f"Failed to get balances: {e}")
        raw = {}
    
    balances = {}
    for k, v in (raw or {}).items():
        sym = _normalize_asset(k)
        try:
            balances[sym] = balances.get(sym, 0.0) + float(v)
        except (ValueError, TypeError):
            pass
    return balances

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
        self.position_size = float(os.getenv('POSITION_SIZE') or '1000')  # Default $1000 per trade
        self.max_positions = int(os.getenv('MAX_POSITIONS') or '3')  # Maximum concurrent positions
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT') or '0.05')  # 5% stop loss
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT') or '0.10')  # 10% take profit
        
        # Bitcoin spot trading symbol
        self.btc_symbol = "XXBTZUSD"  # Bitcoin/USD pair on Kraken
        self.btc_gbp_symbol = "XXBTZGBP"  # Bitcoin/GBP pair on Kraken
        self.min_trade_amount = float(os.getenv('MIN_TRADE_AMOUNT_USD') or '10.0')  # Minimum trade amount in USD
        self.max_slippage_pct = float(os.getenv('MAX_SLIPPAGE_PCT') or '0.02')  # 2% max slippage
        self.max_retries = int(os.getenv('MAX_RETRIES') or '3')
        self.retry_delay_base = float(os.getenv('RETRY_DELAY_BASE') or '1.0')  # Base delay in seconds
        # Allocation logic parameters
        # Prefer MIN_TRADE_DELTA if provided, otherwise fall back to REALLOC_THRESHOLD_PCT
        self.rebalance_threshold_pct = float(os.getenv('MIN_TRADE_DELTA') or os.getenv('REALLOC_THRESHOLD_PCT') or '0.05')
        self.trade_cooldown_hours = float(os.getenv('TRADE_COOLDOWN_HOURS') or '1')
        self._last_trade_ts = 0.0
        # Track executed orders to prevent duplicates
        self._executed_orders = set()
        
        # Check if credentials are available
        if not self.api_key or not self.secret_key:
            logger.warning("Kraken API credentials not found - using demo mode only")
            self.demo_mode = True
        else:
            self.demo_mode = False
        
        logger.info("Kraken Bitcoin trading bot initialized successfully")
    
    def get_ticker(self, pair: str = "XBTUSD") -> tuple[float, float, float]:
        """
        Return (bid, ask, mid) using Kraken /Ticker.
        We pass 'XBTUSD' and read the first key of result (usually 'XXBTZUSD').
        """
        try:
            r = requests.get(f"{K_PUBLIC}/Ticker", params={"pair": pair}, timeout=5)
            r.raise_for_status()
            data = r.json()["result"]
            book = next(iter(data.values()))
            bid = float(book["b"][0][0])
            ask = float(book["a"][0][0])
            mid = (bid + ask) / 2.0
            return bid, ask, mid
        except Exception as e:
            logger.warning(f"get_ticker failed: {e}")
            raise
    
    def get_top_of_book(self, pair: str = "XBTUSD") -> tuple[float, float, float]:
        """
        Return (bid, ask, mid) using Kraken /Depth endpoint.
        """
        try:
            r = requests.get(f"{K_PUBLIC}/Depth", params={"pair": pair, "count": 1}, timeout=5)
            r.raise_for_status()
            data = r.json()["result"]
            book = next(iter(data.values()))
            bid = float(book["bids"][0][0])
            ask = float(book["asks"][0][0])
            mid = (bid + ask) / 2.0
            return bid, ask, mid
        except Exception as e:
            logger.warning(f"get_top_of_book failed: {e}")
            raise
    
    def _get_pair_meta(self, pair=KRAKEN_PAIR):
        """Get pair metadata for lot step and tick size"""
        try:
            # Try to get from Kraken API
            endpoint = "/public/AssetPairs"
            result = self._make_request_with_retry('GET', endpoint, private=False)
            if result and 'result' in result:
                pair_info = result['result'].get(pair, {})
                lot_step = float(pair_info.get("lot_step", FALLBACK_LOT_STEP))
                tick = float(pair_info.get("tick_size", FALLBACK_TICK))
                return lot_step, tick
        except Exception:
            pass
        return FALLBACK_LOT_STEP, FALLBACK_TICK
    
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

    def _make_request_with_retry(self, method: str, endpoint: str, data: Dict = None, private: bool = False, signature_path: str = None) -> Dict:
        """Make API request with exponential backoff retry logic"""
        for attempt in range(self.max_retries + 1):
            try:
                result = self._make_request(method, endpoint, data, private, signature_path)
                
                # Check if it's a transient error that should be retried
                if self._should_retry(result, attempt):
                    if attempt < self.max_retries:
                        delay = self.retry_delay_base * (2 ** attempt) + random.uniform(0, 0.1)
                        logger.warning(f"Retry {attempt + 1}/{self.max_retries} after {delay:.2f}s for {endpoint}")
                        time.sleep(delay)
                        continue
                
                return result
                
            except Exception as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay_base * (2 ** attempt) + random.uniform(0, 0.1)
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries} after {delay:.2f}s for {endpoint}: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"Final attempt failed for {endpoint}: {e}")
                    return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}

    def _should_retry(self, result: Dict, attempt: int) -> bool:
        """Determine if a request should be retried based on the response"""
        if 'error' in result and result['error']:
            error_msg = str(result['error']).lower()
            # Retry on rate limits, temporary server errors, network issues
            retryable_errors = [
                'rate limit', 'too many requests', 'temporary', 'timeout',
                'connection', 'network', 'server error', 'service unavailable'
            ]
            return any(err in error_msg for err in retryable_errors)
        return False

    def _generate_client_order_id(self, side: str, amount: float) -> str:
        """Generate a unique client order ID for idempotency"""
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        return f"btc_{side}_{timestamp}_{unique_id}"

    def _check_slippage(self, decision_price: float, current_price: float) -> Tuple[bool, float]:
        """Check if slippage is within acceptable bounds"""
        if decision_price <= 0:
            return True, 0.0
        
        slippage_pct = abs(current_price - decision_price) / decision_price
        is_acceptable = slippage_pct <= self.max_slippage_pct
        
        logger.info(f"Slippage check: decision=${decision_price:.2f}, current=${current_price:.2f}, slippage={slippage_pct:.3%}, acceptable={is_acceptable}")
        return is_acceptable, slippage_pct

    def _validate_order_size(self, dollar_amount: float, btc_amount: float, current_price: float) -> Tuple[bool, str]:
        """Validate order size meets minimum requirements"""
        if dollar_amount < self.min_trade_amount:
            return False, f"Order size ${dollar_amount:.2f} below minimum ${self.min_trade_amount:.2f}"
        
        # Kraken minimum BTC order size (varies by pair)
        min_btc_amount = 0.0001  # Kraken's typical minimum
        if btc_amount < min_btc_amount:
            return False, f"BTC amount {btc_amount:.6f} below minimum {min_btc_amount:.6f}"
        
        return True, "Order size valid"
    
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
            result = self._make_request_with_retry('POST', endpoint, {}, private=True, signature_path="/0/private/Balance")
            
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
            result = self._make_request_with_retry('POST', endpoint, {}, private=True)
            
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
            result = self._make_request_with_retry('GET', endpoint, params, private=False)
            
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

    def compute_exposure_summary(self, balances: Dict, last_price: float, equity: float) -> Tuple[float, float]:
        """Compute exposure value and percentage using live balances and consistent price.
        
        Args:
            balances: Live account balances from Kraken
            last_price: The same price used for PnL calculations
            equity: Total account equity
            
        Returns:
            Tuple of (exposure_value_usd, exposure_percentage)
        """
        btc_qty = float(balances.get("XXBT", 0.0))
        exposure_value = btc_qty * last_price if last_price else 0.0
        exposure_pct = (exposure_value / equity) if (equity and equity > 0) else 0.0
        return exposure_value, exposure_pct

    def get_current_exposure(self) -> Dict:
        """Return current BTC exposure as fraction of equity in [0,1]."""
        acct = self.get_account_info()
        if isinstance(acct, dict) and 'equity' in acct:
            equity = float(acct.get('equity') or 0.0)
            mkt = self.get_market_data() or {}
            price = float(mkt.get('close', 0.0) or 0.0)
            balances = acct.get('balances', {}) or {}
            
            # Use the new compute_exposure_summary function
            exposure_value, exposure_frac = self.compute_exposure_summary(balances, price, equity)
            btc_qty = float(balances.get('XXBT', '0') or 0.0)
            
            return {"equity": equity, "btc_qty": btc_qty, "price": price, "exposure_frac": exposure_frac}
        return {"equity": 0.0, "btc_qty": 0.0, "price": 0.0, "exposure_frac": 0.0}

    def rebalance_to_target(self, target_exposure: float, decision_price: float = None) -> Dict:
        """Rebalance holdings to reach target exposure in [0,1].

        - Applies a 5% (configurable) threshold on equity to skip small changes
        - Enforces cooldown to avoid frequent trading
        - Does not short: target < 0 means move towards cash
        - Includes slippage guards and client order ID tracking
        """
        now = time.time()
        if self._last_trade_ts and (now - self._last_trade_ts) < self.trade_cooldown_hours * 3600:
            return {"status": "skipped", "reason": f"Cooldown active ({self.trade_cooldown_hours}h)"}

        # Clamp to [0,1] for long-only spot
        target_long = max(0.0, min(1.0, float(target_exposure)))
        state = self.get_current_exposure()
        equity = state["equity"]
        current_price = state["price"]
        current = state["exposure_frac"]
        delta = target_long - current

        # Check slippage if decision price provided
        if decision_price and decision_price > 0:
            slippage_ok, slippage_pct = self._check_slippage(decision_price, current_price)
            if not slippage_ok:
                return {"status": "skipped", "reason": f"Slippage {slippage_pct:.3%} exceeds limit {self.max_slippage_pct:.3%}", "current": current, "target": target_long}

        # Calculate dollar delta and apply dynamic min trade delta logic
        dollar_delta = abs(delta) * equity
        
        # Use centralized risk management function for dynamic delta calculation
        try:
            from risk_management import effective_min_delta_usd
            
            # Create settings dict from bot attributes
            settings = {
                'no_fee_mode': getattr(self, 'no_fee_mode', True),
                'min_trade_delta_usd': getattr(self, 'min_trade_delta_usd', 10.0),
                'min_trade_delta_pct': getattr(self, 'min_trade_delta_pct', 0.00)
            }
            
            min_delta = effective_min_delta_usd(equity, settings)
        except ImportError:
            # Fallback to local calculation if risk_management module not available
            no_fee_mode = getattr(self, 'no_fee_mode', True)
            min_trade_delta_usd = getattr(self, 'min_trade_delta_usd', 10.0)
            min_trade_delta_pct = getattr(self, 'min_trade_delta_pct', 0.00)
            
            if no_fee_mode:
                min_delta = max(min_trade_delta_usd, min_trade_delta_pct * equity)
            else:
                min_delta = max(min_trade_delta_usd, min_trade_delta_pct * equity, 30.0)
        
        if dollar_delta < min_delta:
            logger.info("Skip: delta $%.2f < min $%.2f", dollar_delta, min_delta)
            return {"status": "skipped", "reason": f"min_trade_delta", "current": current, "target": target_long, "delta_usd": dollar_delta, "min_delta": min_delta}

        # Generate client order ID for idempotency
        client_order_id = self._generate_client_order_id('rebalance', dollar_delta)
        
        # Check if we've already executed this order
        if client_order_id in self._executed_orders:
            return {"status": "skipped", "reason": f"Order {client_order_id} already executed", "current": current, "target": target_long}

        # Decide buy or sell based on delta sign
        if delta > 0:
            # Need to buy BTC worth dollar_delta
            result = self.place_buy_order(self.btc_symbol, dollar_delta)
        else:
            # Need to sell BTC worth dollar_delta
            result = self.place_sell_order(self.btc_symbol, dollar_delta)

        if isinstance(result, dict) and result.get('status') == 'success':
            self._last_trade_ts = now
            self._executed_orders.add(client_order_id)
            # Keep only last 1000 orders to prevent memory bloat
            if len(self._executed_orders) > 1000:
                self._executed_orders = set(list(self._executed_orders)[-1000:])
            
            # After every executed order: refresh balances and recompute exposure
            try:
                # 1) Refresh balances from Kraken
                acct = self.get_account_info()
                if isinstance(acct, dict) and 'balances' in acct:
                    balances = acct.get('balances', {})
                    equity = float(acct.get('equity', 0.0))
                    
                    # 2) Recompute exposure using the SAME last_price used for PnL
                    # Use decision_price if provided, otherwise get current market price
                    exposure_price = decision_price if decision_price else current_price
                    exposure_value, exposure_pct = self.compute_exposure_summary(balances, exposure_price, equity)
                    
                    logger.info(f"Post-trade exposure: ${exposure_value:.2f} ({exposure_pct:.3%}) at price ${exposure_price:.2f}")
                    
                    # 3) Update result with fresh exposure data
                    result['post_trade_exposure'] = {
                        'exposure_value_usd': exposure_value,
                        'exposure_percentage': exposure_pct,
                        'price_used': exposure_price,
                        'equity': equity
                    }
            except Exception as e:
                logger.warning(f"Failed to refresh post-trade exposure: {e}")
                
        return {"status": "executed" if result.get('status') == 'success' else 'error', "order": result, "current": current, "target": target_long, "delta": delta, "client_order_id": client_order_id}
    
    def place_buy_order(self, symbol: str = "XXBTZUSD", dollar_amount: float = 0.0, signal_strength: float = 1.0) -> Dict:
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
            
            # Validate order size
            is_valid, reason = self._validate_order_size(dollar_amount, btc_amount, current_price)
            if not is_valid:
                return {'status': 'error', 'reason': reason}

            # Generate client order ID
            client_order_id = self._generate_client_order_id('buy', dollar_amount)

            # Use maker-first order placement with fallback to market
            result = self.place_maker_first_order('buy', btc_amount, self.btc_symbol)
            
            if result.get('status') == 'success':
                # Add additional fields for compatibility
                result.update({
                    'symbol': symbol,
                    'dollar_amount': dollar_amount,
                    'timestamp': datetime.now().isoformat(),
                    'price_per_btc': current_price
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def place_sell_order(self, symbol: str = "XXBTZUSD", dollar_amount: float = 0.0) -> Dict:
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
            
            # Validate order size
            is_valid, reason = self._validate_order_size(dollar_amount, btc_amount, current_price)
            if not is_valid:
                return {'status': 'error', 'reason': reason}

            # Generate client order ID
            client_order_id = self._generate_client_order_id('sell', dollar_amount)

            # Use maker-first order placement with fallback to market
            result = self.place_maker_first_order('sell', btc_amount, self.btc_symbol)
            
            if result.get('status') == 'success':
                # Add additional fields for compatibility
                result.update({
                    'symbol': symbol,
                    'dollar_amount': dollar_amount,
                    'timestamp': datetime.now().isoformat(),
                    'price_per_btc': current_price
                })
            
            return result
            
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
            
            # Validate order size
            is_valid, reason = self._validate_order_size(gbp_amount, btc_amount, current_price)
            if not is_valid:
                return {'status': 'error', 'reason': reason}

            # Generate client order ID
            client_order_id = self._generate_client_order_id('buy', gbp_amount)

            # Use maker-first order placement with fallback to market
            result = self.place_maker_first_order('buy', btc_amount, self.btc_gbp_symbol)
            
            if result.get('status') == 'success':
                # Add additional fields for compatibility
                result.update({
                    'symbol': self.btc_gbp_symbol,
                    'gbp_amount': gbp_amount,
                    'timestamp': datetime.now().isoformat(),
                    'price_per_btc_gbp': current_price
                })
            
            return result
            
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
    
    def get_orderbook_snapshot(self, symbol: str = "XXBTZUSD") -> Dict:
        """Get current orderbook snapshot for price discovery via Depth."""
        try:
            endpoint = "/public/Depth"
            data = {'pair': symbol, 'count': 1}
            result = self._make_request_with_retry('GET', endpoint, data=data, private=False)
            if 'error' in result and result['error']:
                return {'status': 'error', 'reason': result['error']}
            res = result.get('result', {})
            if not res:
                return {'status': 'error', 'reason': 'Empty result from Depth'}
            # pick whatever key Kraken returns (e.g., XXBTZUSD or XBTUSD)
            pair_key = next(iter(res.keys()))
            orderbook = res.get(pair_key, {})
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            best_bid = float(bids[0][0]) if bids else None
            best_ask = float(asks[0][0]) if asks else None
            return {
                'status': 'success',
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None,
                'spread_pct': (((best_ask - best_bid) / best_bid) * 100) if (best_bid and best_ask) else None
            }
        except Exception as e:
            logger.error(f"Error getting orderbook snapshot: {e}")
            return {'status': 'error', 'reason': str(e)}

    def place_limit_post_only(self, side: str, qty: float, price_ref: float) -> tuple[bool, dict]:
        """Robust post-only limit with maker-safe price and retries"""
        api = self  # Use self as the API client
        meta = get_pair_meta(api)
        qts = best_quotes(api)
        tick = meta["tick"]

        # maker-safe price (avoid crossing); if spread < tick, step one more tick
        if side == "buy":
            px = min(qts["bid"], qts["ask"] - tick)
            if qts["ask"] - qts["bid"] < tick: px = qts["bid"] - tick
            px = round_down(px, tick)
        else:
            px = max(qts["ask"], qts["bid"] + tick)
            if qts["ask"] - qts["bid"] < tick: px = qts["ask"] + tick
            px = round_up(px, tick)

        # clamp qty to limits and available funds, apply rounding
        qty = clamp_qty_to_limits(qty, qts["mid"], meta, min_notional_usd=self.min_trade_amount)
        qty = min(qty, get_funds_sanitized(api, qts["mid"], side, meta))
        qty = round_down(qty, meta["lot_step"])
        if qty <= 0:
            return False, {"status": "error", "reason": "qty_le_0_after_limits"}

        # Prepare order payload
        payload = {
            "pair": meta["pair"],
            "type": "buy" if side == "buy" else "sell",
            "ordertype": "limit",
            "price": f"{px:.{meta['price_dec']}f}",
            "volume": f"{qty:.{meta['lot_dec']}f}",
            "oflags": "post",           # post-only
            "userref": int(time.time()),
            "timeinforce": "GTC",
            # "validate": True,         # enable for dry-run if needed
        }

        try:
            endpoint = "/private/AddOrder"
            res = self._make_request_with_retry('POST', endpoint, data=payload, private=True, signature_path="/0/private/AddOrder")
            
            if res.get("error"):
                err = "|".join(res["error"])
                # Common auto-fix paths
                if "EOrder:Post only" in err or "EOrder:Cannot post only" in err:
                    # We would have taken liquidity; step one more tick away and retry once.
                    px2 = round_down(px - tick, tick) if side == "buy" else round_up(px + tick, tick)
                    payload["price"] = f"{px2:.{meta['price_dec']}f}"
                    res2 = self._make_request_with_retry('POST', endpoint, data=payload, private=True, signature_path="/0/private/AddOrder")
                    if res2.get("error"):
                        return False, {"status": "error", "reason": "post_only_reject", "detail": res2["error"]}
                    return True, {"status": "success", "mode":"post_only", "result": res2.get("result", {})}
                if "EOrder:Insufficient funds" in err:
                    # downsize 5% and try once
                    qty2 = round_down(qty * 0.95, meta["lot_step"])
                    if qty2 > 0:
                        payload["volume"] = f"{qty2:.{meta['lot_dec']}f}"
                        res2 = self._make_request_with_retry('POST', endpoint, data=payload, private=True, signature_path="/0/private/AddOrder")
                        if res2.get("error"):
                            return False, {"status": "error", "reason": "insufficient_funds", "detail": res2["error"]}
                        return True, {"status": "success", "mode":"post_only", "result": res2.get("result", {})}
                return False, {"status": "error", "reason": "addorder_error", "detail": res["error"]}

            result = res.get("result", {})
            # Start fill watchdog for post-only orders
            if result.get("txid"):
                txid = result["txid"][0] if isinstance(result["txid"], list) else result["txid"]
                import os
                watchdog = wait_for_fill_or_reprice(api, txid, side, meta,
                                                    max_open_mins=int(os.getenv("MAX_OPEN_MINS", 3)),
                                                    reprice_ticks=int(os.getenv("REPRICE_TICKS", 2)))
                logger.info("WATCHDOG %s", watchdog)
                result["watchdog"] = watchdog
            
            return True, {"status": "success", "mode":"post_only", "result": result}
        except Exception as e:
            return False, {"status": "error", "reason": str(e)}

    def place_market_with_slippage_cap(self, side: str, qty: float, max_slippage_bps: int = 10) -> dict:
        """Market order with slippage cap (synthetic IOC limit)"""
        api = self
        meta = get_pair_meta(api)
        qts = best_quotes(api)
        tick = meta["tick"]

        qty = round_down(qty, meta["lot_step"])
        if qty <= 0:
            return {"status": "error", "reason": "qty_le_0"}

        if side == "buy":
            cap = qts["ask"] * (1.0 + max_slippage_bps/1e4)
            px  = round_up(cap, tick)
            ordertype = "limit"   # IOC limit acts like slippage-capped market
            tif = "IOC"
        else:
            cap = qts["bid"] * (1.0 - max_slippage_bps/1e4)
            px  = round_down(cap, tick)
            ordertype = "limit"
            tif = "IOC"

        payload = {
            "pair": meta["pair"],
            "type": "buy" if side == "buy" else "sell",
            "ordertype": ordertype,
            "price": f"{px:.{meta['price_dec']}f}",
            "volume": f"{qty:.{meta['lot_dec']}f}",
            "timeinforce": tif,
            "userref": int(time.time()),
            # "validate": True,
        }
        
        try:
            endpoint = "/private/AddOrder"
            res = self._make_request_with_retry('POST', endpoint, data=payload, private=True, signature_path="/0/private/AddOrder")
            if res.get("error"):
                return {"status":"error","reason":"addorder_error","detail":res["error"]}
            return {"status": "success", "mode":"ioc_limit_cap", "result": res.get("result", {})}
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    def place_maker_first_order(self, side: str, qty: float) -> dict:
        """Place a maker-first order with fallback to market order"""
        try:
            # Get current orderbook for price discovery
            orderbook = self.get_orderbook_snapshot(KRAKEN_PAIR)
            if orderbook.get('status') != 'success':
                logger.warning(f"Could not get orderbook, falling back to market order: {orderbook.get('reason')}")
                return self.place_market_with_slippage_cap(side, qty, max_slippage_bps=10)
            
            # Determine limit price based on side
            if side == 'buy':
                limit_price = orderbook['best_ask']  # Buy at ask to add liquidity
            else:  # sell
                limit_price = orderbook['best_bid']  # Sell at bid to add liquidity
            
            if not limit_price:
                logger.warning("No valid limit price available, falling back to market order")
                return self.place_market_with_slippage_cap(side, qty, max_slippage_bps=10)
            
            # Try post-only limit order first
            logger.info(f"Attempting post-only {side} order: {qty:.6f} BTC at ${limit_price:.2f}")
            ok, post_only_result = self.place_limit_post_only(side, qty, limit_price)
            
            if ok and post_only_result.get('status') == 'success':
                logger.info(f"✅ Post-only {side} order placed successfully")
                return post_only_result
            else:
                logger.warning(f"❌ Post-only order failed: {post_only_result.get('reason')}")
                # Fallback to market order with slippage cap
                logger.info(f"Falling back to market order with 0.10% slippage cap")
                return self.place_market_with_slippage_cap(side, qty, max_slippage_bps=10)
                
        except Exception as e:
            logger.error(f"Error in maker-first order placement: {e}")
            # Fallback to market order
            return self.place_market_with_slippage_cap(side, qty, max_slippage_bps=10)

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
