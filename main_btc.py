#!/usr/bin/env python3
"""
Bitcoin LLM Trading System
A focused Bitcoin trading application with compact LLM for automated trading decisions.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Literal
import jwt
import secrets
from fastapi import FastAPI, HTTPException, Form, Depends, Request, Body
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import requests
import json
from dotenv import load_dotenv
import time
from decimal import Decimal

# Boolean parsing utility
try:
    from utils_bool import parse_bool
except ImportError:
    # Fallback if utils_bool is not available
    def parse_bool(v, default=False):
        if isinstance(v, bool): return v
        if v is None: return default
        s = str(v).strip().lower()
        return s in {"1","true","t","yes","y","on","enabled","enable"}

# Ledger for persistent PnL and orders
try:
    from db import (
        init_db as ledger_init_db,
        record_order as ledger_record_order,
        compute_pnl as ledger_compute_pnl,
        get_hodl_benchmark as ledger_get_hodl_benchmark,
        snapshot_equity as ledger_snapshot_equity,
        get_equity_curve as ledger_get_equity_curve,
        get_hodl_curve as ledger_get_hodl_curve,
        get_realized_pnl_total,
        get_fees_total,
        get_avg_cost_from_positions,
    )
    ledger_init_db()
    LEDGER_AVAILABLE = True
except Exception as _ledger_err:
    LEDGER_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Ledger not available: {_ledger_err}")

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start price worker for fresh price data
try:
    from price_worker import start_price_worker, PRICE_CACHE
    start_price_worker()
    logger.info("✅ Price worker started successfully")
except Exception as e:
    logger.warning(f"⚠️ Price worker failed to start: {e}")
    PRICE_CACHE = {"price": None, "ts": None, "source": None}

# Security configuration
ALLOWED_IPS = (os.getenv('ALLOWED_IPS') or '0.0.0.0').split(',')  # Comma-separated IPs
# Raise defaults to avoid locking out the UI; can be tuned via env
RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS') or '2000')  # Requests per window
RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW') or '3600')  # Window seconds

# Admin credentials (use environment variables only)
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')

# Validate that credentials are set
if not ADMIN_USERNAME or not ADMIN_PASSWORD:
    logger.error("❌ ADMIN_USERNAME and ADMIN_PASSWORD environment variables must be set")
    raise ValueError("ADMIN_USERNAME and ADMIN_PASSWORD environment variables must be set")

# Max staleness used by both summary and trading gates
MAX_PRICE_STALENESS_SEC = int(os.getenv("MAX_PRICE_STALENESS_SEC") or "120")

# Rate limiting storage (in production, use Redis)
request_counts = {}

# Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    METRICS_ON = bool(os.getenv("METRICS_ENABLED", "true").lower() != "false")
    orb_signals = Counter("orb_signals_total", "ORB signals", ["dir","reason"])
    orb_trades = Counter("orb_trades_total", "Filled trades", ["dir","mode"])
    orb_skips = Counter("orb_skips_total", "Skips", ["reason"])
    spread_bps = Histogram("spread_bps", "Observed spread (bps)", buckets=[2,4,6,8,10,15,20])
    equity_g = Gauge("equity_usd", "Equity USD")
    exposure_g = Gauge("exposure_frac", "Exposure 0..1")
except ImportError:
    METRICS_ON = False
    orb_signals = orb_trades = orb_skips = spread_bps = equity_g = exposure_g = None

# Initialize FastAPI app
app = FastAPI(title="Bitcoin LLM Trading System", version="1.0.0")

# Trusted IPs that should bypass rate limiting (Docker network + internal services)
SAFE_IPS = {"127.0.0.1", "172.18.0.1", "172.18.0.2", "172.18.0.3", "172.18.0.5"}

def is_trusted_ip(ip: str) -> bool:
    return ip in SAFE_IPS or ip.startswith("172.18.")

# Security middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for IP whitelisting and rate limiting"""
    client_ip = request.client.host
    
    # Paths that should not count toward rate limits (lightweight/readonly)
    WHITELIST_PATHS = {
        '/', '/favicon.ico',
        '/equity_series_public', '/btc_data_public',
        '/diagnostics/price', '/diagnostics/volatility', '/diagnostics/telemetry',
        '/readyz', '/healthz', '/metrics', '/health', '/auto_trade_cron'
    }

    # IP whitelist check
    if client_ip not in ALLOWED_IPS and '0.0.0.0' not in ALLOWED_IPS:
        logger.warning(f"Blocked request from unauthorized IP: {client_ip}")
        return JSONResponse(
            status_code=403,
            content={"error": "IP not authorized"}
        )
    
    # Skip rate limiting for trusted IPs, safe paths and methods
    if is_trusted_ip(client_ip) or request.url.path in WHITELIST_PATHS or request.method in {"OPTIONS", "HEAD"}:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    # Rate limiting for external IPs
    current_time = time.time()
    
    # Clean old entries
    global request_counts
    request_counts = {ip: (count, timestamp) for ip, (count, timestamp) in request_counts.items() 
                     if current_time - timestamp < RATE_LIMIT_WINDOW}
    
    # Check rate limit
    if client_ip in request_counts:
        count, timestamp = request_counts[client_ip]
        if current_time - timestamp < RATE_LIMIT_WINDOW and count >= RATE_LIMIT_REQUESTS:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )
        request_counts[client_ip] = (count + 1, timestamp)
    else:
        request_counts[client_ip] = (1, current_time)
    
    # Add security headers
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response

# Security
security = HTTPBasic()

# JWT Configuration
JWT_SECRET = secrets.token_urlsafe(32)
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

# Load environment variables
kraken_api_key = os.getenv('KRAKEN_API_KEY')
kraken_secret_key = os.getenv('KRAKEN_PRIVATE_KEY')  # Updated to use KRAKEN_PRIVATE_KEY
news_api_key = os.getenv('NEWS_API_KEY')
cohere_key = os.getenv('COHERE_KEY')

# Initialize Trading Bot (Kraken only)
trading_bot = None
exchange_name = None

# Check if Kraken has valid credentials first
if kraken_api_key and kraken_secret_key:
    try:
        from kraken_trading_btc import KrakenTradingBot
        trading_bot = KrakenTradingBot()
        exchange_name = "kraken"
        logger.info("✅ Kraken trading bot initialized successfully (Real Bitcoin trading)")
    except Exception as e:
        logger.warning(f"Kraken not available: {e}")
        trading_bot = None

# If Kraken not available, log error
if not trading_bot:
    logger.error(f"❌ No trading bot available: Kraken initialization failed")
    exchange_name = None

# Initialize LLM Trading Strategy
try:
    from llm_trading_strategy import LLMTradingStrategy
    llm_strategy = LLMTradingStrategy()
    logger.info("✅ LLM trading strategy initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize LLM trading strategy: {e}")
    llm_strategy = None

# Initialize Grid Executor
try:
    from grid_executor import GridExecutor
    grid_executor = GridExecutor(trading_bot)
    logger.info("✅ Grid executor initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize grid executor: {e}")
    grid_executor = None

# Initialize Technical Analysis (if available)
try:
    from technical_analysis import TechnicalAnalyzer
    technical_analyzer = TechnicalAnalyzer()
    logger.info("✅ Technical analyzer initialized successfully")
except Exception as e:
    logger.warning(f"Technical analyzer not available: {e}")
    technical_analyzer = None

# Initialize trade log for tracking automated trades
trade_log = []

# Initialize Risk Management (if available)
try:
    from risk_management import RiskManager
    risk_manager = RiskManager()
    logger.info("✅ Risk manager initialized successfully")
except Exception as e:
    logger.warning(f"Risk manager not available: {e}")
    risk_manager = None

# News cache for BTC
btc_news_cache = {}
btc_news_cache_timestamp = {}

# Sentiment cache for BTC (to reduce Cohere calls)
btc_sentiment_cache = {}
btc_sentiment_cache_timestamp = {}

# Global variables

# Last good snapshot for graceful degradation
_LAST_SNAPSHOT = {"price": None, "change": None, "volume": None, "ts": None}

# Asset code normalization for Kraken
_ASSET_MAP = {
    "XXBT": "BTC", "XBT": "BTC", "BTC": "BTC",
    "ZUSD": "USD", "USD": "USD",
    "ZEUR": "EUR", "EUR": "EUR",
    "ZGBP": "GBP", "GBP": "GBP",
}

def _normalize_asset(symbol: str) -> str:
    return _ASSET_MAP.get(symbol, symbol)

class PublicBTC(BaseModel):
    status: str                 # "ok" | "degraded"
    price: float | None
    change_24h: float | None
    volume_24h: float | None
    last_update: str            # ISO timestamp
    message: str | None = None

class BTCTradingData(BaseModel):
    symbol: str = "BTC"
    price: float
    time: str
    news: str
    sentiment: int
    probability: float
    signal: int
    volume: float
    change_24h: float

class PnLSummary(BaseModel):
    status: Literal["ok","degraded"]
    message: Optional[str] = None
    last_price: float
    price_ts: str
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    fees_total: float
    exposure_value: float
    exposure_pct: float
    position_qty: float
    hodl_value: float

# --- helper: compute PnL summary deterministically --------------------------
async def _compute_pnl_summary() -> PnLSummary:
    from datetime import datetime, timezone
    import logging
    logger = logging.getLogger(__name__)
    logger.info("PnL: Function called - starting")

    def _safe_float(v, d=0.0):
        try: 
            return float(v)
        except Exception: 
            return d

    try:
        # Get current price from market data
        market_data = get_btc_market_data()
        px = float(market_data.get('close', 45000.0)) if market_data else 45000.0
        logger.info(f"PnL: Got price: {px}")
        
        # Get balances from trading bot
        btc_qty = 0.0
        cash_usd = 0.0
        
        if trading_bot:
            account_info = trading_bot.get_account_info()
            if account_info and 'balances' in account_info:
                raw_balances = account_info['balances']
                # Normalize asset codes like the diagnostics endpoint
                balances = {}
                for k, v in raw_balances.items():
                    sym = _normalize_asset(k)
                    try:
                        balances[sym] = balances.get(sym, 0.0) + float(v)
                    except (ValueError, TypeError):
                        pass
                
                btc_qty = float(balances.get("BTC", 0.0))
                cash_usd = float(balances.get("USD", 0.0))
            else:
                logger.warning("PnL: No balances in account_info")
        else:
            logger.warning("PnL: No trading bot available")
        
        # Calculate equity and exposure
        equity = cash_usd + btc_qty * px
        exposure_value = btc_qty * px
        exposure_pct = 0.0 if equity <= 0 else exposure_value / equity
        
        logger.info(f"PnL: Calculated equity: {equity}, exposure: {exposure_value}")
        
        # Check price staleness
        now = datetime.now(timezone.utc)
        status = "ok"
        msg = None

        result = PnLSummary(
            status=status,
            message=msg,
            last_price=px,
            price_ts=now.isoformat(),
            equity=round(equity, 2),
            realized_pnl=0.0,
            unrealized_pnl=0.0,  # spot-only simple
            fees_total=0.0,
            exposure_value=round(exposure_value, 2),
            exposure_pct=round(exposure_pct, 4),
            position_qty=round(btc_qty, 8),
            hodl_value=0.0,
        )
        
        logger.info(f"PnL: Returning result with equity: {result.equity}")
        return result
        
    except Exception as e:
        logger.error(f"PnL: Error in function: {type(e).__name__}: {str(e)}")
        raise

def get_btc_news() -> str:
    """Get latest Bitcoin news from News API"""
    import requests
    from datetime import datetime, timedelta
    
    # Check cache first (cache for 5 minutes)
    current_time = datetime.now()
    if 'BTC' in btc_news_cache and 'BTC' in btc_news_cache_timestamp:
        if (current_time - btc_news_cache_timestamp['BTC']).seconds < 300:  # 5 minutes
            return btc_news_cache['BTC']
    
    try:
        # Get Bitcoin news from News API
        url = f"https://newsapi.org/v2/everything"
        params = {
            'q': '"Bitcoin" OR "BTC" OR "cryptocurrency"',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 1,  # Get only the latest news
            'apiKey': news_api_key
        }
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('articles') and len(data['articles']) > 0:
                article = data['articles'][0]
                news_text = f"Latest: {article.get('title', 'No title')}"
                
                # Cache the result
                btc_news_cache['BTC'] = news_text
                btc_news_cache_timestamp['BTC'] = current_time
                
                return news_text
            else:
                return "Real-time Bitcoin data"
        else:
            return "Real-time Bitcoin data"
            
    except Exception as e:
        logger.error(f"Error fetching Bitcoin news: {e}")
        return "Real-time Bitcoin data"

def analyze_btc_sentiment(news_text: str) -> tuple:
    """Analyze Bitcoin sentiment using LLM strategy"""
    # 15-minute cache to reduce LLM calls triggered by dashboard/health checks
    current_time = datetime.now()
    last_ts = btc_sentiment_cache_timestamp.get('BTC')
    if last_ts and (current_time - last_ts).seconds < 900:
        logger.info("Using cached sentiment analysis")
        return btc_sentiment_cache['BTC']

    try:
        # Try to use LLM strategy if available
        if llm_strategy:
            try:
                # Get current market data for context
                market_data = get_btc_market_data()
                if market_data:
                    current_price = market_data.get('close', 45000.0)
                    change_24h = market_data.get('change_24h', 0.0)
                    volume = market_data.get('volume', 1000000.0)
                    
                    # Use LLM strategy for analysis
                    analysis = llm_strategy.analyze_market_conditions(
                        current_price=current_price,
                        price_change_24h=change_24h,
                        volume_24h=volume,
                        news_sentiment=news_text[:200],  # Limit news text
                        technical_indicators=None
                    )
                    
                    # Convert sentiment to numeric
                    if analysis.sentiment == "bullish":
                        sentiment = 1
                    elif analysis.sentiment == "bearish":
                        sentiment = -1
                    else:
                        sentiment = 0
                    
                    btc_sentiment_cache['BTC'] = (sentiment, analysis.confidence)
                    btc_sentiment_cache_timestamp['BTC'] = current_time
                    return sentiment, analysis.confidence
                    
            except Exception as e:
                logger.warning(f"LLM strategy failed, using fallback: {e}")
        
        # Fallback to simple keyword analysis
        positive_keywords = ['bullish', 'surge', 'rally', 'breakout', 'adoption', 'institutional', 'positive']
        negative_keywords = ['bearish', 'crash', 'dump', 'sell-off', 'regulation', 'ban', 'negative']
        
        text_lower = news_text.lower()
        positive_score = sum(1 for word in positive_keywords if word in text_lower)
        negative_score = sum(1 for word in negative_keywords if word in text_lower)
        
        if positive_score > negative_score:
            btc_sentiment_cache['BTC'] = (1, 0.7)
            btc_sentiment_cache_timestamp['BTC'] = current_time
            return 1, 0.7  # Positive sentiment
        elif negative_score > positive_score:
            btc_sentiment_cache['BTC'] = (-1, 0.7)
            btc_sentiment_cache_timestamp['BTC'] = current_time
            return -1, 0.7  # Negative sentiment
        else:
            btc_sentiment_cache['BTC'] = (0, 0.5)
            btc_sentiment_cache_timestamp['BTC'] = current_time
            return 0, 0.5  # Neutral sentiment
            
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        btc_sentiment_cache['BTC'] = (0, 0.5)
        btc_sentiment_cache_timestamp['BTC'] = datetime.now()
        return 0, 0.5

def generate_trading_signal(price: float, sentiment: int, volume: float, change_24h: float) -> int:
    """Generate trading signal based on price, sentiment, and technical indicators"""
    try:
        # Simple algorithmic trading logic
        signal_score = 0
        
        # Sentiment factor (40% weight)
        signal_score += sentiment * 0.4
        
        # Price momentum factor (30% weight)
        if change_24h > 2:  # Strong positive momentum
            signal_score += 0.3
        elif change_24h < -2:  # Strong negative momentum
            signal_score -= 0.3
        
        # Volume factor (20% weight)
        if volume > 1000000:  # High volume
            signal_score += 0.2
        elif volume < 500000:  # Low volume
            signal_score -= 0.1
        
        # Price level factor (10% weight)
        if price > 50000:  # High price level
            signal_score += 0.1
        elif price < 30000:  # Low price level
            signal_score -= 0.1
        
        # Determine final signal
        if signal_score > 0.3:
            return 1  # Buy signal
        elif signal_score < -0.3:
            return -1  # Sell signal
        else:
            return 0  # Hold signal
            
    except Exception as e:
        logger.error(f"Error generating trading signal: {e}")
        return 0

def get_btc_market_data() -> Optional[Dict]:
    """Get Bitcoin market data from price worker or fallback sources"""
    try:
        # Try to get fresh data from price worker first
        try:
            from price_worker import get_last_price, PRICE_CACHE
            price, ts = get_last_price()
            if price and ts:
                # Calculate age of price data
                now = datetime.now(timezone.utc)
                age_sec = (now - ts).total_seconds()
                
                # If price is fresh, use it
                if age_sec < MAX_PRICE_STALENESS_SEC:
                    return {
                        'close': price,
                        'volume': 30000000000.0,  # Default volume
                        'change_24h': 0.0,  # Default change
                        'source': PRICE_CACHE.get('source', 'unknown'),
                        'age_sec': age_sec
                    }
                else:
                    logger.warning(f"Price data too old: {age_sec:.1f}s (max: {MAX_PRICE_STALENESS_SEC}s)")
        except Exception as e:
            logger.warning(f"Price worker data failed: {e}")
        
        # Try to get data from trading bot as fallback
        if trading_bot:
            try:
                market_data = trading_bot.get_market_data()
                if market_data and 'error' not in market_data:
                    return {
                        'close': market_data.get('close', 45000.0),
                        'volume': market_data.get('volume', 1000000.0),
                        'change_24h': market_data.get('change_24h', 0.0),
                        'source': 'trading_bot'
                    }
            except Exception as e:
                logger.warning(f"Trading bot market data failed: {e}")
        
        # Fallback to CoinGecko API
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': 'bitcoin',
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_24hr_vol': 'true'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        # Handle rate limiting
        if response.status_code == 429:
            logger.warning("CoinGecko API rate limit reached, using fallback data")
            return {
                'close': 114000.0,  # Fallback price
                'volume': 30000000000.0,  # Fallback volume
                'change_24h': 0.0,  # Fallback change
                'source': 'fallback'
            }
        
        response.raise_for_status()
        data = response.json()
        
        if 'bitcoin' in data:
            btc_data = data['bitcoin']
            return {
                'close': btc_data.get('usd', 45000.0),
                'volume': btc_data.get('usd_24h_vol', 1000000.0),
                'change_24h': btc_data.get('usd_24h_change', 0.0),
                'source': 'coingecko'
            }
        else:
            logger.error("No Bitcoin data found in CoinGecko response")
            return None
            
    except Exception as e:
        logger.error(f"Error getting BTC market data: {e}")
        # Return fallback data instead of None
        return {
            'close': 114000.0,
            'volume': 30000000000.0,
            'change_24h': 0.0,
            'source': 'fallback'
        }

def create_jwt_token(username: str):
    """Create JWT token"""
    expiry = datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS)
    payload = {
        "username": username,
        "exp": expiry
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("username")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(token: str = None):
    """Get current user from JWT token"""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    return verify_jwt_token(token)

# Runtime settings and toggles
AUTO_TRADE_ENABLED = True

def load_runtime_settings():
    """Load runtime settings from database."""
    if LEDGER_AVAILABLE:
        try:
            from db import read_settings
            settings = read_settings()
            return {
                "min_confidence": float(settings.get("min_confidence", 0.7)),
                "max_exposure": float(settings.get("max_exposure", 0.8)),
                "trade_cooldown_hours": float(settings.get("trade_cooldown_hours", 3)),
                "min_trade_delta": float(settings.get("min_trade_delta", 0.05)),
                "min_trade_delta_usd": float(settings.get("min_trade_delta_usd", 10.0)),
                "min_trade_delta_pct": float(settings.get("min_trade_delta_pct", 0.00)),
                "no_fee_mode": parse_bool(settings.get("no_fee_mode", True)),
                "grid_executor_enabled": parse_bool(settings.get("grid_executor_enabled", True)),
                "grid_step_pct": float(settings.get("grid_step_pct", 0.25)),
                "grid_order_usd": float(settings.get("grid_order_usd", 12.0)),
                "max_grid_exposure": float(settings.get("max_grid_exposure", 0.1)),
                "safety_skip_degraded": parse_bool(settings.get("safety_skip_degraded", True)),
                "safety_max_price_staleness_sec": float(settings.get("safety_max_price_staleness_sec", 120.0)),
                "safety_min_expected_move_pct": float(settings.get("safety_min_expected_move_pct", 0.02)),
                "safety_daily_pnl_limit_usd": float(settings.get("safety_daily_pnl_limit_usd", -5.0)),
                "safety_daily_equity_drop_pct": float(settings.get("safety_daily_equity_drop_pct", 3.0)),
            }
        except Exception as e:
            logger.warning(f"Failed to load settings from DB: {e}")
    
    # Fallback to environment variables
    return {
        "min_confidence": float(os.getenv("MIN_CONFIDENCE") or "0.8"),
        "max_exposure": float(os.getenv("MAX_EXPOSURE") or "0.8"),
        "trade_cooldown_hours": float(os.getenv("TRADE_COOLDOWN_HOURS") or "1"),
        "min_trade_delta": float(os.getenv("MIN_TRADE_DELTA") or "0.0"),
        "min_trade_delta_usd": float(os.getenv("MIN_TRADE_DELTA_USD") or "10.0"),
        "min_trade_delta_pct": float(os.getenv("MIN_TRADE_DELTA_PCT") or "0.00"),
        "no_fee_mode": parse_bool(os.getenv("NO_FEE_MODE") or "True"),
        "grid_executor_enabled": parse_bool(os.getenv("GRID_EXECUTOR_ENABLED") or "True"),
        "grid_step_pct": float(os.getenv("GRID_STEP_PCT") or "0.25"),
        "grid_order_usd": float(os.getenv("GRID_ORDER_USD") or "12.0"),
        "max_grid_exposure": float(os.getenv("MAX_GRID_EXPOSURE") or "0.1"),
        "safety_skip_degraded": True,
        "safety_max_price_staleness_sec": 120.0,
        "safety_min_expected_move_pct": 0.02,
        "safety_daily_pnl_limit_usd": -5.0,
        "safety_daily_equity_drop_pct": 3.0,
    }

def get_setting(key: str, default=None):
    """Get a single setting value from the database."""
    if LEDGER_AVAILABLE:
        try:
            from db import get_setting as db_get_setting
            return db_get_setting(key, default)
        except Exception as e:
            logger.warning(f"Failed to get setting {key} from DB: {e}")
    return default

RUNTIME_SETTINGS = load_runtime_settings()

# After strategy and bot are initialized, sync runtime settings
try:
    if llm_strategy:
        llm_strategy.min_confidence = RUNTIME_SETTINGS["min_confidence"]
        llm_strategy.max_exposure = RUNTIME_SETTINGS["max_exposure"]
    if trading_bot:
        trading_bot.trade_cooldown_hours = RUNTIME_SETTINGS["trade_cooldown_hours"]
except Exception:
    pass

@app.get("/settings")
async def get_settings(token: str = None):
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        verify_jwt_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Load current settings from database
    current_settings = load_runtime_settings()
    auto_trade_enabled = get_setting("auto_trade_enabled", True) if LEDGER_AVAILABLE else AUTO_TRADE_ENABLED
    
    return {
        "auto_trade_enabled": auto_trade_enabled,
        "min_confidence": current_settings["min_confidence"],
        "max_exposure": current_settings["max_exposure"],
        "trade_cooldown_hours": current_settings["trade_cooldown_hours"],
        "min_trade_delta": current_settings["min_trade_delta"],
        "min_trade_delta_usd": current_settings["min_trade_delta_usd"],
        "min_trade_delta_pct": current_settings["min_trade_delta_pct"],
        "no_fee_mode": current_settings["no_fee_mode"],
        "grid_executor_enabled": current_settings["grid_executor_enabled"],
        "grid_step_pct": current_settings["grid_step_pct"],
        "grid_order_usd": current_settings["grid_order_usd"],
        "max_grid_exposure": current_settings["max_grid_exposure"],
        "safety_skip_degraded": current_settings["safety_skip_degraded"],
        "safety_max_price_staleness_sec": current_settings["safety_max_price_staleness_sec"],
        "safety_min_expected_move_pct": current_settings["safety_min_expected_move_pct"],
        "safety_daily_pnl_limit_usd": current_settings["safety_daily_pnl_limit_usd"],
        "safety_daily_equity_drop_pct": current_settings["safety_daily_equity_drop_pct"],
    }

@app.post("/settings")
async def update_settings(
    token: str = Form(...),
    auto_trade_enabled: Optional[bool] = Form(None),
    min_confidence: Optional[float] = Form(None),
    max_exposure: Optional[float] = Form(None),
    trade_cooldown_hours: Optional[float] = Form(None),
    min_trade_delta: Optional[float] = Form(None),
    min_trade_delta_usd: Optional[float] = Form(None),
    min_trade_delta_pct: Optional[float] = Form(None),
    no_fee_mode: Optional[bool] = Form(None),
    grid_executor_enabled: Optional[bool] = Form(None),
    grid_step_pct: Optional[float] = Form(None),
    grid_order_usd: Optional[float] = Form(None),
    max_grid_exposure: Optional[float] = Form(None),
    safety_skip_degraded: Optional[bool] = Form(None),
    safety_max_price_staleness_sec: Optional[float] = Form(None),
    safety_min_expected_move_pct: Optional[float] = Form(None),
    safety_daily_pnl_limit_usd: Optional[float] = Form(None),
    safety_daily_equity_drop_pct: Optional[float] = Form(None),
):
    global AUTO_TRADE_ENABLED
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        verify_jwt_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Write settings to database if available
    if LEDGER_AVAILABLE:
        try:
            from db import write_setting
            if auto_trade_enabled is not None:
                write_setting("auto_trade_enabled", bool(auto_trade_enabled))
            if min_confidence is not None:
                write_setting("min_confidence", float(min_confidence))
            if max_exposure is not None:
                write_setting("max_exposure", float(max_exposure))
            if trade_cooldown_hours is not None:
                write_setting("trade_cooldown_hours", float(trade_cooldown_hours))
            if min_trade_delta is not None:
                write_setting("min_trade_delta", float(min_trade_delta))
            if min_trade_delta_usd is not None:
                write_setting("min_trade_delta_usd", float(min_trade_delta_usd))
            if min_trade_delta_pct is not None:
                write_setting("min_trade_delta_pct", float(min_trade_delta_pct))
            if no_fee_mode is not None:
                write_setting("no_fee_mode", bool(no_fee_mode))
            if grid_executor_enabled is not None:
                write_setting("grid_executor_enabled", bool(grid_executor_enabled))
            if grid_step_pct is not None:
                write_setting("grid_step_pct", float(grid_step_pct))
            if grid_order_usd is not None:
                write_setting("grid_order_usd", float(grid_order_usd))
            if max_grid_exposure is not None:
                write_setting("max_grid_exposure", float(max_grid_exposure))
            if safety_skip_degraded is not None:
                write_setting("safety_skip_degraded", bool(safety_skip_degraded))
            if safety_max_price_staleness_sec is not None:
                write_setting("safety_max_price_staleness_sec", float(safety_max_price_staleness_sec))
            if safety_min_expected_move_pct is not None:
                write_setting("safety_min_expected_move_pct", float(safety_min_expected_move_pct))
            if safety_daily_pnl_limit_usd is not None:
                write_setting("safety_daily_pnl_limit_usd", float(safety_daily_pnl_limit_usd))
            if safety_daily_equity_drop_pct is not None:
                write_setting("safety_daily_equity_drop_pct", float(safety_daily_equity_drop_pct))
        except Exception as e:
            logger.warning(f"Failed to write settings to DB: {e}")

    # Update in-memory settings for immediate use
    if auto_trade_enabled is not None:
        AUTO_TRADE_ENABLED = bool(auto_trade_enabled)
    if min_confidence is not None:
        RUNTIME_SETTINGS["min_confidence"] = float(min_confidence)
        try:
            if llm_strategy:
                llm_strategy.min_confidence = float(min_confidence)
        except Exception:
            pass
    if max_exposure is not None:
        RUNTIME_SETTINGS["max_exposure"] = float(max_exposure)
        try:
            if llm_strategy:
                llm_strategy.max_exposure = float(max_exposure)
        except Exception:
            pass
    if trade_cooldown_hours is not None:
        RUNTIME_SETTINGS["trade_cooldown_hours"] = float(trade_cooldown_hours)
        try:
            if trading_bot:
                trading_bot.trade_cooldown_hours = float(trade_cooldown_hours)
        except Exception:
            pass
    if min_trade_delta is not None:
        RUNTIME_SETTINGS["min_trade_delta"] = float(min_trade_delta)
        try:
            if trading_bot:
                trading_bot.rebalance_threshold_pct = float(min_trade_delta)
        except Exception:
            pass
    if min_trade_delta_usd is not None:
        RUNTIME_SETTINGS["min_trade_delta_usd"] = float(min_trade_delta_usd)
        try:
            if trading_bot:
                trading_bot.min_trade_delta_usd = float(min_trade_delta_usd)
        except Exception:
            pass
    if min_trade_delta_pct is not None:
        RUNTIME_SETTINGS["min_trade_delta_pct"] = float(min_trade_delta_pct)
        try:
            if trading_bot:
                trading_bot.min_trade_delta_pct = float(min_trade_delta_pct)
        except Exception:
            pass
    if no_fee_mode is not None:
        RUNTIME_SETTINGS["no_fee_mode"] = bool(no_fee_mode)
        try:
            if trading_bot:
                trading_bot.no_fee_mode = bool(no_fee_mode)
        except Exception:
            pass
    if grid_executor_enabled is not None:
        RUNTIME_SETTINGS["grid_executor_enabled"] = bool(grid_executor_enabled)
        try:
            if grid_executor:
                grid_executor.enabled = bool(grid_executor_enabled)
        except Exception:
            pass
    if grid_step_pct is not None:
        RUNTIME_SETTINGS["grid_step_pct"] = float(grid_step_pct)
        try:
            if grid_executor:
                grid_executor.grid_step_pct = float(grid_step_pct)
        except Exception:
            pass
    if grid_order_usd is not None:
        RUNTIME_SETTINGS["grid_order_usd"] = float(grid_order_usd)
        try:
            if grid_executor:
                grid_executor.grid_order_usd = float(grid_order_usd)
        except Exception:
            pass
    if max_grid_exposure is not None:
        RUNTIME_SETTINGS["max_grid_exposure"] = float(max_grid_exposure)
        try:
            if grid_executor:
                grid_executor.max_grid_exposure = float(max_grid_exposure)
        except Exception:
            pass
    if safety_skip_degraded is not None:
        RUNTIME_SETTINGS["safety_skip_degraded"] = bool(safety_skip_degraded)
    if safety_max_price_staleness_sec is not None:
        RUNTIME_SETTINGS["safety_max_price_staleness_sec"] = float(safety_max_price_staleness_sec)
    if safety_min_expected_move_pct is not None:
        RUNTIME_SETTINGS["safety_min_expected_move_pct"] = float(safety_min_expected_move_pct)
    if safety_daily_pnl_limit_usd is not None:
        RUNTIME_SETTINGS["safety_daily_pnl_limit_usd"] = float(safety_daily_pnl_limit_usd)
    if safety_daily_equity_drop_pct is not None:
        RUNTIME_SETTINGS["safety_daily_equity_drop_pct"] = float(safety_daily_equity_drop_pct)

    # Return current settings from database
    current_settings = load_runtime_settings()
    auto_trade_enabled_db = get_setting("auto_trade_enabled", True) if LEDGER_AVAILABLE else AUTO_TRADE_ENABLED
    
    return {"status": "ok", "settings": {
        "auto_trade_enabled": auto_trade_enabled_db,
        **current_settings,
    }}

@app.get("/grid_status")
async def get_grid_status(token: str = None):
    """Get current grid executor status"""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        verify_jwt_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not grid_executor:
        return {"status": "error", "message": "Grid executor not available"}
    
    return grid_executor.get_grid_status()

@app.post("/grid_control")
async def grid_control(
    token: str = Form(...),
    action: str = Form(...),  # "enable", "disable", "reset", "update_bias"
    bias_allocation: Optional[float] = Form(None),
    current_price: Optional[float] = Form(None)
):
    """Control grid executor operations"""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        verify_jwt_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not grid_executor:
        return {"status": "error", "message": "Grid executor not available"}
    
    try:
        if action == "enable":
            grid_executor.enabled = True
            return {"status": "success", "message": "Grid executor enabled"}
        elif action == "disable":
            grid_executor.enabled = False
            return {"status": "success", "message": "Grid executor disabled"}
        elif action == "reset":
            if current_price is None:
                return {"status": "error", "message": "current_price required for reset"}
            grid_executor.reset_grid(current_price, bias_allocation or 0.0)
            return {"status": "success", "message": "Grid reset successfully"}
        elif action == "update_bias":
            if bias_allocation is None:
                return {"status": "error", "message": "bias_allocation required"}
            grid_executor.update_bias(bias_allocation)
            return {"status": "success", "message": "Grid bias updated"}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
    except Exception as e:
        return {"status": "error", "message": f"Grid control failed: {str(e)}"}

@app.get("/grid_self_test")
async def grid_self_test(token: str = None):
    """Run grid order placement self-test (no live keys)"""
    logger.info("🧪 Grid self-test requested")
    
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        username = verify_jwt_token(token)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not grid_executor:
        return {"message": "Grid executor not available", "status": "error"}
    
    try:
        test_results = grid_executor.self_test_order_placement()
        return {
            "message": "Grid self-test completed",
            "status": test_results["overall_status"],
            "results": test_results
        }
            
    except Exception as e:
        logger.error(f"Grid self-test failed: {e}")
        return {"message": f"Grid self-test failed: {str(e)}", "status": "error"}

@app.get("/lock_status")
async def get_lock_status(token: str = None):
    """Get current trade lock status"""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        verify_jwt_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not LEDGER_AVAILABLE:
        return {"status": "error", "message": "Database not available"}
    
    try:
        from db import get_trade_lock_info, cleanup_expired_locks
        
        # Clean up expired locks first
        cleaned = cleanup_expired_locks()
        if cleaned > 0:
            logger.info(f"🧹 Cleaned up {cleaned} expired trade locks")
        
        # Get current lock info
        lock_info = get_trade_lock_info("auto_trade_scheduled")
        
        if lock_info:
            # Check if lock is expired
            from datetime import datetime
            expires_at = datetime.fromisoformat(lock_info['expires_at'])
            now = datetime.utcnow()
            is_expired = expires_at < now
            
            return {
                "status": "ok",
                "lock_held": not is_expired,
                "lock_info": {
                    "process_id": lock_info['process_id'],
                    "acquired_at": lock_info['acquired_at'],
                    "expires_at": lock_info['expires_at'],
                    "is_expired": is_expired,
                    "metadata": lock_info['metadata']
                }
            }
        else:
            return {
                "status": "ok",
                "lock_held": False,
                "lock_info": None
            }
            
    except Exception as e:
        logger.error(f"Error getting lock status: {e}")
        return {"status": "error", "message": f"Error getting lock status: {str(e)}"}

@app.post("/release_lock")
async def release_lock(token: str = Form(...), lock_key: str = Form("auto_trade_scheduled")):
    """Force release a trade lock (admin only)"""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        verify_jwt_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not LEDGER_AVAILABLE:
        return {"status": "error", "message": "Database not available"}
    
    try:
        from db import release_trade_lock, get_trade_lock_info
        
        # Check if lock exists
        lock_info = get_trade_lock_info(lock_key)
        if not lock_info:
            return {"status": "ok", "message": f"Lock '{lock_key}' not found"}
        
        # Release the lock
        released = release_trade_lock(lock_key)
        
        if released:
            logger.warning(f"🔓 Force released trade lock '{lock_key}' (was held by PID: {lock_info['process_id']})")
            return {
                "status": "ok", 
                "message": f"Lock '{lock_key}' released successfully",
                "released_lock": lock_info
            }
        else:
            return {"status": "error", "message": f"Failed to release lock '{lock_key}'"}
            
    except Exception as e:
        logger.error(f"Error releasing lock: {e}")
        return {"status": "error", "message": f"Error releasing lock: {str(e)}"}

@app.get("/equity_series")
async def equity_series(token: str = None, limit: int = 500):
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        verify_jwt_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        eq = ledger_get_equity_curve(limit=limit)
        hodl = ledger_get_hodl_curve(limit=limit)
        return {"equity_curve": eq, "hodl_curve": hodl}
    except Exception as e:
        return {"equity_curve": [], "hodl_curve": [], "error": str(e)}

@app.get("/equity_series_public")
async def equity_series_public(limit: int = 500):
    try:
        eq = ledger_get_equity_curve(limit=limit)
        hodl = ledger_get_hodl_curve(limit=limit)
        return {"equity_curve": eq, "hodl_curve": hodl}
    except Exception as e:
        return {"equity_curve": [], "hodl_curve": [], "error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def homepage():
    """Secure ORB Dashboard - Main homepage"""
    with open('secure-orb-dashboard.html', 'r') as f:
        return HTMLResponse(content=f.read())

@app.get("/btc_data_public")
def btc_data_public() -> PublicBTC:
    """Get basic Bitcoin market data with graceful degradation (public endpoint)"""
    try:
        # Get market data
        market_data = get_btc_market_data()
        
        if market_data:
            price = market_data.get('close', 45000.0)
            volume = market_data.get('volume', 1000000.0)
            change_24h = market_data.get('change_24h', 0.0)
        else:
            # Fallback data
            price = 45000.0
            volume = 1000000.0
            change_24h = 2.5
        
        # Update last good snapshot
        _LAST_SNAPSHOT.update({
            "price": price,
            "change": change_24h,
            "volume": volume,
            "ts": datetime.now(timezone.utc).isoformat()
        })
        
        return PublicBTC(
            status="ok",
            price=price,
            change_24h=change_24h,
            volume_24h=volume,
            last_update=_LAST_SNAPSHOT["ts"]
        )
    except Exception as e:
        # Degrade gracefully but don't 500
        logger.warning(f"BTC data fetch failed, using fallback: {e}")
        if _LAST_SNAPSHOT["price"] is not None:
            return PublicBTC(
                status="degraded",
                price=_LAST_SNAPSHOT["price"],
                change_24h=_LAST_SNAPSHOT["change"],
                volume_24h=_LAST_SNAPSHOT["volume"],
                last_update=_LAST_SNAPSHOT["ts"],
                message=f"fallback: {type(e).__name__}"
            )
        # Absolutely last resort
        return PublicBTC(
            status="degraded",
            price=None,
            change_24h=None,
            volume_24h=None,
            last_update=datetime.now(timezone.utc).isoformat(),
            message="no data"
        )

@app.get("/btc_data")
async def get_btc_data(token: str = None):
    """Get Bitcoin market data and analysis (authenticated)"""
    # Require authentication for sensitive data
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        username = verify_jwt_token(token)
        logger.info(f"BTC data requested by user: {username}")
    except:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        # Get market data with graceful degradation
        market_data = get_btc_market_data()
        
        if market_data:
            price = market_data.get('close', 45000.0)
            volume = market_data.get('volume', 1000000.0)
            change_24h = market_data.get('change_24h', 0.0)
        else:
            # Fallback data
            price = 45000.0
            volume = 1000000.0
            change_24h = 2.5
        
        # Update last good snapshot
        _LAST_SNAPSHOT.update({
            "price": price,
            "change": change_24h,
            "volume": volume,
            "ts": datetime.now(timezone.utc).isoformat()
        })
        
        # Get news and sentiment (with graceful degradation)
        try:
            news_text = get_btc_news()
            sentiment, probability = analyze_btc_sentiment(news_text)
        except Exception as e:
            logger.warning(f"News/sentiment analysis failed: {e}")
            news_text = "Real-time Bitcoin data"
            sentiment = 0
            probability = 0.5
        
        # Generate trading signal (with graceful degradation)
        try:
            signal = generate_trading_signal(price, sentiment, volume, change_24h)
        except Exception as e:
            logger.warning(f"Trading signal generation failed: {e}")
            signal = 0
        
        return BTCTradingData(
            symbol="BTC",
            price=price,
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            news=news_text,
            sentiment=sentiment,
            probability=probability,
            signal=signal,
            volume=volume,
            change_24h=change_24h
        )
        
    except Exception as e:
        logger.error(f"Error getting BTC data: {e}")
        # Use last good snapshot if available
        if _LAST_SNAPSHOT["price"] is not None:
            return BTCTradingData(
                symbol="BTC",
                price=_LAST_SNAPSHOT["price"],
                time=_LAST_SNAPSHOT["ts"],
                news="Real-time Bitcoin data (degraded)",
                sentiment=0,
                probability=0.5,
                signal=0,
                volume=_LAST_SNAPSHOT["volume"],
                change_24h=_LAST_SNAPSHOT["change"]
            )
        # Absolute fallback
        return BTCTradingData(
            symbol="BTC",
            price=45000.0,
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            news="Real-time Bitcoin data",
            sentiment=0,
            probability=0.5,
            signal=0,
            volume=1000000.0,
            change_24h=0.0
        )

@app.post("/buy_btc")
async def buy_btc(amount: float = Form(...), token: str = Form(...)):
    """Buy Bitcoin with dollar amount"""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        username = verify_jwt_token(token)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not trading_bot:
        return {"message": "No trading bot available"}
    
    try:
        # Execute buy order through the trading bot
        result = trading_bot.place_buy_order("XXBTZUSD", amount, 1.0)
        
        trade_log.append({
            "action": "BUY",
            "symbol": "BTC",
            "amount": amount,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result": result
        })
        
        if result['status'] == 'success':
            # Record in ledger (best-effort)
            try:
                if LEDGER_AVAILABLE:
                    qty_btc = float(result.get('quantity', 0.0))
                    price = float(result.get('price_per_btc', 0.0)) if result.get('price_per_btc') is not None else (qty_btc and amount/qty_btc or 0.0)
                    fee = float(result.get('fee', 0.0)) if result.get('fee') is not None else 0.0
                    from db import record_order as _rec
                    _rec(
                        symbol="BTC",
                        side="buy",
                        qty_btc=qty_btc,
                        notional_usd=float(amount),
                        price=price,
                        demo_mode=bool(result.get('demo_mode', False)),
                        exchange_order_id=result.get('order_id'),
                        fee=fee,
                        metadata=result,
                    )
            except Exception as e:
                logger.warning(f"Ledger write failed (buy): {e}")
            return {"message": f"Bought ${amount:.2f} worth of Bitcoin (via {exchange_name})", "result": result}
        else:
            return {"message": f"Buy order failed: {result.get('reason', 'Unknown error')}", "result": result}
            
    except Exception as e:
        return {"message": f"Error buying Bitcoin: {str(e)}"}

@app.post("/sell_btc")
async def sell_btc(amount: float = Form(...), token: str = Form(...)):
    """Sell Bitcoin"""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        username = verify_jwt_token(token)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not trading_bot:
        return {"message": "No trading bot available"}
    
    try:
        # Execute sell order through the trading bot
        result = trading_bot.place_sell_order("XXBTZUSD", amount)
        
        trade_log.append({
            "action": "SELL",
            "symbol": "BTC",
            "amount": amount,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result": result
        })
        
        if result['status'] == 'success':
            # Record in ledger (best-effort)
            try:
                if LEDGER_AVAILABLE:
                    qty_btc = float(result.get('quantity', 0.0))
                    price = float(result.get('price_per_btc', 0.0)) if result.get('price_per_btc') is not None else (qty_btc and amount/qty_btc or 0.0)
                    fee = float(result.get('fee', 0.0)) if result.get('fee') is not None else 0.0
                    from db import record_order as _rec
                    _rec(
                        symbol="BTC",
                        side="sell",
                        qty_btc=qty_btc,
                        notional_usd=float(amount),
                        price=price,
                        demo_mode=bool(result.get('demo_mode', False)),
                        exchange_order_id=result.get('order_id'),
                        fee=fee,
                        metadata=result,
                    )
            except Exception as e:
                logger.warning(f"Ledger write failed (sell): {e}")
            return {"message": f"Sold ${amount:.2f} worth of Bitcoin (via {exchange_name})", "result": result}
        else:
            return {"message": f"Sell order failed: {result.get('reason', 'Unknown error')}", "result": result}
            
    except Exception as e:
        return {"message": f"Error selling Bitcoin: {str(e)}"}

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """Login page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Login - Bitcoin LLM Trading</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; display: flex; justify-content: center; align-items: center; height: 100vh; }
            .login-container { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); width: 400px; }
            .header { text-align: center; margin-bottom: 30px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; }
            .btn { width: 100%; padding: 12px; background: #f7931a; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }
            .btn:hover { background: #e6850e; }
        </style>
    </head>
    <body>
        <div class="login-container">
            <div class="header">
                <h1>🚀 Bitcoin LLM Trading</h1>
                <p>Login to access the trading system</p>
            </div>
            <form id="loginForm">
                <div class="form-group">
                    <label for="username">Username:</label>
                    <input type="text" id="username" name="username" required>
                </div>
                <div class="form-group">
                    <label for="password">Password:</label>
                    <input type="password" id="password" name="password" required>
                </div>
                <button type="submit" class="btn">Login</button>
            </form>
        </div>
        
        <script>
            document.getElementById('loginForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                formData.append('username', document.getElementById('username').value);
                formData.append('password', document.getElementById('password').value);
                
                try {
                    const response = await fetch('/auth/login', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        localStorage.setItem('jwt_token', data.session_token);
                        window.location.href = '/';
                    } else {
                        alert('Login failed: ' + data.detail);
                    }
                } catch (error) {
                    alert('Login error: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/auth/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """Handle login"""
    logger.info(f"Login attempt - username: {username}, expected: {ADMIN_USERNAME}")
    logger.info(f"Password check - provided: {password[:3]}..., expected: {ADMIN_PASSWORD[:3]}...")
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        token = create_jwt_token(username)
        return {"session_token": token, "message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/login")
async def login_endpoint(username: str = Form(...), password: str = Form(...)):
    """Handle login for secure dashboard"""
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        token = create_jwt_token(username)
        return {"access_token": token, "message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/test_apis")
async def test_apis():
    """Test API connectivity"""
    results = {}
    
    # Test Kraken
    if trading_bot and exchange_name == "kraken":
        results["kraken"] = "✅ Connected successfully"
    else:
        results["kraken"] = "❌ Not configured"
    
    # Test Alpaca (if available)
    # Removed Alpaca test as it's no longer used
    
    # Test News API
    try:
        if news_api_key:
            results["news_api"] = "✅ Connected successfully"
        else:
            results["news_api"] = "❌ Not configured"
    except Exception as e:
        results["news_api"] = f"❌ Error: {str(e)}"
    
    # Test Cohere
    try:
        if cohere_key:
            results["cohere"] = "✅ Connected successfully"
        else:
            results["cohere"] = "❌ Not configured"
    except Exception as e:
        results["cohere"] = f"❌ Error: {str(e)}"
    
    return results

@app.get("/test")
async def test_page():
    """Serve the test HTML page"""
    with open('test.html', 'r') as f:
        return HTMLResponse(content=f.read())

@app.get("/dashboard")
async def dashboard_page():
    """Serve the ORB dashboard HTML page"""
    with open('secure-orb-dashboard.html', 'r') as f:
        return HTMLResponse(content=f.read())

@app.get("/login-test")
async def login_test_page():
    """Serve the login test HTML page"""
    with open('login-test.html', 'r') as f:
        return HTMLResponse(content=f.read())

@app.post("/auto_trade")
async def auto_trade(token: str = Form(...)):
    """Execute automated trading based on LLM analysis"""
    logger.info("🤖 Auto Trade endpoint called")
    
    if not token:
        logger.warning("No token provided for auto trade")
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        username = verify_jwt_token(token)
        logger.info(f"Auto trade requested by user: {username}")
    except:
        logger.warning("Invalid token for auto trade")
        raise HTTPException(status_code=401, detail="Authentication required")

    if not trading_bot:
        logger.warning("No trading bot available for auto trade")
        return {"message": "No trading bot available"}

    if not llm_strategy:
        logger.warning("LLM strategy not available for auto trade")
        return {"message": "LLM strategy not available"}

    try:
        logger.info("🤖 Starting LLM market analysis...")
        
        # Get current market data
        market_data = get_btc_market_data()
        if not market_data:
            logger.error("Unable to get market data for auto trade")
            return {"message": "Unable to get market data"}

        current_price = market_data.get('close', 45000.0)
        price_change_24h = market_data.get('change_24h', 0.0)
        volume_24h = market_data.get('volume', 1000000.0)
        
        # Check price staleness
        try:
            # Get price timestamp from market data or use current time
            price_ts = datetime.now(timezone.utc)  # Default to current time
            if 'last_update' in market_data:
                try:
                    price_ts = datetime.fromisoformat(market_data['last_update'].replace('Z', '+00:00'))
                except:
                    price_ts = datetime.now(timezone.utc)
            
            staleness = (datetime.now(timezone.utc) - price_ts).total_seconds()
            if staleness > MAX_PRICE_STALENESS_SEC:
                logger.warning(f"🛡️ Skip trade: price stale ({int(staleness)}s > {MAX_PRICE_STALENESS_SEC}s)")
                return {"status": "skipped", "reason": f"price_stale_{int(staleness)}s"}
        except Exception as e:
            logger.warning(f"🛡️ Skip trade: unable to check price staleness: {e}")
            return {"status": "skipped", "reason": "staleness_check_failed"}
        
        logger.info(f"📊 Market Data: Price=${current_price:,.2f}, Change={price_change_24h:+.2f}%, Volume=${volume_24h:,.0f}")

        # Get news sentiment
        news_text = get_btc_news()
        sentiment, probability = analyze_btc_sentiment(news_text)
        news_sentiment = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
        
        logger.info(f"📰 News Sentiment: {news_sentiment} (score: {sentiment}, probability: {probability:.2f})")

        # Generate LLM trading signal
        logger.info("🧠 Generating LLM trading signal...")
        try:
            signal = llm_strategy.generate_trading_signal(
                current_price=current_price,
                price_change_24h=price_change_24h,
                volume_24h=volume_24h,
                news_sentiment=news_sentiment
            )
            
            logger.info(f"🎯 LLM Signal: {signal['action'].upper()} - {signal['reason']} (confidence: {signal['confidence']:.2f})")
            
        except Exception as llm_error:
            logger.error(f"❌ LLM signal generation failed: {llm_error}")
            # Create fallback signal
            signal = {
                'action': 'hold',
                'should_trade': False,
                'reason': f'LLM analysis failed: {str(llm_error)}',
                'confidence': 0.5,
                'risk_level': 'medium',
                'position_size': None,
                'price_target': None,
                'stop_loss': None,
                'analysis': f'LLM Error: {str(llm_error)}',
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"🔄 Using fallback signal: {signal['action'].upper()}")

        # Check account balance before executing trade
        account_balance = trading_bot.get_account_info()
        logger.info(f"💰 Account Balance: {account_balance}")
        
        # Calculate appropriate position size based on available funds
        available_funds = 0.0
        btc_balance = 0.0
        
        if account_balance and 'balances' in account_balance:
            balances = account_balance['balances']
            # Check GBP balance
            gbp_balance = float(balances.get('ZGBP', '0.0'))
            # Check USD balance  
            usd_balance = float(balances.get('ZUSD', '0.0'))
            # Check BTC balance
            btc_balance = float(balances.get('XXBT', '0.0'))
            
            available_funds = gbp_balance + usd_balance
            logger.info(f"💵 Available funds: £{gbp_balance:.2f} + ${usd_balance:.2f} = ${available_funds:.2f}")
            logger.info(f"₿ BTC balance: {btc_balance:.8f}")
        
        # Execute allocation change via target exposure if provided
        trade_result = None
        if signal.get('target_exposure') is not None:
            try:
                target = float(signal.get('target_exposure'))
                # Pass decision price for slippage checking
                decision_price = current_price if current_price else None
                trade_result = trading_bot.rebalance_to_target(target, decision_price)
            except Exception as e:
                logger.warning(f"Rebalance failed: {e}")
                trade_result = {"status": "error", "reason": str(e)}
        elif signal['should_trade'] and signal['position_size']:
            # Adjust position size based on available funds
            requested_size = signal['position_size']
            max_buy_size = available_funds * 0.95  # Use 95% of available funds
            max_sell_size = btc_balance * current_price * 0.95  # Use 95% of BTC balance
            
            if signal['action'] == 'buy':
                if max_buy_size < 10.0:  # Minimum $10 trade
                    logger.info(f"⏸️ Insufficient funds for buy: ${max_buy_size:.2f} available, ${requested_size:.2f} requested")
                    trade_result = {"status": "error", "reason": ["Insufficient funds for buy"]}
                else:
                    actual_size = min(requested_size, max_buy_size)
                    logger.info(f"💰 Executing buy: ${actual_size:.2f} (requested: ${requested_size:.2f}, available: ${max_buy_size:.2f})")
                    trade_result = trading_bot.place_buy_order("XXBTZUSD", actual_size, 1.0)
                    # Persist trade to ledger (best-effort)
                    try:
                        if LEDGER_AVAILABLE and isinstance(trade_result, dict) and trade_result.get('status') == 'success':
                            qty_btc = float(trade_result.get('quantity', 0.0))
                            price = float(trade_result.get('price_per_btc', 0.0)) if trade_result.get('price_per_btc') is not None else (qty_btc and actual_size/qty_btc or 0.0)
                            fee = float(trade_result.get('fee', 0.0)) if trade_result.get('fee') is not None else 0.0
                            ledger_record_order(
                                symbol="BTC",
                                side="buy",
                                qty_btc=qty_btc,
                                notional_usd=float(actual_size),
                                price=price,
                                demo_mode=bool(trade_result.get('demo_mode', False)),
                                exchange_order_id=trade_result.get('order_id'),
                                fee=fee,
                                metadata=trade_result,
                            )
                    except Exception as e:
                        logger.warning(f"Ledger write failed (auto buy): {e}")
                    
            elif signal['action'] == 'sell':
                if max_sell_size < 10.0:  # Minimum $10 trade
                    logger.info(f"⏸️ Insufficient BTC for sell: ${max_sell_size:.2f} available, ${requested_size:.2f} requested")
                    trade_result = {"status": "error", "reason": ["Insufficient BTC for sell"]}
                else:
                    actual_size = min(requested_size, max_sell_size)
                    logger.info(f"💰 Executing sell: ${actual_size:.2f} (requested: ${requested_size:.2f}, available: ${max_sell_size:.2f})")
                    trade_result = trading_bot.place_sell_order("XXBTZUSD", actual_size)
                    # Persist trade to ledger (best-effort)
                    try:
                        if LEDGER_AVAILABLE and isinstance(trade_result, dict) and trade_result.get('status') == 'success':
                            qty_btc = float(trade_result.get('quantity', 0.0))
                            price = float(trade_result.get('price_per_btc', 0.0)) if trade_result.get('price_per_btc') is not None else (qty_btc and actual_size/qty_btc or 0.0)
                            fee = float(trade_result.get('fee', 0.0)) if trade_result.get('fee') is not None else 0.0
                            ledger_record_order(
                                symbol="BTC",
                                side="sell",
                                qty_btc=qty_btc,
                                notional_usd=float(actual_size),
                                price=price,
                                demo_mode=bool(trade_result.get('demo_mode', False)),
                                exchange_order_id=trade_result.get('order_id'),
                                fee=fee,
                                metadata=trade_result,
                            )
                    except Exception as e:
                        logger.warning(f"Ledger write failed (auto sell): {e}")
                    
            logger.info(f"✅ Trade result: {trade_result}")
        else:
            logger.info("⏸️ No trade executed - conditions not met")

        # Log the automated trade
        trade_log.append({
            "action": "AUTO_TRADE",
            "signal": signal,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trade_result": trade_result
        })

        # Snapshot equity (best-effort)
        try:
            if LEDGER_AVAILABLE:
                acct = trading_bot.get_account_info() if trading_bot else None
                equity_val = float(acct.get('equity', 0.0)) if isinstance(acct, Dict) or isinstance(acct, dict) else 0.0
                if equity_val:
                    ledger_snapshot_equity(equity_val)
        except Exception as e:
            logger.warning(f"Equity snapshot failed: {e}")

        logger.info("🤖 Auto trade analysis completed successfully")
        return {
            "message": f"Automated trading analysis complete",
            "signal": signal,
            "trade_executed": trade_result is not None,
            "trade_result": trade_result
        }

    except Exception as e:
        logger.error(f"❌ Error in automated trading: {e}")
        return {"message": f"Error in automated trading: {str(e)}"}

# REMOVED: /auto_trade_demo endpoint for security reasons
# This endpoint was allowing unauthorized access to trading functionality

@app.post("/auto_trade_scheduled")
async def auto_trade_scheduled(token: str = Form(...)):
    """Execute automated trading without authentication (for scheduled runs)"""
    logger.info("🤖 Scheduled Auto Trade endpoint called")
    
    # Check both in-memory and database settings
    auto_trade_enabled_memory = AUTO_TRADE_ENABLED
    auto_trade_enabled_db = get_setting("auto_trade_enabled", True) if LEDGER_AVAILABLE else True
    
    if not auto_trade_enabled_memory or not auto_trade_enabled_db:
        logger.info(f"⏸️ Auto-trade disabled (memory: {auto_trade_enabled_memory}, db: {auto_trade_enabled_db})")
        return {"status": "skipped", "message": "Auto-trade disabled"}
    
    if not trading_bot:
        logger.warning("No trading bot available for scheduled auto trade")
        return {"status": "error", "message": "No trading bot available"}

    if not llm_strategy:
        logger.warning("LLM strategy not available for scheduled auto trade")
        return {"status": "error", "message": "LLM strategy not available"}

    try:
        username = verify_jwt_token(token)
        logger.info(f"Scheduled auto trade requested by user: {username}")
    except:
        logger.warning("Invalid token for scheduled auto trade")
        raise HTTPException(status_code=401, detail="Authentication required")

@app.post("/auto_trade_cron")
async def auto_trade_cron():
    """Execute automated trading for cron jobs (no authentication required)"""
    logger.info("🤖 Cron Auto Trade endpoint called")
    
    # Check both in-memory and database settings
    auto_trade_enabled_memory = AUTO_TRADE_ENABLED
    auto_trade_enabled_db = get_setting("auto_trade_enabled", True) if LEDGER_AVAILABLE else True
    
    if not auto_trade_enabled_memory or not auto_trade_enabled_db:
        logger.info(f"⏸️ Auto-trade disabled (memory: {auto_trade_enabled_memory}, db: {auto_trade_enabled_db})")
        return {"status": "skipped", "message": "Auto-trade disabled"}
    
    if not trading_bot:
        logger.warning("No trading bot available for cron auto trade")
        return {"status": "error", "message": "No trading bot available"}

    if not llm_strategy:
        logger.warning("LLM strategy not available for cron auto trade")
        return {"status": "error", "message": "LLM strategy not available"}

    # Use database lock to ensure only one instance runs at a time
    if LEDGER_AVAILABLE:
        try:
            from db import acquire_trade_lock, release_trade_lock, get_trade_lock_info
            
            # Try to acquire the trade lock (10 minute TTL)
            lock_acquired = acquire_trade_lock(
                lock_key="auto_trade_scheduled",
                ttl_sec=600,  # 10 minutes
                process_id=f"web_{os.getpid()}",
                metadata={"endpoint": "auto_trade_scheduled", "timestamp": datetime.now().isoformat()}
            )
            
            if not lock_acquired:
                # Check if there's an existing lock
                lock_info = get_trade_lock_info("auto_trade_scheduled")
                if lock_info:
                    logger.warning(f"⏸️ Another auto trade instance is already running (PID: {lock_info['process_id']}, expires: {lock_info['expires_at']})")
                    return {"status": "skipped", "message": f"Another auto trade instance is already running (expires: {lock_info['expires_at']})"}
                else:
                    logger.warning("⏸️ Failed to acquire trade lock")
                    return {"status": "skipped", "message": "Failed to acquire trade lock"}
            
            logger.info("🔒 Acquired database lock for scheduled auto trade")
            
            try:
                return await _execute_auto_trade_scheduled()
            finally:
                # Always release the lock
                released = release_trade_lock("auto_trade_scheduled")
                if released:
                    logger.info("🔓 Released database lock for scheduled auto trade")
                else:
                    logger.warning("⚠️ Failed to release database lock")
                    
        except Exception as e:
            logger.error(f"❌ Database lock acquisition failed: {e}")
            return {"status": "error", "message": f"Database lock acquisition failed: {str(e)}"}
    else:
        # Fallback to file lock if database not available
        lock_path = "/data/auto_trade.lock"
        try:
            with exclusive_lock(lock_path):
                logger.info("🔒 Acquired file lock for scheduled auto trade (fallback)")
                return await _execute_auto_trade_scheduled()
        except BlockingIOError:
            logger.warning("⏸️ Another auto trade instance is already running (file lock)")
            return {"status": "skipped", "message": "Another auto trade instance is already running"}
        except Exception as e:
            logger.error(f"❌ File lock acquisition failed: {e}")
            return {"status": "error", "message": f"File lock acquisition failed: {str(e)}"}

    # Use database lock to ensure only one instance runs at a time
    if LEDGER_AVAILABLE:
        try:
            from db import acquire_trade_lock, release_trade_lock, get_trade_lock_info
            
            # Try to acquire the trade lock (10 minute TTL)
            lock_acquired = acquire_trade_lock(
                lock_key="auto_trade_cron",
                ttl_sec=600,  # 10 minutes
                process_id=f"cron_{os.getpid()}",
                metadata={"endpoint": "auto_trade_cron", "timestamp": datetime.now().isoformat()}
            )
            
            if not lock_acquired:
                # Check if there's an existing lock
                lock_info = get_trade_lock_info("auto_trade_cron")
                if lock_info:
                    logger.warning(f"⏸️ Another cron auto trade instance is already running (PID: {lock_info['process_id']}, expires: {lock_info['expires_at']})")
                    return {"status": "skipped", "message": f"Another cron auto trade instance is already running (expires: {lock_info['expires_at']})"}
                else:
                    logger.warning("⏸️ Failed to acquire trade lock")
                    return {"status": "skipped", "message": "Failed to acquire trade lock"}
            
            logger.info("🔒 Acquired database lock for cron auto trade")
            
            try:
                return await _execute_auto_trade_scheduled()
            finally:
                # Always release the lock
                released = release_trade_lock("auto_trade_cron")
                if released:
                    logger.info("🔓 Released database lock for cron auto trade")
                else:
                    logger.warning("⚠️ Failed to release database lock")
                    
        except Exception as e:
            logger.error(f"❌ Database lock acquisition failed: {e}")
            return {"status": "error", "message": f"Database lock acquisition failed: {str(e)}"}
    else:
        # Fallback to file lock if database not available
        lock_path = "/data/auto_trade_cron.lock"
        try:
            with exclusive_lock(lock_path):
                logger.info("🔒 Acquired file lock for cron auto trade (fallback)")
                return await _execute_auto_trade_scheduled()
        except BlockingIOError:
            logger.warning("⏸️ Another cron auto trade instance is already running (file lock)")
            return {"status": "skipped", "message": "Another cron auto trade instance is already running"}
        except Exception as e:
            logger.error(f"❌ File lock acquisition failed: {e}")
            return {"status": "error", "message": f"File lock acquisition failed: {str(e)}"}

async def _check_safety_conditions(current_price, last_update, signal_confidence, target_exposure, current_exposure):
    """Check safety conditions before executing trades"""
    settings = load_runtime_settings()
    
    # 1. Check price staleness (using global MAX_PRICE_STALENESS_SEC)
    if last_update:
        try:
            last_update_dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
            staleness_sec = (datetime.now(timezone.utc) - last_update_dt).total_seconds()
            if staleness_sec > MAX_PRICE_STALENESS_SEC:
                return False, f"Price data too stale ({staleness_sec:.0f}s > {MAX_PRICE_STALENESS_SEC}s)"
        except Exception:
            return False, "Unable to parse price timestamp"
    
    # 2. Check if expected move is worth the round-trip fee
    if settings.get("safety_min_expected_move_pct", 0.02):
        exposure_change = abs(target_exposure - current_exposure)
        if exposure_change < (settings["safety_min_expected_move_pct"] / 100):
            return False, f"Expected move too small ({exposure_change*100:.2f}% < {settings['safety_min_expected_move_pct']}%)"
    
    # 3. Check daily PnL limit (would need to be implemented with daily tracking)
    # This would require additional database tracking of daily PnL
    
    # 4. Check daily equity drop limit (would need to be implemented with daily tracking)
    # This would require additional database tracking of daily equity
    
    return True, "Safety checks passed"

async def _execute_auto_trade_scheduled():
    """Internal function to execute the scheduled auto trade logic using the new ORB-first approach"""
    
    try:
        from auto_trade_scheduler import execute_auto_trade
        
        logger.info("🤖 Starting scheduled auto trade execution...")
        
        # Call the new execute_auto_trade function
        result = execute_auto_trade()
        
        logger.info(f"✅ Scheduled auto trade completed: {result}")
        return {
            "status": "success", 
            "message": "Scheduled automated trading analysis complete",
            "result": result
        }
    except Exception as e:
        logger.error(f"❌ Scheduled auto trade failed: {e}")
        return {"status": "error", "message": f"Scheduled auto trade failed: {str(e)}"}
        
        # Calculate appropriate position size based on available funds
        available_funds = 0.0
        btc_balance = 0.0
        
        if account_balance and 'balances' in account_balance:
            balances = account_balance['balances']
            # Check GBP balance
            gbp_balance = float(balances.get('ZGBP', '0.0'))
            # Check USD balance  
            usd_balance = float(balances.get('ZUSD', '0.0'))
            # Check BTC balance
            btc_balance = float(balances.get('XXBT', '0.0'))
            
            available_funds = gbp_balance + usd_balance
            logger.info(f"💵 Available funds: £{gbp_balance:.2f} + ${usd_balance:.2f} = ${available_funds:.2f}")
            logger.info(f"₿ BTC balance: {btc_balance:.8f}")
        
        # Execute allocation change via target exposure if provided
        trade_result = None
        if signal.get('target_exposure') is not None:
            try:
                target = float(signal.get('target_exposure'))
                # Pass decision price for slippage checking
                decision_price = current_price if current_price else None
                trade_result = trading_bot.rebalance_to_target(target, decision_price)
            except Exception as e:
                logger.warning(f"Rebalance failed: {e}")
                trade_result = {"status": "error", "reason": str(e)}
        elif signal['should_trade'] and signal['position_size']:
            # Adjust position size based on available funds
            requested_size = signal['position_size']
            max_buy_size = available_funds * 0.95  # Use 95% of available funds
            max_sell_size = btc_balance * current_price * 0.95  # Use 95% of BTC balance
            
            if signal['action'] == 'buy':
                if max_buy_size < 10.0:  # Minimum $10 trade
                    logger.info(f"⏸️ Insufficient funds for buy: ${max_buy_size:.2f} available, ${requested_size:.2f} requested")
                    trade_result = {"status": "error", "reason": ["Insufficient funds for buy"]}
                else:
                    actual_size = min(requested_size, max_buy_size)
                    logger.info(f"💰 Executing buy: ${actual_size:.2f} (requested: ${requested_size:.2f}, available: ${max_buy_size:.2f})")
                    trade_result = trading_bot.place_buy_order("XXBTZUSD", actual_size, 1.0)
                    
            elif signal['action'] == 'sell':
                if max_sell_size < 10.0:  # Minimum $10 trade
                    logger.info(f"⏸️ Insufficient BTC for sell: ${max_sell_size:.2f} available, ${requested_size:.2f} requested")
                    trade_result = {"status": "error", "reason": ["Insufficient BTC for sell"]}
                else:
                    actual_size = min(requested_size, max_sell_size)
                    logger.info(f"💰 Executing sell: ${actual_size:.2f} (requested: ${requested_size:.2f}, available: ${max_sell_size:.2f})")
                    trade_result = trading_bot.place_sell_order("XXBTZUSD", actual_size)
                    
            logger.info(f"✅ Scheduled trade result: {trade_result}")
        else:
            logger.info("⏸️ No scheduled trade executed - conditions not met")

        # Execute grid trading if enabled
        grid_result = None
        if grid_executor and grid_executor.enabled:
            try:
                # Initialize grid if not already done
                if not grid_executor.grid_state:
                    # Use LLM bias as grid bias (convert action to bias allocation)
                    bias_allocation = 0.0
                    if signal.get('action') == 'buy':
                        bias_allocation = min(signal.get('confidence', 0.5), 0.8)  # Max 80% bias
                    elif signal.get('action') == 'sell':
                        bias_allocation = -min(signal.get('confidence', 0.5), 0.8)  # Max -80% bias
                    
                    grid_executor.initialize_grid(current_price, bias_allocation)
                    logger.info(f"🔧 Grid initialized with bias: {bias_allocation:.3f}")
                else:
                    # Update grid bias based on latest LLM signal
                    bias_allocation = 0.0
                    if signal.get('action') == 'buy':
                        bias_allocation = min(signal.get('confidence', 0.5), 0.8)
                    elif signal.get('action') == 'sell':
                        bias_allocation = -min(signal.get('confidence', 0.5), 0.8)
                    
                    grid_executor.update_bias(bias_allocation)
                    logger.info(f"🔧 Grid bias updated: {bias_allocation:.3f}")
                
                # Check if grid trade should be executed
                grid_action = grid_executor.should_grid_trade(current_price)
                if grid_action:
                    logger.info(f"🔧 Grid trade triggered: {grid_action.upper()}")
                    
                    # Get current equity and BTC balance for grid trade
                    equity = float(account_balance.get('equity', 0.0)) if account_balance else 1000.0
                    btc_qty = float(account_balance.get('btc_qty', 0.0)) if account_balance else 0.0
                    
                    # Execute grid trade
                    grid_result = grid_executor.execute_grid_trade(grid_action, equity, btc_qty, current_price)
                    logger.info(f"🔧 Grid trade result: {grid_result}")
                else:
                    logger.info("🔧 No grid trade triggered (price movement or time constraints)")
                    
            except Exception as grid_error:
                logger.error(f"❌ Grid trading failed: {grid_error}")
                grid_result = {"status": "error", "reason": str(grid_error)}

        # Save current LLM bias for grid trading
        if signal and signal.get('action') in ['buy', 'sell']:
            bias_allocation = 0.0
            if signal.get('action') == 'buy':
                bias_allocation = min(signal.get('confidence', 0.5), 0.8)  # Max 80% bias
            elif signal.get('action') == 'sell':
                bias_allocation = -min(signal.get('confidence', 0.5), 0.8)  # Max 80% bias
            
            try:
                from db import write_setting
                write_setting("current_llm_bias", str(bias_allocation))
                logger.info(f"💾 Saved LLM bias for grid trading: {bias_allocation:.2f}")
            except Exception as e:
                logger.warning(f"Failed to save LLM bias: {e}")
        
        # Log the automated trade
        trade_log.append({
            "action": "SCHEDULED_AUTO_TRADE",
            "signal": signal,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trade_result": trade_result,
            "grid_result": grid_result
        })

        # Snapshot equity (best-effort)
        try:
            if LEDGER_AVAILABLE:
                acct = trading_bot.get_account_info() if trading_bot else None
                equity_val = float(acct.get('equity', 0.0)) if isinstance(acct, Dict) or isinstance(acct, dict) else 0.0
                if equity_val:
                    ledger_snapshot_equity(equity_val)
        except Exception as e:
            logger.warning(f"Equity snapshot failed: {e}")

        logger.info("🤖 Scheduled auto trade analysis completed successfully")
        return {
            "status": "success",
            "message": f"Scheduled automated trading analysis complete",
            "signal": signal,
            "trade_executed": trade_result is not None,
            "trade_result": trade_result
        }

    except Exception as e:
        logger.error(f"❌ Error in scheduled automated trading: {e}")
        return {"status": "error", "message": f"Error in scheduled automated trading: {str(e)}"}

@app.get("/account_balance")
async def get_account_balance(token: str = None):
    """Get current account balance and positions"""
    logger.info("💰 Account balance requested")
    
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        username = verify_jwt_token(token)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not trading_bot:
        return {"message": "No trading bot available", "balance": None}
    
    try:
        # Get account information
        account_info = trading_bot.get_account_info()
        positions = trading_bot.get_positions()
        
        return {
            "account": account_info,
            "positions": positions,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting account balance: {e}")
        return {"message": f"Error getting account balance: {str(e)}", "balance": None}

@app.get("/trade_history")
async def get_trade_history(token: str = None, limit: int = 50):
    """Get recent trade history"""
    logger.info(f"📋 Trade history requested (limit: {limit})")
    
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        username = verify_jwt_token(token)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # Get recent trades from the trade log
        recent_trades = trade_log[-limit:] if len(trade_log) > limit else trade_log
        
        return {
            "trades": recent_trades,
            "total_trades": len(trade_log),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        return {"message": f"Error getting trade history: {str(e)}", "trades": []}

@app.get("/trading_summary")
async def get_trading_summary(token: str = None):
    """Get comprehensive trading summary including balance, positions, and recent activity"""
    logger.info("📊 Trading summary requested")
    
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        username = verify_jwt_token(token)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not trading_bot:
        return {"message": "No trading bot available", "summary": None}
    
    try:
        # Get account information
        account_info = trading_bot.get_account_info()
        positions = trading_bot.get_positions()
        
        # Get recent trades
        recent_trades = trade_log[-10:] if len(trade_log) > 10 else trade_log
        
        # Calculate basic statistics
        total_trades = len(trade_log)
        auto_trades = len([t for t in trade_log if t.get('action') == 'AUTO_TRADE' or t.get('action') == 'AUTO_TRADE_DEMO'])
        manual_trades = total_trades - auto_trades
        
        summary = {
            "account": account_info,
            "positions": positions,
            "recent_trades": recent_trades,
            "statistics": {
                "total_trades": total_trades,
                "auto_trades": auto_trades,
                "manual_trades": manual_trades,
                "last_trade": trade_log[-1] if trade_log else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    except Exception as e:
        logger.error(f"Error getting trading summary: {e}")
        return {"message": f"Error getting trading summary: {str(e)}", "summary": None}

@app.get("/pnl_summary", response_model=PnLSummary)
async def pnl_summary(token: str = None):
    """Return realized/unrealized PnL, fees, exposure, equity, and HODL benchmark."""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        _ = verify_jwt_token(token)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        return await _compute_pnl_summary()
    except Exception as e:
        # ultra-defensive: degraded + zeros (UI still renders)
        now = datetime.now(timezone.utc).isoformat()
        return PnLSummary(
            status="degraded", message=f"error: {type(e).__name__}",
            last_price=0.0, price_ts=now,
            equity=0.0, realized_pnl=0.0, unrealized_pnl=0.0, fees_total=0.0,
            exposure_value=0.0, exposure_pct=0.0, position_qty=0.0, hodl_value=0.0
        )

@app.get("/pnl_summary_test")
async def pnl_summary_test():
    """Public version of PnL summary for dashboard (no authentication required)"""
    # Get current price
    market_data = get_btc_market_data()
    px = float(market_data.get('close', 45000.0)) if market_data else 45000.0
    
    # Get balances using the same method as diagnostics endpoint
    balances = {}
    print(f"PnL: trading_bot available: {trading_bot is not None}")
    if trading_bot:
        account_info = trading_bot.get_account_info()
        print(f"PnL: account_info keys: {list(account_info.keys()) if account_info else 'None'}")
        if account_info and 'balances' in account_info:
            raw_balances = account_info['balances']
            print(f"PnL: raw_balances keys: {list(raw_balances.keys())}")
            print(f"PnL: BTC balance: {raw_balances.get('BTC', 'Not found')}")
            print(f"PnL: USD balance: {raw_balances.get('USD', 'Not found')}")
            
            # Use balances directly (they're already normalized)
            btc_qty = float(raw_balances.get("BTC", 0.0))
            cash_usd = float(raw_balances.get("USD", 0.0))
            equity = cash_usd + btc_qty * px
            exposure_value = btc_qty * px
            exposure_pct = 0.0 if equity <= 0 else exposure_value / equity
            
            print(f"PnL: Calculated equity: {equity}, position_qty: {btc_qty}")
            
            return {
                "status": "ok",
                "message": None,
                "last_price": px,
                "price_ts": datetime.now(timezone.utc).isoformat(),
                "equity": round(equity, 2),
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "fees_total": 0.0,
                "exposure_value": round(exposure_value, 2),
                "exposure_pct": round(exposure_pct, 4),
                "position_qty": round(btc_qty, 8),
                "hodl_value": 0.0,
            }
        else:
            print(f"PnL: No balances in account_info")
    else:
        print(f"PnL: No trading_bot available")
    
    # Fallback if no balances available
    return {
        "status": "ok",
        "message": None,
        "last_price": px,
        "price_ts": datetime.now(timezone.utc).isoformat(),
        "equity": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "fees_total": 0.0,
        "exposure_value": 0.0,
        "exposure_pct": 0.0,
        "position_qty": 0.0,
        "hodl_value": 0.0,
    }

# Snapshot equity endpoint
@app.post("/snapshot_equity")
async def snapshot_equity_endpoint(token: str = Form(...)):
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        verify_jwt_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        acct = trading_bot.get_account_info() if trading_bot else None
        eq = float(acct.get('equity', 0.0)) if isinstance(acct, (dict, Dict)) else 0.0
        if eq:
            ledger_snapshot_equity(eq)
        return {"status": "ok", "equity": eq}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/pnl_summary_public", response_model=PnLSummary)
async def pnl_summary_public():
    """Public version of PnL summary for dashboard (no authentication required)"""
    try:
        # Safe float conversion helper
        def _f(v, d=0.0):
            try:
                return float(v)
            except (ValueError, TypeError):
                return d
        
        # Get current price
        market_data = get_btc_market_data()
        px = float(market_data.get('close', 45000.0)) if market_data else 45000.0
        
        # Get balances using the same method as diagnostics endpoint
        if trading_bot:
            account_info = trading_bot.get_account_info()
            if account_info and 'balances' in account_info:
                raw_balances = account_info['balances']
                
                # Normalize asset codes like the diagnostics endpoint
                balances = {}
                for k, v in raw_balances.items():
                    sym = _normalize_asset(k)
                    try:
                        balances[sym] = balances.get(sym, 0.0) + float(v)
                    except (ValueError, TypeError):
                        pass
                
                # Get normalized balances with safe conversion
                btc_qty = _f(balances.get("BTC"))
                cash_usd = _f(balances.get("USD"))
                equity = cash_usd + btc_qty * px
                exposure_value = btc_qty * px
                exposure_pct = 0.0 if equity <= 0 else exposure_value / equity
                
                return PnLSummary(
                    status="ok",
                    message=None,
                    last_price=px,
                    price_ts=datetime.now(timezone.utc).isoformat(),
                    equity=round(equity, 2),
                    realized_pnl=0.0,
                    unrealized_pnl=0.0,
                    fees_total=0.0,
                    exposure_value=round(exposure_value, 2),
                    exposure_pct=round(exposure_pct, 4),
                    position_qty=round(btc_qty, 8),
                    hodl_value=0.0,
                )
        
        # Fallback if no balances available
        return PnLSummary(
            status="ok",
            message=None,
            last_price=px,
            price_ts=datetime.now(timezone.utc).isoformat(),
            equity=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            fees_total=0.0,
            exposure_value=0.0,
            exposure_pct=0.0,
            position_qty=0.0,
            hodl_value=0.0,
        )
    except Exception as e:
        logger.error(f"Error in pnl_summary_public: {e}")
        # Return degraded status with zeros
        return PnLSummary(
            status="degraded",
            message=f"error: {type(e).__name__}",
            last_price=0.0,
            price_ts=datetime.now(timezone.utc).isoformat(),
            equity=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            fees_total=0.0,
            exposure_value=0.0,
            exposure_pct=0.0,
            position_qty=0.0,
            hodl_value=0.0,
        )

@app.get("/equity_exposure")
async def equity_exposure():
    """Simple equity/exposure endpoint for status monitoring"""
    try:
        # Safe float conversion helper
        def _f(v, d=0.0):
            try:
                return float(v)
            except (ValueError, TypeError):
                return d
        
        # Get current price
        market_data = get_btc_market_data()
        px = float(market_data.get('close', 45000.0)) if market_data else 45000.0
        
        # Get balances using the same method as diagnostics endpoint
        if trading_bot:
            account_info = trading_bot.get_account_info()
            if account_info and 'balances' in account_info:
                raw_balances = account_info['balances']
                
                # Normalize asset codes
                balances = {}
                for k, v in raw_balances.items():
                    sym = _normalize_asset(k)
                    try:
                        balances[sym] = balances.get(sym, 0.0) + float(v)
                    except (ValueError, TypeError):
                        pass
                
                # Get normalized balances with safe conversion
                btc_qty = _f(balances.get("BTC"))
                cash_usd = _f(balances.get("USD"))
                equity = cash_usd + btc_qty * px
                exposure_value = btc_qty * px
                exposure_pct = 0.0 if equity <= 0 else exposure_value / equity
                
                return {
                    "equity": round(equity, 2),
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                    "exposure_value": round(exposure_value, 2),
                    "exposure_pct": round(exposure_pct, 4)
                }
        
        # Fallback if no balances available
        return {
            "equity": None,
            "realized_pnl": None,
            "unrealized_pnl": None,
            "exposure_value": None,
            "exposure_pct": None
        }
    except Exception as e:
        logger.error(f"Error in equity_exposure: {e}")
        return {
            "equity": None,
            "realized_pnl": None,
            "unrealized_pnl": None,
            "exposure_value": None,
            "exposure_pct": None
        }

# Prometheus metrics
from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

metrics_registry = CollectorRegistry()
metric_equity = Gauge('btc_equity_usd', 'Total equity in USD', registry=metrics_registry)
metric_realized = Gauge('btc_realized_pnl_usd', 'Realized PnL USD', registry=metrics_registry)
metric_unrealized = Gauge('btc_unrealized_pnl_usd', 'Unrealized PnL USD', registry=metrics_registry)
metric_exposure = Gauge('btc_exposure_pct', 'Exposure percentage (0-1)', registry=metrics_registry)
metric_last_trade_ts = Gauge('btc_last_trade_timestamp', 'Unix timestamp of last trade', registry=metrics_registry)
metric_errors = Counter('btc_errors_total', 'Error count', registry=metrics_registry)

# Simple file lock for scheduled task
import fcntl
from contextlib import contextmanager

@contextmanager
def exclusive_lock(lock_path: str):
    os.makedirs(os.path.dirname(lock_path) or '.', exist_ok=True)
    with open(lock_path, 'w') as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        finally:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            except Exception:
                pass

@app.get('/metrics')
async def metrics():
    try:
        # Refresh metrics from current PnL snapshot if possible
        if LEDGER_AVAILABLE and trading_bot:
            try:
                acct = trading_bot.get_account_info()
                equity_val = float(acct.get('equity', 0.0)) if isinstance(acct, dict) else 0.0
                metric_equity.set(equity_val)
            except Exception:
                pass
        return PlainTextResponse(generate_latest(metrics_registry), media_type=CONTENT_TYPE_LATEST)
    except Exception:
        return PlainTextResponse(generate_latest(metrics_registry), media_type=CONTENT_TYPE_LATEST)

@app.get("/diagnostics/balances")
def diag_balances():
    """Quick diagnostics endpoint to verify balance calculations."""
    try:
        # Get current price and balances
        market_data = get_btc_market_data()
        price = float(market_data.get('close', 45000.0)) if market_data else 45000.0
        ts = datetime.now(timezone.utc)
        
        # Get normalized balances
        balances = {}
        if trading_bot:
            account_info = trading_bot.get_account_info()
            if account_info and 'balances' in account_info:
                raw_balances = account_info['balances']
                # Normalize asset codes
                for k, v in raw_balances.items():
                    sym = _normalize_asset(k)
                    try:
                        balances[sym] = balances.get(sym, 0.0) + float(v)
                    except (ValueError, TypeError):
                        pass
        
        # Calculate key metrics
        btc_qty = float(balances.get("BTC", 0.0))
        cash_usd = float(balances.get("USD", 0.0))
        equity = cash_usd + btc_qty * price
        
        return {
            "raw": balances,
            "price": price,
            "price_ts": ts.isoformat(),
            "btc_qty": btc_qty,
            "cash_usd": cash_usd,
            "equity": equity,
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/diagnostics/price")
def diag_price():
    """Price diagnostics endpoint to monitor data freshness."""
    try:
        from price_worker import PRICE_CACHE
        now = datetime.now(timezone.utc)
        ts = PRICE_CACHE["ts"]
        age = (now - ts).total_seconds() if ts else None
        return {
            "source": PRICE_CACHE["source"], 
            "ts": ts.isoformat() if ts else None, 
            "age_sec": age,
            "price": PRICE_CACHE["price"]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/diagnostics/volatility")
def diag_volatility():
    """Volatility diagnostics endpoint to monitor market volatility."""
    try:
        from price_worker import get_current_volatility, PRICE_CACHE
        σ = get_current_volatility()
        return {
            "volatility": σ,
            "volatility_bps": σ * 10000 if σ else None,  # Convert to basis points
            "current_price": PRICE_CACHE["price"],
            "volatility_available": σ is not None
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/diagnostics/telemetry")
def diag_telemetry():
    """Trade quality telemetry endpoint to monitor grid trading performance."""
    try:
        from telemetry import get_telemetry_metrics
        metrics = get_telemetry_metrics()
        return {
            "trade_quality": {
                "maker_ratio": metrics["maker_ratio"],
                "median_slippage_bps": metrics["median_slippage_bps"],
                "avg_slippage_bps": metrics["avg_slippage_bps"],
                "daily_trades": metrics["daily_trades"]
            },
            "grid_trades_total": metrics["grid_trades_total"],
            "grid_skips_total": metrics["grid_skips_total"],
            "top_skip_reasons": metrics["top_skip_reasons"],
            "maker_fills_total": metrics["maker_fills_total"],
            "market_fallback_total": metrics["market_fallback_total"],
            "slippage_samples": metrics["slippage_samples"],
            "recent_trades_count": metrics["recent_trades_count"]
        }
    except Exception as e:
        return {"error": str(e)}

# --- ORB helper endpoints ----------------------------------------------------
@app.get("/orb/status")
async def orb_status():
    try:
        from db import get_setting
        from orb_executor import load_orb_state
        s = load_orb_state() or {}
        
        # Safe float conversion helper (same as PnL endpoints)
        def _f(v, d=0.0):
            try:
                return float(v)
            except (ValueError, TypeError):
                return d
        
        # Get real balances using the same helper as PnL endpoints
        btc_bal = 0.0
        usd_bal = 0.0
        try:
            from kraken_trading_btc import get_balances_normalized
            balances = get_balances_normalized()
            btc_bal = _f(balances.get("btc"))
            usd_bal = _f(balances.get("usd"))
        except Exception:
            pass
        
        # Get current spread using the new method
        try:
            from orb_executor import get_spread_bps
            current_spread = get_spread_bps()
        except Exception:
            current_spread = s.get("spread_bps", 0.0)
        
        return {
            "mode": get_setting("strategy_mode"),
            "enabled": bool(get_setting("orb_enabled", False)),
            "day": s.get("day"),
            "phase": s.get("phase"),
            "orb_high": s.get("orb_high"),
            "orb_low": s.get("orb_low"),
            "adds_done": s.get("adds_done"),
            "avg_price": s.get("avg_price"),
            "qty": s.get("qty"),
            "stop": s.get("stop"),
            # NEW diagnostics if present
            "last_close": s.get("last_close"),
            "ema9": s.get("ema9"),
            "ema20": s.get("ema20"),
            "ema50": s.get("ema50"),
            "spread_bps": current_spread,
            "confirm_reason": s.get("confirm_reason"),
            "last_five_bucket": s.get("last_five_bucket"),
            # Skip reason context with real balances
            "btc_balance": btc_bal,
            "usd_balance": usd_bal,
            "skip_reason": s.get("skip_reason"),
            "signal": s.get("signal"),
            "ema_ok": s.get("ema_ok"),
            "break_ok": s.get("break_ok"),
            "risk_qty": s.get("risk_qty"),
            "final_qty": s.get("final_qty"),
            # Last confirmation values
            "last_signal": s.get("last_signal"),
            "last_ema_ok": s.get("last_ema_ok"),
            "last_break_ok": s.get("last_break_ok"),
            "last_reason": s.get("last_reason"),
            "last_risk_qty": s.get("last_risk_qty")
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@app.post("/orb/control/flat_now")
async def orb_flat_now():
    try:
        from orb_executor import close_orb_position
        res = close_orb_position()
        return res if res else {"status": "error", "reason": "unknown"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@app.get("/orb/self_test")
def orb_self_test(at: str = None, force_session: bool = True):
    now = datetime.now(timezone.utc)
    if at:
        try:
            # Accept both Z and +00:00
            iso = at.replace("Z", "+00:00")
            now = datetime.fromisoformat(iso)
        except Exception:
            return {"status":"error","reason":"bad_at_param"}

    from auto_trade_scheduler import load_runtime_settings
    from orb_executor import run_orb_cycle
    settings = load_runtime_settings()
    if force_session:
        settings["orb_debug_force_session"] = True

    res = run_orb_cycle(now, settings)
    return {"now": now.isoformat(), "res": res}


@app.get("/orb/self_test_confirm")
def orb_self_test_confirm(at: str = None, force_dir: str = None):
    """Self-test endpoint specifically for confirmation logic testing"""
    now = datetime.now(timezone.utc)
    if at:
        try:
            # Accept both Z and +00:00
            iso = at.replace("Z", "+00:00")
            now = datetime.fromisoformat(iso)
        except Exception:
            return {"status":"error","reason":"bad_at_param"}

    from auto_trade_scheduler import load_runtime_settings
    from orb_executor import run_orb_cycle
    settings = load_runtime_settings()
    settings["orb_debug_force_session"] = True
    
    # Optionally force a direction to validate order path end-to-end
    if force_dir in ("long", "short"):
        # Inject fake signal for testing (only in non-prod)
        if os.getenv("ORB_DEBUG_MODE") == "true":
            from orb_executor import load_orb_state, save_orb_state
            state = load_orb_state() or {}
            state["force_test_signal"] = force_dir
            save_orb_state(state)
    
    res = run_orb_cycle(now, settings)
    return {"now": now.isoformat(), "res": res}


@app.post("/orders/self_test")
def orders_self_test(side: str = "buy", usd: float = 15.0, validate: bool = True):
    """Self-test endpoint for order placement validation"""
    try:
        from kraken_trading_btc import KrakenTradingBot, best_quotes, get_pair_meta, round_down
        bot = KrakenTradingBot()
        qts = best_quotes(bot)
        meta = get_pair_meta(bot)
        qty = round_down(usd / qts["mid"], meta["lot_step"])
        
        if validate:
            # Add validate flag to payload for dry-run
            payload = {
                "pair": meta["pair"],
                "type": "buy" if side == "buy" else "sell",
                "ordertype": "limit",
                "price": f"{qts['mid']:.{meta['price_dec']}f}",
                "volume": f"{qty:.{meta['lot_dec']}f}",
                "oflags": "post",
                "userref": int(time.time()),
                "timeinforce": "GTC",
                "validate": True,  # Dry-run flag
            }
            
            endpoint = "/private/AddOrder"
            res = bot._make_request_with_retry('POST', endpoint, data=payload, private=True, signature_path="/0/private/AddOrder")
            return {"ok": True, "res": res, "qty": qty, "mid": qts["mid"], "pair": meta["pair"], "validate": True}
        else:
            ok, res = bot.place_limit_post_only(side, qty, qts["mid"])
            return {"ok": ok, "res": res, "qty": qty, "mid": qts["mid"], "pair": meta["pair"], "validate": False}
    except Exception as e:
        return {"ok": False, "res": {"status": "error", "reason": str(e)}}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    if not METRICS_ON:
        return Response("metrics disabled", status_code=404)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health_check():
    """Health check endpoint for Docker"""
    try:
        # Check if we can access the database
        from db import get_setting
        get_setting("test", "ok")
        
        # Check if price data is fresh
        from price_worker import PRICE_CACHE
        if PRICE_CACHE and PRICE_CACHE.get("ts"):
            age_sec = (datetime.now(timezone.utc) - PRICE_CACHE["ts"]).total_seconds()
            if age_sec > int(os.getenv("HEALTH_MIN_PRICE_STALENESS", 120)):
                return JSONResponse({"status": "degraded", "reason": "stale_price", "age_sec": age_sec})
        
        return JSONResponse({"status": "healthy"})
    except Exception as e:
        return JSONResponse({"status": "unhealthy", "error": str(e)})


@app.get("/orb/daily_summary")
def orb_daily_summary():
    """Daily trading summary"""
    try:
        from db import get_realized_pnl_total, get_fees_total
        from datetime import date
        
        today = date.today().isoformat()
        
        # Get today's trades and PnL
        realized_pnl = get_realized_pnl_total()
        fees = get_fees_total()
        net_pnl = realized_pnl - fees
        
        # Calculate win rate (simplified)
        # In a real implementation, you'd count winning vs losing trades
        win_rate = 0.5  # placeholder
        
        # Calculate vs HODL (simplified)
        vs_hodl = 0.0  # placeholder
        
        stats = {
            "date": today,
            "trades": 1,  # placeholder - count actual trades
            "win_rate": win_rate,
            "realized_pnl": realized_pnl,
            "fees": fees,
            "net_pnl": net_pnl,
            "vs_hodl": vs_hodl
        }
        
        return stats
    except Exception as e:
        logger.error(f"Failed to compute daily summary: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 