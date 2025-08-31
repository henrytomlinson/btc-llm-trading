#!/usr/bin/env python3
"""
Automated Bitcoin Trading Scheduler
This script can be run via cron to automatically execute trading decisions
"""

import requests
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict
import time
import os
import sys

# Add the current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)

# Create a session for all requests with proper headers
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "btc-llm-scheduler/1.0"})

# Configure session for better reliability
SESSION.timeout = 30  # 30 second timeout for all requests

# Add the current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add the current directory to Python path to import db module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Grid executor imports
try:
    from grid_executor import GridState, GridExecutor, DEFAULT_GRID_ORDER_USD
    GRID_AVAILABLE = True
except ImportError:
    GRID_AVAILABLE = False

# Price worker import for fresh data
try:
    from price_worker import get_last_price, PRICE_CACHE
    PRICE_WORKER_AVAILABLE = True
except ImportError:
    PRICE_WORKER_AVAILABLE = False

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

# Grid trading constants
from math import floor

KRAKEN_PAIR = "XXBTZUSD"
LOT_STEP_BTC = 0.00001     # 1e-5 BTC lot (safe default)
PRICE_TICK   = 0.10        # $0.10 tick (safe default)

def _round_qty(q):
    return floor(q / LOT_STEP_BTC) * LOT_STEP_BTC

def _cap_by_exposure(side, qty, price, equity, max_exposure, btc_qty_now):
    """Cap quantity by exposure limits and current holdings."""
    # current allocation
    alloc_now = (btc_qty_now * price) / equity if equity > 0 else 0.0
    # max BTC value allowed
    max_val   = max_exposure * equity
    cur_val   = btc_qty_now * price

    if side == "buy":
        room_val = max(0.0, max_val - cur_val)
        max_qty  = room_val / price
        return max(0.0, min(qty, max_qty))
    else: # sell
        return max(0.0, min(qty, btc_qty_now))

def _current_candle_ts(dt: datetime, minutes=15):
    """Get the timestamp for the current 15-minute candle."""
    epoch = dt.replace(second=0, microsecond=0)
    off = (epoch.minute // minutes) * minutes
    return epoch.replace(minute=off)

def _is_same_candle(db, now):
    """Check if we've already traded in this candle, and update if not."""
    from db import read_settings, write_setting
    cur = _current_candle_ts(now).isoformat()
    last = (read_settings() or {}).get("last_trade_candle_ts")
    if last == cur: 
        return True
    write_setting("last_trade_candle_ts", cur); 
    return False

def _is_same_candle_per_side(side: str, now: datetime):
    """Check if we've already traded this side in this candle, and update if not."""
    from db import read_settings, write_setting
    cur = _current_candle_ts(now, minutes=15).isoformat()
    key = f"last_trade_candle_{side}_ts"
    settings = read_settings() or {}
    last = settings.get(key)
    if last == cur:
        return True
    settings[key] = cur
    write_setting(key, cur)
    return False

def mark_remainder_satisfied(userref: str):
    """Mark a remainder as satisfied to avoid retrying small unfilled portions."""
    from db import write_setting
    key = f"remainder_satisfied_{userref}"
    write_setting(key, datetime.now(timezone.utc).isoformat())
    logger.info(f"Marked remainder satisfied for userref: {userref}")

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

load_env_file()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/auto_trade.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Configuration
TRADING_URL = "https://henryt-btc.live/auto_trade_scheduled"
MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds

# Max staleness used by both summary and trading gates (same as main_btc.py)
MAX_PRICE_STALENESS_SEC = int(os.getenv("MAX_PRICE_STALENESS_SEC", "120"))

def load_grid_state(symbol: str) -> GridState:
    """Load grid state from database for a given symbol."""
    try:
        from db import read_settings
        settings = read_settings()
        state_data = settings.get(f"grid_state_{symbol}")
        if state_data:
            return GridState(**json.loads(state_data))
        return None
    except Exception as e:
        logger.warning(f"Failed to load grid state for {symbol}: {e}")
        return None

def save_grid_state(symbol: str, state: GridState):
    """Save grid state to database for a given symbol."""
    try:
        from db import write_setting
        state_data = json.dumps({
            "last_grid_price": state.last_grid_price,
            "bias_allocation": state.bias_allocation,
            "last_update": state.last_update.isoformat(),
            "grid_step_pct": state.grid_step_pct,
            "grid_order_usd": state.grid_order_usd,
            "max_grid_exposure": state.max_grid_exposure,
            "grid_trades_count": state.grid_trades_count
        })
        write_setting(f"grid_state_{symbol}", state_data)
    except Exception as e:
        logger.warning(f"Failed to save grid state for {symbol}: {e}")

def get_current_bias() -> float:
    """Get current bias allocation from the last LLM signal or default to 0."""
    try:
        from db import read_settings
        settings = read_settings()
        
        # Try to get the new persisted bias first
        bias_data = settings.get("llm_bias_BTC")
        if bias_data:
            try:
                data = json.loads(bias_data)
                bias = data.get("bias", 0.0)
                logger.info(f"📊 Retrieved persisted LLM bias: {bias:.3f}")
                return bias
            except Exception as e:
                logger.warning(f"Failed to parse persisted bias: {e}")
        
        # Fallback to old bias format
        bias_data = settings.get("current_llm_bias")
        if bias_data:
            bias = float(bias_data)
            logger.info(f"📊 Retrieved legacy LLM bias: {bias:.3f}")
            return bias
            
        logger.info("📊 No bias found, using neutral (0.0)")
        return 0.0
    except Exception as e:
        logger.warning(f"Failed to get current bias: {e}")
        return 0.0

def place_delta_notional(action: str, notional: float, settings: dict):
    """Place a grid micro-trade, respecting exposure, holdings and LLM bias."""
    try:
        from kraken_trading_btc import KrakenTradingBot
        bot = KrakenTradingBot()

        state   = bot.get_current_exposure()
        price   = state["price"]
        equity  = state["equity"]
        btc_qty = state.get("btc_qty", 0.0)

        qty_raw = max(0.0, notional / price)
        side    = "buy" if action == "buy" else "sell"

        # Cap by exposure and holdings
        qty_cap = _cap_by_exposure(side, qty_raw, price, equity,
                                   settings["max_exposure"], btc_qty)

        qty = _round_qty(qty_cap)
        if qty <= 0:
            logger.info("⛔ Grid trade cancelled (no room for %s)", side)
            return {"status":"skipped","reason":"exposure_or_holdings"}

        # Prefer maker-first; fallback to market (slippage-capped)
        ob = bot.get_orderbook_snapshot(KRAKEN_PAIR)
        if ob.get("status") == "success":
            limit_price = ob["best_ask"] if side == "buy" else ob["best_bid"]
            limit_price = round(limit_price / PRICE_TICK) * PRICE_TICK
            res = bot.place_limit_post_only(side, qty, limit_price, KRAKEN_PAIR)
            if res.get("status") == "success":
                logger.info("✅ Grid %s PO: %.6f BTC @ $%.2f", side, qty, limit_price)
                return res

        # Fallback
        res = bot.place_market_with_slippage_cap(side, qty, max_slippage_bps=10, symbol=KRAKEN_PAIR)
        logger.info("⚠️ Grid %s MK: %.6f BTC", side, qty)
        return res

    except Exception as e:
        logger.error(f"Failed to place grid trade: {e}")
        return {"status":"error","reason":str(e)}

def load_runtime_settings():
    """Load runtime settings from database."""
    try:
        from db import read_settings
        settings = read_settings()
        return {
            "max_exposure": float(settings.get("max_exposure", 0.8)),
            "cooldown_hours": float(settings.get("trade_cooldown_hours", 3)),
            "min_confidence": float(settings.get("min_confidence", 0.7)),
            "min_trade_delta": float(settings.get("min_trade_delta", 0.05)),
            "min_trade_delta_usd": float(settings.get("min_trade_delta_usd", 10.0)),
            "min_trade_delta_pct": float(settings.get("min_trade_delta_pct", 0.00)),
            "no_fee_mode": parse_bool(settings.get("no_fee_mode", True)),
            "grid_executor_enabled": parse_bool(settings.get("grid_executor_enabled", True)),
            "grid_step_pct": float(settings.get("grid_step_pct", 0.25)),
            "grid_order_usd": float(settings.get("grid_order_usd", 12.0)),
            "max_grid_exposure": float(settings.get("max_grid_exposure", 0.1)),
            # Strategy mode and ORB params
            "strategy_mode": str(settings.get("strategy_mode", "ORB")),
            "orb_enabled": parse_bool(settings.get("orb_enabled", False)),
            "orb_open_minutes": int(settings.get("orb_open_minutes", 15)),
            "orb_confirm_minutes": int(settings.get("orb_confirm_minutes", 5)),
            "orb_min_range_pct": float(settings.get("orb_min_range_pct", 0.002)),
            "orb_ema_fast": int(settings.get("orb_ema_fast", 9)),
            "orb_ema_mid": int(settings.get("orb_ema_mid", 20)),
            "orb_ema_slow": int(settings.get("orb_ema_slow", 50)),
            "orb_risk_per_trade": float(settings.get("orb_risk_per_trade", 0.005)),
            "orb_buffer_frac": float(settings.get("orb_buffer_frac", 0.10)),
            "orb_max_spread_bps": int(settings.get("orb_max_spread_bps", 8)),
            "orb_max_adds": int(settings.get("orb_max_adds", 2)),
            "orb_trail_atr_mult": float(settings.get("orb_trail_atr_mult", 2.0)),
        }
    except Exception as e:
        logger.warning(f"Failed to load settings from DB: {e}")
        # Fallback to environment variables
        return {
                    "max_exposure": float(os.getenv("MAX_EXPOSURE", "0.8")),
        "cooldown_hours": float(os.getenv("TRADE_COOLDOWN_HOURS", "1")),
        "min_confidence": float(os.getenv("MIN_CONFIDENCE", "0.8")),
            "min_trade_delta": float(os.getenv("MIN_TRADE_DELTA", "0.0")),
            "min_trade_delta_usd": float(os.getenv("MIN_TRADE_DELTA_USD", "10.0")),
            "min_trade_delta_pct": float(os.getenv("MIN_TRADE_DELTA_PCT", "0.00")),
            "no_fee_mode": parse_bool(os.getenv("NO_FEE_MODE", "True")),
            "grid_executor_enabled": parse_bool(os.getenv("GRID_EXECUTOR_ENABLED", "True")),
            "grid_step_pct": float(os.getenv("GRID_STEP_PCT", "0.25")),
            "grid_order_usd": float(os.getenv("GRID_ORDER_USD", "12.0")),
            "max_grid_exposure": float(os.getenv("MAX_GRID_EXPOSURE", "0.1")),
            # Strategy mode and ORB params
            "strategy_mode": str(os.getenv("STRATEGY_MODE", "ORB")),
            "orb_enabled": parse_bool(os.getenv("ORB_ENABLED", "True")),
            "orb_open_minutes": int(os.getenv("ORB_OPEN_MINUTES", "15")),
            "orb_confirm_minutes": int(os.getenv("ORB_CONFIRM_MINUTES", "5")),
            "orb_min_range_pct": float(os.getenv("ORB_MIN_RANGE_PCT", "0.002")),
            "orb_ema_fast": int(os.getenv("ORB_EMA_FAST", "9")),
            "orb_ema_mid": int(os.getenv("ORB_EMA_MID", "20")),
            "orb_ema_slow": int(os.getenv("ORB_EMA_SLOW", "50")),
            "orb_risk_per_trade": float(os.getenv("ORB_RISK_PER_TRADE", "0.005")),
            "orb_buffer_frac": float(os.getenv("ORB_BUFFER_FRAC", "0.10")),
            "orb_max_spread_bps": int(os.getenv("ORB_MAX_SPREAD_BPS", "8")),
            "orb_max_adds": int(os.getenv("ORB_MAX_ADDS", "2")),
            "orb_trail_atr_mult": float(os.getenv("ORB_TRAIL_ATR_MULT", "2.0")),
        }

def check_price_staleness():
    """Check if price data is stale before executing trades"""
    try:
        # Get current price data from the public endpoint
        response = SESSION.get(
            "https://henryt-btc.live/btc_data_public",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            status = data.get('status', 'unknown')
            
            # Lower the trade gate only when feed is healthy
            if status != "ok":
                logger.warning(f"🛡️ Skip trade: data status is {status}")
                return False, "degraded"
            
            if data.get('last_update'):
                try:
                    # Parse the timestamp
                    price_ts = datetime.fromisoformat(data['last_update'].replace('Z', '+00:00'))
                    staleness = (datetime.now(timezone.utc) - price_ts).total_seconds()
                    
                    # Keep your staleness & degraded gates. You can trade more, but only on fresh data.
                    if staleness > MAX_PRICE_STALENESS_SEC:
                        logger.warning(f"🛡️ Skip trade: price stale ({int(staleness)}s > {MAX_PRICE_STALENESS_SEC}s)")
                        return False, "degraded"
                    else:
                        logger.info(f"✅ Price data is fresh ({int(staleness)}s old)")
                        return True, None
                except Exception as e:
                    logger.warning(f"🛡️ Skip trade: unable to parse price timestamp: {e}")
                    return False, "degraded"
            else:
                logger.warning(f"🛡️ Skip trade: no timestamp in data")
                return False, "degraded"
        else:
            logger.warning(f"🛡️ Skip trade: unable to fetch price data (status {response.status_code})")
            return False, "degraded"
            
    except Exception as e:
        logger.warning(f"🛡️ Skip trade: price staleness check failed: {e}")
        return False, "degraded"

def execute_auto_trade(now: datetime | None = None) -> Dict[str, Any]:
    """
    Main scheduler entrypoint — called by /auto_trade (cron every minute).
    Runs ORB if enabled. Only falls back to other strategies when ORB disabled.
    Logs every outcome. Never raises (returns a dict instead).
    """
    now = now or datetime.now(timezone.utc)
    settings = load_runtime_settings()  # must contain orb_enabled, max_price_staleness_sec, max_spread_bps, etc.

    # 1) ORB path
    try:
        if settings.get("orb_enabled"):
            from orb_executor import run_orb_cycle
            res = run_orb_cycle(now, settings) or {"status": "error", "reason": "orb_cycle_returned_none"}
            logger.info("ORB_CYCLE %s", safe_json(res))
            return res
    except Exception as e:
        logger.exception("ORB_RUN_ERROR %s", e)

    # 2) (optional) Grid path
    try:
        if settings.get("grid_executor_enabled"):
            from grid_executor import run_grid_cycle
            res = run_grid_cycle(now, settings) or {"status":"error","reason":"grid_cycle_returned_none"}
            logger.info("GRID_CYCLE %s", safe_json(res))
            return res
    except Exception as e:
        logger.exception("GRID_RUN_ERROR %s", e)

    # 3) (optional) LLM path
    try:
        if settings.get("llm_enabled"):
            from llm_trading_strategy import run_llm_cycle
            res = run_llm_cycle(now, settings) or {"status":"error","reason":"llm_cycle_returned_none"}
            logger.info("LLM_CYCLE %s", safe_json(res))
            return res
    except Exception as e:
        logger.exception("LLM_RUN_ERROR %s", e)

    logger.info("SCHED_NOOP no strategy enabled")
    return {"status": "noop", "reason": "no_strategy_enabled"}

def main():
    """Main function to execute automated trading"""
    logger.info("🚀 Automated Bitcoin Trading Scheduler Started")
    logger.info(f"🕐 Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        result = execute_auto_trade()
        if result and result.get("status") != "error":
            logger.info("🎉 Automated trading completed successfully")
        else:
            logger.error("💥 Automated trading failed")
            exit(1)
            
    except Exception as e:
        logger.error(f"💥 Fatal error in automated trading: {e}")
        exit(1)

if __name__ == "__main__":
    main()
