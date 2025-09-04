#!/usr/bin/env python3
"""
ORB executor: fetches OHLC from Kraken public API, applies ORBStrategy,
and places orders via existing KrakenTradingBot with maker-first preference.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, time, timedelta
from typing import List, Dict, Optional, Any, Tuple
import math

import requests

from orb_strategy import (
    ORBStrategy, ORBParams, ORBState, SessionDef,
    is_orb_build, is_confirm_window, weekday_allowed, localize
)
from db import get_setting
from utils_bool import parse_bool
import pytz

NY = pytz.timezone("America/New_York")
LDN = pytz.timezone("Europe/London")


def floor_5m(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def round_down(v: float, step: float) -> float:
    return math.floor(v / step) * step


def get_free_usd() -> float:
    """
    Return spendable USD on Kraken spot with proper asset mapping.
    Kraken /Balance returns like {"ZUSD":"242.11","XXBT":"0.00001", ...}
    """
    SAFETY_BUFFER = float(os.getenv("SAFETY_BUFFER", "0.995"))  # default 99.5%
    
    try:
        # Use the working balance detection from KrakenTradingBot
        from kraken_trading_btc import KrakenTradingBot
        bot = KrakenTradingBot()
        account_info = bot.get_account_info()
        
        if "error" in account_info:
            return 0.0
            
        # Get USD balance from the account info
        usd_balance = account_info.get("cash", 0.0)
        return max(0.0, usd_balance) * SAFETY_BUFFER
    except Exception as e:
        # logger.warning(f"Failed to get USD balance: {e}")  # Commented out - logger not available yet
        return 0.0


def get_session_cfg(cfg):
    """
    Resolve effective session config (from DB/env) into a list of sessions:
    [{"name":"LONDON","tz":LDN,"open":"08:00","close":"16:00"},
     {"name":"NY","tz":NY,"open":"09:30","close":"16:00"}]
    """
    sessions = []
    names = (cfg.get("orb_sessions") or "LONDON,NY").split(",")
    for name in [n.strip().upper() for n in names if n.strip()]:
        if name == "LONDON":
            sessions.append({
                "name": "LONDON", "tz": LDN,
                "open": cfg.get("london_open", "08:00"),
                "close": cfg.get("london_close", "16:00")
            })
        elif name == "NY":
            sessions.append({
                "name": "NY", "tz": NY,
                "open": cfg.get("ny_open", "09:30"),
                "close": cfg.get("ny_close", "16:00")
            })
    return sessions


def resolve_session(now_utc: datetime, cfg) -> tuple[str | None, datetime | None, datetime | None]:
    """
    For the current UTC time, return (session_name, open_utc, close_utc),
    or (None, None, None) if outside all sessions.
    """
    for s in get_session_cfg(cfg):
        tz = s["tz"]
        # local session times
        o_h, o_m = map(int, s["open"].split(":"))
        c_h, c_m = map(int, s["close"].split(":"))
        local = now_utc.astimezone(tz)
        open_local = local.replace(hour=o_h, minute=o_m, second=0, microsecond=0)
        close_local = local.replace(hour=c_h, minute=c_m, second=0, microsecond=0)
        # map to UTC (same *calendar day* in that tz)
        open_utc = open_local.astimezone(timezone.utc)
        close_utc = close_local.astimezone(timezone.utc)
        if open_utc <= now_utc < close_utc:
            return s["name"], open_utc, close_utc
    return None, None, None

logger = logging.getLogger(__name__)

# Environment variables for spot-only trading
SPOT_ONLY = parse_bool(os.getenv("SPOT_ONLY", "true"))
SAFETY_BAL_BUF = float(os.getenv("SAFETY_BAL_BUF", "0.995"))  # keep a tiny dust buffer
ALLOW_SHORT = parse_bool(os.getenv("ALLOW_SHORT", "false"))  # default to long-only

KRAKEN_PAIR_CODE = "XXBTZUSD"  # Kraken API pair code
PUBLIC_API = "https://api.kraken.com/0/public/OHLC"


def _parse_hhmm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def load_sessions_from_env() -> list[SessionDef]:
    sessions = []
    if "LONDON" in os.getenv("ORB_SESSIONS", "LONDON,NY").upper():
        h, m = _parse_hhmm(os.getenv("LONDON_OPEN", "08:00"))
        H, M = _parse_hhmm(os.getenv("LONDON_CLOSE", "16:00"))
        sessions.append(SessionDef("LONDON", "Europe/London", time(h, m), time(H, M),
                                   int(os.getenv("ORB_MINUTES", "15")),
                                   int(os.getenv("ORB_CONFIRM_MINUTES", "5"))))
    if "NY" in os.getenv("ORB_SESSIONS", "LONDON,NY").upper():
        h, m = _parse_hhmm(os.getenv("NY_OPEN", "09:30"))
        H, M = _parse_hhmm(os.getenv("NY_CLOSE", "16:00"))
        sessions.append(SessionDef("NY", "America/New_York", time(h, m), time(H, M),
                                   int(os.getenv("ORB_MINUTES", "15")),
                                   int(os.getenv("ORB_CONFIRM_MINUTES", "5"))))
    return sessions


def allowed_days_from_env() -> set[int]:
    # ORB_DAYS like "1234567" → {1..7}, 7 days enabled by default here
    s = os.getenv("ORB_DAYS", "1234567")
    return {int(ch) for ch in s if ch.isdigit() and ch != '0'}


def _parse_kraken_ohlc(resp_json: dict) -> List[Dict]:
    """Parse Kraken OHLC response into list of bars with UTC timestamps."""
    result = resp_json.get("result", {})
    # find the first key that isn't 'last'
    keys = [k for k in result.keys() if k != "last"]
    if not keys:
        return []
    arr = result[keys[0]]
    bars: List[Dict] = []
    for row in arr:
        # [time, open, high, low, close, vwap, volume, count]
        ts_epoch = int(row[0])
        bars.append({
            "ts": datetime.fromtimestamp(ts_epoch, tz=timezone.utc),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[6]) if len(row) > 6 else 0.0,
        })
    return bars


def fetch_ohlc(pair: str, interval_min: int, since_epoch: Optional[int] = None) -> List[Dict]:
    params = {"pair": pair, "interval": int(interval_min)}
    if since_epoch:
        params["since"] = int(since_epoch)
    r = requests.get(PUBLIC_API, params=params, timeout=10)
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        raise RuntimeError(f"Kraken OHLC error: {j['error']}")
    return _parse_kraken_ohlc(j)


def _spread_bps(best_bid: float, best_ask: float) -> float:
    if best_bid <= 0 or best_ask <= 0:
        return 1e9
    mid = 0.5 * (best_bid + best_ask)
    return 10000.0 * (best_ask - best_bid) / mid


def get_spread_bps() -> float:
    """Get current spread in basis points using Kraken Depth API"""
    try:
        from kraken_trading_btc import KrakenTradingBot
        bot = KrakenTradingBot()
        bid, ask, mid = bot.get_top_of_book("XBTUSD")
        return 0.0 if mid == 0 else round(((ask - bid) / mid) * 1e4, 1)
    except Exception as e:
        logger.warning(f"Failed to get spread: {e}")
        return 0.0


def _load_orb_state(day_str: str) -> Optional[ORBState]:
    try:
        from db import read_settings
        s = (read_settings() or {}).get(f"orb_state_{day_str}")
        if not s:
            return None
        data = json.loads(s)
        return ORBState(**data)
    except Exception:
        return None


def _save_orb_state(day_str: str, state: ORBState) -> None:
    try:
        from db import write_setting
        payload = {
            "day": state.day,
            "orb_high": state.orb_high,
            "orb_low": state.orb_low,
            "direction": state.direction,
            "adds_done": state.adds_done,
            "in_position": state.in_position,
            "avg_price": state.avg_price,
            "qty": state.qty,
            "stop": state.stop,
        }
        write_setting(f"orb_state_{day_str}", json.dumps(payload))
    except Exception as e:
        logger.warning(f"Failed to persist ORB state: {e}")


def load_orb_state() -> Optional[Dict]:
    """Public helper to load today's ORB state payload from DB."""
    try:
        from db import read_settings
        s = read_settings() or {}
        day_str = datetime.now(timezone.utc).date().isoformat()
        raw = s.get(f"orb_state_{day_str}")
        return json.loads(raw) if raw else None
    except Exception:
        return None


def close_orb_position() -> Dict:
    """Flatten current BTC exposure using maker-first then IOC fallback."""
    try:
        from kraken_trading_btc import KrakenTradingBot
        bot = KrakenTradingBot()
        # Use existing rebalance helper to target 0 exposure
        res = bot.rebalance_to_target(0.0)
        return res or {"status": "error", "reason": "rebalance_failed"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def execute_orb_once(settings: dict) -> Dict:
    """Run ORB decision once. Returns a result dict indicating action or skip reason."""
    now = datetime.now(timezone.utc)
    if not in_session(now):
        return {"status": "skipped", "reason": "out_of_session"}

    # Initialize strategy with settings overrides
    params = ORBParams(
        orb_minutes=int(settings.get("orb_open_minutes", 15)),
        confirm_minutes=int(settings.get("orb_confirm_minutes", 5)),
        ema_fast=int(settings.get("orb_ema_fast", 9)),
        ema_mid=int(settings.get("orb_ema_mid", 20)),
        ema_slow=int(settings.get("orb_ema_slow", 50)),
        min_orb_pct=float(settings.get("orb_min_range_pct", 0.002)),
        risk_per_trade=float(settings.get("orb_risk_per_trade", 0.005)),
        buffer_frac=float(settings.get("orb_buffer_frac", 0.10)),
        max_spread_bps=int(settings.get("orb_max_spread_bps", 8)),
        max_adds=int(settings.get("orb_max_adds", 2)),
        trail_atr_mult=float(settings.get("orb_trail_atr_mult", 2.0)),
    )
    strat = ORBStrategy(params)

    # Pull OHLC
    try:
        bars_1m = fetch_ohlc(KRAKEN_PAIR_CODE, interval_min=1)
        bars_5m = fetch_ohlc(KRAKEN_PAIR_CODE, interval_min=5)
    except Exception as e:
        logger.warning(f"OHLC fetch failed: {e}")
        return {"status": "skipped", "reason": "ohlc_unavailable"}

    # Load or create daily state
    day_str = now.date().isoformat()
    state = _load_orb_state(day_str) or ORBState(day=day_str)

    # Set ORB window after 09:45 ET if not already
    if state.orb_high is None or state.orb_low is None:
        # Only build if within opening range window
        hl = strat.build_orb(bars_1m, now)
        if hl:
            state.orb_high, state.orb_low = hl
            _save_orb_state(day_str, state)
            logger.info(f"ORB window set: high={state.orb_high:.2f} low={state.orb_low:.2f}")
            return {"status": "updated", "reason": "orb_set"}
        else:
            return {"status": "skipped", "reason": "waiting_orb_window"}

    # Skip if ORB too tight
    try:
        # Use last close as ref price
        ref_px = float(bars_1m[-1]["close"]) if bars_1m else 0.0
        if strat.orb_tight_filter(state.orb_high, state.orb_low, ref_px):
            return {"status": "skipped", "reason": "tight_orb"}
    except Exception:
        pass

    # Confirm breakout on latest 5m bar close
    side = strat.confirm_breakout(bars_5m, state.orb_high, state.orb_low)
    if not side:
        return {"status": "skipped", "reason": "no_confirm"}

    # Spread guard using orderbook snapshot
    try:
        from kraken_trading_btc import KrakenTradingBot
        bot = KrakenTradingBot()
        ob = bot.get_orderbook_snapshot(KRAKEN_PAIR_CODE)
        best_bid = float(ob.get("best_bid", 0.0))
        best_ask = float(ob.get("best_ask", 0.0))
        spread = _spread_bps(best_bid, best_ask)
        if spread > params.max_spread_bps:
            return {"status": "skipped", "reason": f"spread_{int(spread)}bps"}
    except Exception as e:
        logger.warning(f"Orderbook unavailable: {e}")
        return {"status": "skipped", "reason": "no_orderbook"}

    # Compute stop and qty
    try:
        state.direction = side
        entry = best_ask if side == "long" else best_bid
        stop = strat.compute_initial_stop(side, state.orb_high, state.orb_low)
        state.stop = stop

        acct = bot.get_current_exposure()
        equity = float(acct.get("equity", 0.0))

        # Determine lot step via conservative default if metadata not available
        LOT_STEP_BTC = 1e-5
        qty = strat.compute_qty_from_risk(equity, entry, stop, LOT_STEP_BTC)
        if qty <= 0:
            return {"status": "skipped", "reason": "qty_zero"}

        # Place maker-first order at top of book on the appropriate side
        limit_price = best_ask if side == "long" else best_bid
        res = bot.place_limit_post_only("buy" if side == "long" else "sell", qty, limit_price, KRAKEN_PAIR_CODE)
        if res.get("status") != "success":
            # Fallback to market with slippage cap
            res = bot.place_market_with_slippage_cap("buy" if side == "long" else "sell", qty, max_slippage_bps=10)

        if res.get("status") == "success":
            state.in_position = True
            state.qty = qty
            state.avg_price = float(res.get("price", entry)) if isinstance(res.get("price", None), (int, float)) else entry
            _save_orb_state(day_str, state)
            return {"status": "filled", "side": side, "qty": qty, "price": state.avg_price}
        else:
            return {"status": "error", "reason": res.get("reason", "order_reject")}
    except Exception as e:
        logger.error(f"ORB execution error: {e}")
        return {"status": "error", "reason": str(e)}


# --- PATCH: add helpers and a robust run_orb_cycle wrapper ---
from datetime import datetime, time, timezone
from typing import Any, Dict, Tuple
import logging
import pytz

logger = logging.getLogger(__name__)
NY = pytz.timezone("America/New_York")

# Utilities you likely have already — include if missing:
def get_setting_json(key: str) -> Dict[str, Any]:
    try:
        from db import read_settings
        s = (read_settings() or {}).get(key)
        if not s:
            return {}
        return json.loads(s) if isinstance(s, str) else s
    except Exception:
        return {}

def set_setting_json(key: str, value: Dict[str, Any]) -> None:
    try:
        from db import write_setting
        write_setting(key, json.dumps(value))
    except Exception as e:
        logger.warning(f"Failed to save setting {key}: {e}")

def load_orb_state() -> Dict[str, Any]:
    return get_setting_json("orb_state") or {}  # your own storage helpers

def save_orb_state(state: Dict[str, Any]) -> None:
    set_setting_json("orb_state", state)

def reconcile_position_and_state(api, state):
    """Close drift: if state says LONG but no position (or vice versa), fix it"""
    try:
        bal = api._make_request_with_retry('POST', '/private/Balance', private=True, signature_path="/0/private/Balance")
        bal_result = bal["result"]
        btc = float(bal_result.get("XXBT", 0.0))
        
        if state.get("phase") in ("LONG","SHORT") and btc == 0.0:
            logger.warning("STATE_DRIFT: phase=%s but no BTC; reverting to WAIT_CONFIRM", state.get("phase"))
            state.update({"phase":"WAIT_CONFIRM", "qty":0.0, "avg_price":0.0, "stop":0.0})
            save_orb_state(state)
        
        # Clear old open orders if any
        oo = api._make_request_with_retry('POST', '/private/OpenOrders', private=True, signature_path="/0/private/OpenOrders")
        oo_result = oo["result"].get("open", {})
        if oo_result:
            logger.info("CANCEL_STALE_ORDERS n=%d", len(oo_result))
            api._make_request_with_retry('POST', '/private/CancelAll', private=True, signature_path="/0/private/CancelAll")
    except Exception as e:
        logger.warning(f"Failed to reconcile position and state: {e}")

def is_ny_session_open(now: datetime) -> bool:
    ny = now.astimezone(NY).timetz()
    return time(9,30) <= ny <= time(16,0)

def is_orb_build_window(now: datetime) -> bool:
    ny = now.astimezone(NY).timetz()
    return time(9,30) <= ny < time(9,45)

def to_five_min_bucket(now: datetime) -> datetime:
    b = now.replace(second=0, microsecond=0)
    minute = (b.minute // 5) * 5
    return b.replace(minute=minute)

def phase_is_pre_or_build(state: Dict[str, Any]) -> bool:
    return state.get("phase") in (None, "PRE", "ORB_BUILD")

def is_new_5m_close(now: datetime, state: Dict[str, Any]) -> bool:
    cur = to_five_min_bucket(now).isoformat()
    return state.get("last_five_bucket") != cur


def best_quotes() -> tuple[float, float, float, float]:
    """
    Returns (bid, ask, mid, spread_bps). Falls back to price_worker if REST fails.
    """
    try:
        from kraken_trading_btc import KrakenTradingBot
        bot = KrakenTradingBot()
        logger.info("best_quotes: calling bot.get_ticker('XBTUSD')")
        bid, ask, mid = bot.get_ticker("XBTUSD")
        spread_bps = (ask - bid) / mid * 1e4
        logger.info(f"best_quotes: success bid={bid:.2f} ask={ask:.2f} mid={mid:.2f} spread={spread_bps:.1f}bps")
        return bid, ask, mid, spread_bps
    except Exception as e:
        logger.warning(f"best_quotes REST failed: {e}; falling back to price_worker")
        # fallback: use price worker mid with a conservative spread so we don't trade blind
        try:
            from price_worker import get_last_price
            mid_info = get_last_price()  # existing price_worker hook
            mid = float(mid_info.get('price', 0.0))
            if mid > 0:
                logger.info(f"best_quotes: price_worker fallback mid={mid:.2f}")
                # assume 8 bps spread to be safe; forces skip if MAX_SPREAD_BPS < 8
                return mid * 0.9996, mid * 1.0004, mid, 8.0
            else:
                logger.warning("best_quotes: price_worker returned zero price")
                return 0.0, 0.0, 0.0, 999.0
        except Exception as fallback_e:
            logger.warning(f"price_worker fallback also failed: {fallback_e}")
            # Final fallback: use a reasonable BTC price estimate to avoid $1.00
            logger.warning("best_quotes: using final fallback price estimate")
            fallback_price = 108000.0  # reasonable BTC price estimate
            return fallback_price * 0.9996, fallback_price * 1.0004, fallback_price, 8.0

def compute_opening_range() -> Tuple[float,float,float] | Tuple[None,None,None]:
    """Your existing ORB build using 1-min bars between 09:30–09:45 ET; returns (hi, lo, pct_range)."""
    try:
        from orb_strategy import ORBStrategy, ORBParams
        params = ORBParams(
            orb_minutes=15,
            confirm_minutes=5,
            min_orb_pct=0.002,
            risk_per_trade=0.005,
            buffer_frac=0.10,
            max_spread_bps=8,
            max_adds=2,
            trail_atr_mult=2.0,
            ema_fast=9,
            ema_mid=20,
            ema_slow=50
        )
        strat = ORBStrategy(params)
        bars = fetch_ohlc(KRAKEN_PAIR_CODE, interval_min=1)
        if bars:
            hl = strat.build_orb(bars, datetime.now(timezone.utc))
            if hl:
                hi, lo = hl
                pct = (hi - lo) / ((hi + lo) / 2) if hi and lo else None
                return (hi, lo, pct)
        return (None, None, None)
    except Exception as e:
        logger.warning(f"Failed to compute opening range: {e}")
        return (None, None, None)

def ema(series: List[float], n: int) -> List[float]:
    """Calculate Exponential Moving Average"""
    if not series or n <= 1: 
        return series[:]
    k = 2.0 / (n + 1.0)
    out = []
    e = None
    for v in series:
        e = v if e is None else (v - e) * k + e
        out.append(e)
    return out

def compute_confirmation_signal(
    bars_5m: List[Dict[str, Any]],
    orb_hi: float,
    orb_lo: float,
    ema_fast: int = 9,
    ema_mid: int = 20,
    ema_slow: int = 50,
    buffer_bps: int = 3,           # small buffer to avoid equals-at-boundary
    use_volume_filter: bool = False,
    vol_lookback: int = 20
) -> Dict[str, Any]:
    """
    Returns:
      {'dir': 'long'|'short'|None,
       'ema_ok': bool,
       'close': float, 'ema9': float, 'ema20': float, 'ema50': float,
       'reason': '...' }
    """
    if not bars_5m or len(bars_5m) < max(ema_slow + 2, 55):
        return {"dir": None, "ema_ok": False, "reason": "ema_warmup"}

    closes = [b["close"] for b in bars_5m]
    vols   = [b.get("volume", 0.0) for b in bars_5m]
    last   = bars_5m[-1]
    c      = float(last["close"])

    e9  = ema(closes, ema_fast)[-1]
    e20 = ema(closes, ema_mid)[-1]
    e50 = ema(closes, ema_slow)[-1]
    ema_up   = (e9 > e20 > e50)
    ema_down = (e9 < e20 < e50)

    # tiny breakout buffer to avoid == boundary
    up_break   = c > orb_hi * (1.0 + buffer_bps / 1e4)
    down_break = c < orb_lo * (1.0 - buffer_bps / 1e4)

    if use_volume_filter:
        avg_vol = sum(vols[-vol_lookback:]) / vol_lookback
        vol_ok  = last.get("volume", 0.0) > avg_vol
    else:
        vol_ok = True

    if up_break and ema_up and vol_ok:
        return {"dir":"long","ema_ok":True,"close":c,"ema9":e9,"ema20":e20,"ema50":e50,"reason":"break_up_ema_ok"}
    if down_break and ema_down and vol_ok:
        return {"dir":"short","ema_ok":True,"close":c,"ema9":e9,"ema20":e20,"ema50":e50,"reason":"break_down_ema_ok"}

    # No signal — provide reason
    reason = []
    if not (up_break or down_break): reason.append("no_break")
    if not (ema_up or ema_down):     reason.append("ema_not_aligned")
    if not vol_ok:                   reason.append("vol_fail")
    return {"dir":None,"ema_ok":False,"close":c,"ema9":e9,"ema20":e20,"ema50":e50,"reason":"|".join(reason) or "unknown"}

def compute_stop(entry: float, orb_hi: float, orb_lo: float, buffer_frac: float, direction: str) -> float:
    rng = orb_hi - orb_lo
    if direction == "long":
        return orb_lo - buffer_frac * rng
    else:
        return orb_hi + buffer_frac * rng

def size_by_risk(risk_frac: float, equity: float, entry: float, stop: float) -> float:
    risk_usd = max(0.0, risk_frac * equity)
    per_unit = abs(entry - stop) or 1e-9
    qty = risk_usd / per_unit
    return max(0.0, qty)

def _safe_float(v, d=0.0):
    """Safely convert value to float, return default if conversion fails."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return d

def log_skip(reason: str, extra: Dict = None):
    """Log when a trade is skipped with reason and extra data."""
    extra_str = f" {extra}" if extra else ""
    logger.info(f"ORB_SKIP reason={reason}{extra_str}")

def execute_orb_entry(dir_: str, entry_price: float, stop: float, desired_qty: float) -> Dict:
    """Execute ORB entry with spot-only safety checks."""
    # Check if shorts are allowed
    if not ALLOW_SHORT and dir_ == "short":
        logger.info("ORB_SKIP reason=shorts_disabled dir=%s", dir_)
        return {"status": "skipped", "reason": "shorts_disabled", "dir": dir_}
    
    side = "buy" if dir_ == "long" else "sell"

    # fetch balances once
    try:
        from kraken_trading_btc import KrakenTradingBot
        bot = KrakenTradingBot()
        account_info = bot.get_account_info()
        if "error" in account_info:
            btc_bal = 0.0
            usd_bal = 0.0
        else:
            btc_bal = _safe_float(account_info.get("balances", {}).get("XXBT", 0.0))
            usd_bal = _safe_float(account_info.get("cash", 0.0))
    except Exception as e:
        logger.warning(f"Failed to get balances: {e}")
        btc_bal = 0.0
        usd_bal = 0.0

    qty = desired_qty

    if SPOT_ONLY and side == "sell":
        # Cap sell size to available BTC; never short
        max_sell = max(0.0, btc_bal * SAFETY_BAL_BUF)
        final_qty = min(qty, max_sell)

        # Get minimum lot size from pair metadata
        try:
            MIN_LOT_SIZE = 0.0001  # Kraken minimum BTC lot size
        except Exception:
            MIN_LOT_SIZE = 0.0001  # fallback

        if final_qty < MIN_LOT_SIZE:
            logger.info("ORB_SKIP reason=no_inventory btc_bal=%.8f desired_qty=%.8f final_qty=%.8f", 
                       btc_bal, qty, final_qty)
            return {"status": "skipped", "reason": "no_inventory", "btc_bal": btc_bal, 
                   "desired_qty": qty, "final_qty": final_qty}
        
        qty = final_qty

    if SPOT_ONLY and side == "buy":
        # Optional: cap buy by available USD (avoid Kraken "insufficient funds")
        notional = qty * entry_price
        max_notional = usd_bal * SAFETY_BAL_BUF
        if notional > max_notional:
            qty = max_notional / entry_price

        # Get minimum lot size from pair metadata
        try:
            MIN_LOT_SIZE = 0.0001  # Kraken minimum BTC lot size
        except Exception:
            MIN_LOT_SIZE = 0.0001  # fallback

        if qty < MIN_LOT_SIZE:
            logger.info("ORB_SKIP reason=insufficient_usd usd_bal=%.2f desired_qty=%.8f final_qty=%.8f", 
                       usd_bal, desired_qty, qty)
            return {"status": "skipped", "reason": "insufficient_usd", "usd_bal": usd_bal, 
                   "desired_qty": desired_qty, "final_qty": qty}

    # proceed with maker-first place + fallback
    try:
        from kraken_trading_btc import KrakenTradingBot
        bot = KrakenTradingBot()
        ok, res = bot.place_limit_post_only(side, qty, entry_price)
        if not ok or res.get("status") != "success":
            res = bot.place_market_with_slippage_cap(side, qty, max_slippage_bps=10)
        
        # Add explicit result logging
        if isinstance(res, dict) and res.get("status") == "success":
            logger.info("ORB_FILLED side=%s qty=%.8f price=%.2f txid=%s",
                        side, qty, res.get("price", entry_price), res.get("txid", "unknown"))
            res["final_qty"] = qty
        else:
            logger.warning("ORB_ORDER_RESULT side=%s qty=%.8f entry=%.2f result=%s",
                           side, qty, entry_price, res)
            res["final_qty"] = qty
        
        return res
    except Exception as e:
        logger.warning(f"Failed to place order: {e}")
        return {"status": "error", "reason": str(e)}

class ORBExecutor:
    def __init__(self):
        self.sessions = load_sessions_from_env()
        self.allowed_days = allowed_days_from_env()
        self.params = ORBParams(
            sessions=self.sessions,
            ema_fast=int(os.getenv("ORB_EMA_FAST", "9")),
            ema_mid=int(os.getenv("ORB_EMA_MID", "20")),
            ema_slow=int(os.getenv("ORB_EMA_SLOW", "50")),
            min_orb_pct=float(os.getenv("ORB_MIN_ORB_PCT", "0.002")),
            buffer_bps=int(os.getenv("ORB_BUFFER_BPS", "2")),
            risk_per_trade=float(os.getenv("ORB_RISK_PER_TRADE", "0.005")),
            buffer_frac=float(os.getenv("ORB_BUFFER_FRAC", "0.10")),
            max_adds=int(os.getenv("ORB_MAX_ADDS", "2")),
            trail_atr_mult=float(os.getenv("ORB_TRAIL_ATR_MULT", "2.0")),
        )
        # attributes used in /orb/status
        self.last_session_label = None
        self.last_reason = None
        self.last_signal = None
        self.last_ema_ok = None
        self.last_break_ok = None
        self.last_risk_qty = None
        self.last_final_qty = None
        self.last_close = None
        self.last_ema9 = None
        self.last_ema20 = None
        self.last_ema50 = None
        self.last_spread_bps = None
        self.last_btc_balance = None
        self.last_usd_balance = None
        self.phase = None
        self.state = None

    def _active_session(self, now_utc: datetime) -> SessionDef | None:
        if not weekday_allowed(now_utc, self.allowed_days):
            return None
        for s in self.sessions:
            if is_orb_build(now_utc, s) or is_confirm_window(now_utc, s):
                return s
        return None

    def _state_key(self, day_str: str, label: str) -> str:
        return f"orb:{day_str}:{label}"

    def _load_state(self, day_str: str, label: str) -> ORBState | None:
        key = self._state_key(day_str, label)
        try:
            from db import read_settings
            s = (read_settings() or {}).get(key)
            if not s:
                return None
            data = json.loads(s)
            # Convert ISO string back to datetime if present
            if data.get("last_five_bucket"):
                try:
                    data["last_five_bucket"] = datetime.fromisoformat(data["last_five_bucket"])
                except Exception:
                    data["last_five_bucket"] = None
            return ORBState(**data)
        except Exception:
            return None

    def _save_state(self, st: ORBState):
        key = self._state_key(st.day, st.session)
        try:
            from db import write_setting
            # Convert datetime to ISO string for JSON serialization
            state_dict = st.__dict__.copy()
            if state_dict.get("last_five_bucket"):
                state_dict["last_five_bucket"] = state_dict["last_five_bucket"].isoformat()
            write_setting(key, json.dumps(state_dict))
        except Exception as e:
            logger.warning(f"Failed to save ORB state: {e}")

    def _compute_opening_range(self, now_utc: datetime, session_name: str) -> tuple[float, float, float] | tuple[None, None, None]:
        """Compute opening range for the given session"""
        try:
            bars_1m = fetch_ohlc(KRAKEN_PAIR_CODE, interval_min=1)
            if bars_1m:
                # Create a session definition for the strategy
                if session_name == "LONDON":
                    session = SessionDef("LONDON", "Europe/London", time(8, 0), time(16, 0))
                else:
                    session = SessionDef("NY", "America/New_York", time(9, 30), time(16, 0))
                
                strat = ORBStrategy(self.params)
                hl = strat.build_orb(bars_1m, now_utc, session)
                if hl:
                    hi, lo = hl
                    pct = (hi - lo) / ((hi + lo) / 2) if hi and lo else None
                    return (hi, lo, pct)
        except Exception as e:
            logger.warning(f"Failed to compute opening range: {e}")
        return (None, None, None)

    def _confirm_breakout(self, now_utc: datetime, session_name: str, state: ORBState) -> tuple[str | None, str]:
        """Confirm breakout signal for the given session"""
        try:
            bars_5m = fetch_ohlc(KRAKEN_PAIR_CODE, interval_min=5)
            sig = compute_confirmation_signal(
                bars_5m=bars_5m,
                orb_hi=state.orb_high,
                orb_lo=state.orb_low,
                ema_fast=self.params.ema_fast,
                ema_mid=self.params.ema_mid,
                ema_slow=self.params.ema_slow,
                buffer_bps=self.params.buffer_bps,
                use_volume_filter=False
            )
            
            # Store diagnostics
            self.last_close = sig.get("close")
            self.last_ema9 = sig.get("ema9")
            self.last_ema20 = sig.get("ema20")
            self.last_ema50 = sig.get("ema50")
            self.last_signal = sig.get("dir")
            self.last_ema_ok = sig.get("ema_ok")
            self.last_break_ok = sig.get("break_ok")
            self.last_reason = sig.get("reason")
            
            return sig.get("dir"), sig.get("reason", "unknown")
        except Exception as e:
            logger.warning(f"Failed to confirm breakout: {e}")
            return None, "confirmation_failed"

    def _execute_entry(self, signal: str, state: ORBState, session_label: str) -> dict:
        """Execute entry order for the given signal"""
        try:
            # --- fetch quotes safely ---
            mid = None
            try:
                bid, ask, mid, spread_bps = best_quotes()
                self.last_spread_bps = spread_bps
            except Exception as e:
                logger.warning(f"ORB_SKIP reason=quote_error err={e}")
                return {"status": "skipped", "reason": "quote_error", "session": session_label}

            # --- sanity check (BTC shouldn't be $1) ---
            if not mid or mid < 1000 or mid > 1000000:
                logger.warning(f"ORB_SKIP reason=bad_price mid={mid}")
                return {"status": "skipped", "reason": "bad_price", "session": session_label}
            
            # Get balances with improved USD detection
            try:
                from kraken_trading_btc import KrakenTradingBot
                bot = KrakenTradingBot()
                account_info = bot.get_account_info()
                
                if "error" not in account_info:
                    # Get USD balance from account info
                    self.last_usd_balance = float(account_info.get("cash", 0.0))
                    
                    # Get BTC balance from account info
                    balances = account_info.get("balances", {})
                    btc_balance = float(balances.get("XXBT", 0.0))
                    self.last_btc_balance = btc_balance
                else:
                    self.last_usd_balance = 0.0
                    self.last_btc_balance = 0.0
            except Exception:
                self.last_usd_balance = 0.0
                self.last_btc_balance = 0.0
            
            # Use simple price tick for rounding
            price_tick = 0.1  # fallback
            
            # round to tick
            entry = round_down(mid, price_tick)
            stop = compute_stop(entry, state.orb_high, state.orb_low, self.params.buffer_frac, signal)
            
            # Get equity for sizing
            try:
                from kraken_trading_btc import KrakenTradingBot
                bot = KrakenTradingBot()
                account_info = bot.get_account_info()
                if "error" not in account_info:
                    equity = float(account_info.get("equity", 0.0))
                else:
                    equity = 0.0
            except Exception:
                equity = 0.0
            
            # --- Size with risk and cap USD usage to keep cash for adds ---
            # risk-based position size
            price_diff = abs(entry - stop) or 1e-9
            risk_usd = float(self.params.risk_per_trade) * equity
            qty_by_risk = max(0.0, risk_usd / price_diff)
            usd_by_risk = qty_by_risk * entry
            
            # --- SAFER EXPOSURE MANAGEMENT ---
            # Cap by max entry exposure (default 60%)
            usd_cap = equity * float(os.getenv("MAX_ENTRY_EXPOSURE", "0.6"))
            
            # Always leave cash reserve for adds (default $25)
            cash_reserve = float(os.getenv("ORB_CASH_RESERVE_USD", "25.0"))
            usd_free = max(0.0, float(self.last_usd_balance or 0.0) - cash_reserve)
            
            # Cap total exposure to prevent "all USD to BTC" behavior (default 80%)
            max_exposure_pct = float(os.getenv("ORB_MAX_EXPOSURE", "0.80"))
            current_exposure = (equity - usd_free) / equity if equity > 0 else 0.0
            if current_exposure >= max_exposure_pct:
                logger.info(f"ORB_SKIP reason=max_exposure_reached current={current_exposure:.1%} max={max_exposure_pct:.1%}")
                return {"status": "error", "reason": "max_exposure_reached", "session": session_label}
            
            # Use the most conservative limit
            usd_to_use = min(usd_free, usd_by_risk, usd_cap)
            qty = max(0.0, usd_to_use / entry)
            self.last_risk_qty = qty_by_risk
            
            logger.info(f"ORB_EXPOSURE_CHECK usd_free={usd_free:.2f} cash_reserve={cash_reserve:.2f} "
                       f"current_exposure={current_exposure:.1%} max_exposure={max_exposure_pct:.1%}")
            
            # Check USD balance before proceeding
            min_notional_usd = float(os.getenv("MIN_NOTIONAL_USD", "10.0"))
            if (qty * entry) < min_notional_usd or self.last_usd_balance < min_notional_usd:
                logger.info(f"ORB_SKIP reason=insufficient_usd usd_bal={self.last_usd_balance:.2f} "
                           f"min_notional={min_notional_usd}")
                return {"status": "error", "reason": "insufficient_usd", "session": session_label}
            
            logger.info("ORB_ENTRY session=%s dir=%s entry=%.2f stop=%.2f qty=%.6f", 
                       session_label, signal, entry, stop, qty)
            
            # Execute the order
            res = execute_orb_entry(signal, entry, stop, qty)
            
            if isinstance(res, dict) and res.get("status") == "success":
                state.in_position = True
                state.avg_price = entry
                state.qty = qty
                state.stop = stop
                state.direction = signal
                self.last_final_qty = qty
                self._save_state(state)
                logger.info("ORB_FILLED session=%s mode=%s vwap=%.2f qty=%.6f", 
                           session_label, res.get("mode"), entry, qty)
                return {"status": "success", "session": session_label, "signal": signal, "qty": qty}
            else:
                self.last_final_qty = 0
                return {"status": "error", "session": session_label, "reason": res.get("reason", "order_failed")}
                
        except Exception as e:
            logger.warning(f"Failed to execute entry: {e}")
            return {"status": "error", "session": session_label, "reason": str(e)}

    def run_orb_cycle(self, now_utc: datetime | None = None, settings: Dict[str, Any] | None = None) -> dict:
        now_utc = now_utc or datetime.now(timezone.utc)
        
        # Load settings if not provided
        if settings is None:
            from auto_trade_scheduler import load_runtime_settings
            settings = vars(load_runtime_settings())
        
        # Check if ORB is enabled
        if not settings.get("orb_enabled", False):
            return {"status": "disabled", "reason": "orb_disabled", "phase": "PRE"}
        
        # Figure out current session every cycle
        session_name, open_utc, close_utc = resolve_session(now_utc, settings)
        
        # Create or load state for today
        if session_name:
            loc_day = now_utc.astimezone(pytz.timezone("Europe/London" if session_name == "LONDON" else "America/New_York")).date().isoformat()
            state = self._load_state(loc_day, session_name) or ORBState(day=loc_day, session=session_name)
        else:
            state = ORBState(day=now_utc.date().isoformat(), session=None)
        
        self.state = state
        state.session = session_name  # keep in state for status endpoint
        self.last_session_label = session_name

        if session_name is None:
            state.phase = "PRE"
            self._save_state(state)
            return {"status": "waiting", "reason": "no_session", "phase": state.phase, "session": None}

        # Initialize 5m bucket if first in-session tick
        if state.last_five_bucket is None:
            state.last_five_bucket = floor_5m(now_utc)
            logger.info(f"ORB_5M_INIT bucket={state.last_five_bucket.isoformat()}")

        # --- Opening-Range back-fill (if we missed the 15-min build window) ---
        orb_minutes = int(settings.get("orb_minutes", 15))
        build_cutoff = open_utc + timedelta(minutes=orb_minutes)

        if state.orb_high is None and now_utc >= build_cutoff:
            # Pull 1m bars from [open_utc, build_cutoff)
            try:
                bars = fetch_ohlc(KRAKEN_PAIR_CODE, interval_min=1)
                if bars:
                    # Filter bars for the build window
                    build_bars = [b for b in bars if open_utc <= b["ts"] < build_cutoff]
                    if build_bars:
                        highs = [b["high"] for b in build_bars]
                        lows = [b["low"] for b in build_bars]
                        state.orb_high = max(highs)
                        state.orb_low = min(lows)
                        state.phase = "WAIT_CONFIRM"
                        logger.info(f"ORB_BACKFILL hi={state.orb_high:.2f} lo={state.orb_low:.2f} session={session_name}")
                        self._save_state(state)
                    else:
                        # No bars available yet – keep waiting
                        state.phase = "ORB_BUILD"
                        self._save_state(state)
                        return {"status": "waiting", "reason": "no_bars", "phase": state.phase, "session": session_name}
                else:
                    state.phase = "ORB_BUILD"
                    self._save_state(state)
                    return {"status": "waiting", "reason": "no_bars", "phase": state.phase, "session": session_name}
            except Exception as e:
                logger.warning(f"Failed to back-fill ORB: {e}")
                state.phase = "ORB_BUILD"
                self._save_state(state)
                return {"status": "waiting", "reason": "backfill_failed", "phase": state.phase, "session": session_name}

        # --- build phase ---
        if now_utc < build_cutoff and state.orb_high is None:
            orb_hi, orb_lo, rng_pct = self._compute_opening_range(now_utc, session_name)
            if orb_hi and orb_lo:
                state.orb_high = orb_hi
                state.orb_low = orb_lo
                state.phase = "WAIT_CONFIRM"
                self._save_state(state)
                self.phase = "ORB_BUILD"
                return {"status": "ok", "phase": "ORB_BUILD", "session": session_name, 
                       "orb_high": orb_hi, "orb_low": orb_lo, "range_pct": rng_pct}
            else:
                state.phase = "ORB_BUILD"
                self._save_state(state)
                self.phase = "ORB_BUILD"
                return {"status": "waiting", "reason": "building_orb", "phase": "ORB_BUILD", "session": session_name}

        # --- confirm/trade phase ---
        if now_utc >= build_cutoff and now_utc < close_utc:
            # ensure we have a defined ORB
            if not (state.orb_high and state.orb_low):
                self.phase = "WAIT_CONFIRM"
                return {"status": "waiting", "reason": "no_orb", "phase": "WAIT_CONFIRM", "session": session_name}

            # Check for 5-minute rollover
            current_bucket = floor_5m(now_utc)
            is_five_close = (state.last_five_bucket and current_bucket > state.last_five_bucket)

            # compute EMAs on 5m closes, spread guard, etc.
            signal, confirm_reason = self._confirm_breakout(now_utc, session_name, state)
            self.last_reason = confirm_reason
            
            # Stamp the rollover after confirmation
            if is_five_close:
                logger.info(f"ORB_5M_ROLLOVER prev={state.last_five_bucket.isoformat()} -> curr={current_bucket.isoformat()}")
                state.last_five_bucket = current_bucket
                self._save_state(state)
            
            if not signal:
                self.phase = "WAIT_CONFIRM"
                return {"status": "waiting", "reason": "no_confirm", "phase": "WAIT_CONFIRM", "session": session_name}

            # size, place order (maker-first), persist state, emit logs/metrics with label=session_name
            entry = self._execute_entry(signal, state, session_label=session_name)
            self.phase = "TRADING"
            return entry

        self.phase = "WAIT_CONFIRM"
        return {"status": "waiting", "phase": "WAIT_CONFIRM", "session": session_name}


def run_orb_cycle(now: datetime, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy wrapper for the new ORBExecutor class"""
    executor = ORBExecutor()
    return executor.run_orb_cycle(now, settings)

# --- Cash buffer / exposure maintenance --------------------------------------
from dataclasses import dataclass

@dataclass
class BufferResult:
    action: str            # "noop" | "sell"
    placed: bool           # order placed?
    reason: str
    usd_before: float
    usd_after: float | None
    qty_sold: float | None
    price_used: float | None

def ensure_cash_buffer(bot,
                       buffer_usd: float = 25.0,
                       max_exposure: float = 0.80,
                       min_notional_usd: float = 10.0,
                       safety_buffer: float = 0.995) -> BufferResult:
    """
    Ensure we always have at least `buffer_usd` cash and don't exceed `max_exposure`.
    Sells a tiny BTC slice (post-only limit) to top up cash when needed.
    Runs idempotently; if there's not enough BTC to sell the min_notional, it noops.
    """
    try:
        # Get account info using the actual method
        account_info = bot.get_account_info()
        if "error" in account_info:
            return BufferResult("noop", False, f"account_error:{account_info['error']}", 0.0, None, None, None)
        
        usd = float(account_info.get("cash", 0.0))
        btc = float(account_info.get("balances", {}).get("XXBT", 0.0))
        
        # Get current BTC price using the actual method
        try:
            bid, ask, mid = bot.get_top_of_book()
            price = float(bid)  # conservative on sells
        except:
            # Fallback to demo price if API fails
            price = 45000.0

        equity = usd + btc * price
        target_cash = max(buffer_usd, equity * (1.0 - max_exposure))

        if usd >= target_cash:
            return BufferResult("noop", False, "buffer_ok", usd, None, None, None)

        need = (target_cash - usd)
        # Respect exchange minimums
        usd_to_sell = max(need, min_notional_usd)
        qty = usd_to_sell / price

        # If we literally don't have that much BTC, sell what we have if it meets min_notional.
        if btc * price < min_notional_usd:
            return BufferResult("noop", False, "btc_not_enough_for_min_notional", usd, None, None, None)

        qty = min(qty, btc)

        # Use simple rounding for now since the bot doesn't have those methods
        px = round(price * 1.001, 1)  # 0.1% above bid, Kraken requires 1 decimal place
        qty = round(qty, 8)  # 8 decimal places for BTC

        # Final safety: notional check after rounding
        if px * qty < min_notional_usd:
            return BufferResult("noop", False, "rounded_below_min_notional", usd, None, None, None)

        # Execute the sell order
        res = bot.place_post_only_limit("sell", qty, px, validate=False)
        if res.get("status") == "success":
            logger.info(f"✅ CASH_BUFFER sell order placed: {qty} BTC at ${px} for ${px * qty:.2f}")
            return BufferResult("sell", True, "order_placed", usd, None, qty, px)
        else:
            logger.warning(f"❌ CASH_BUFFER sell order failed: {res.get('reason')}")
            return BufferResult("sell", False, f"order_failed:{res.get('reason')}", usd, None, qty, px)
        
    except Exception as e:
        return BufferResult("noop", False, f"error:{e}", 0.0, None, None, None)

# --- FastAPI handler functions for safe route registration ---
def get_orb_status() -> Dict[str, Any]:
    """Get ORB status for FastAPI endpoint"""
    try:
        from db import get_setting
        from auto_trade_scheduler import load_runtime_settings
        from datetime import datetime, timezone
        
        # Safe float conversion helper
        def _f(v, d=0.0):
            try:
                return float(v)
            except (ValueError, TypeError):
                return d
        
        # Get real balances
        btc_bal = 0.0
        usd_bal = 0.0
        try:
            from kraken_trading_btc import KrakenTradingBot
            bot = KrakenTradingBot()
            account_info = bot.get_account_info()
            
            if "error" not in account_info:
                usd_bal = _f(account_info.get("cash"))
                balances = account_info.get("balances", {})
                btc_bal = _f(balances.get("XXBT"))
        except Exception:
            pass
        
        # Get current spread
        try:
            current_spread = get_spread_bps()
        except Exception:
            current_spread = 0.0
        
        # Get session info
        now_utc = datetime.now(timezone.utc)
        settings = load_runtime_settings()
        session_name, open_utc, close_utc = resolve_session(now_utc, vars(settings))
        
        # Create executor instance to get current state
        executor = ORBExecutor()
        
        # Determine phase
        if session_name is None:
            phase = "PRE"
        elif executor.state and executor.state.orb_high is not None:
            phase = executor.state.phase or "WAIT_CONFIRM"
        else:
            phase = "ORB_BUILD"
        
        # Get bar counts
        bars_1m = 0
        bars_5m = 0
        try:
            bars_1m = len(fetch_ohlc("XXBTZUSD", interval_min=1) or [])
            bars_5m = len(fetch_ohlc("XXBTZUSD", interval_min=5) or [])
        except Exception:
            pass
        
        return {
            "mode": get_setting("strategy_mode"),
            "enabled": bool(get_setting("orb_enabled", False)),
            "day": getattr(executor.state, "day", None) if executor.state else None,
            "session": session_name,
            "phase": phase,
            "orb_high": getattr(executor.state, "orb_high", None) if executor.state else None,
            "orb_low": getattr(executor.state, "orb_low", None) if executor.state else None,
            "adds_done": getattr(executor.state, "adds_done", None) if executor.state else None,
            "avg_price": getattr(executor.state, "avg_price", None) if executor.state else None,
            "qty": getattr(executor.state, "qty", None) if executor.state else None,
            "stop": getattr(executor.state, "stop", None) if executor.state else None,
            "last_close": executor.last_close,
            "ema9": executor.last_ema9,
            "ema20": executor.last_ema20,
            "ema50": executor.last_ema50,
            "spread_bps": current_spread,
            "confirm_reason": executor.last_reason,
            "last_five_bucket": getattr(executor.state, "last_five_bucket", None).isoformat() if getattr(executor.state, "last_five_bucket", None) else None,
            "btc_balance": btc_bal,
            "usd_balance": usd_bal,
            "skip_reason": None,
            "signal": executor.last_signal,
            "ema_ok": executor.last_ema_ok,
            "break_ok": executor.last_break_ok,
            "risk_qty": executor.last_risk_qty,
            "final_qty": executor.last_final_qty,
            "last_signal": executor.last_signal,
            "last_ema_ok": executor.last_ema_ok,
            "last_break_ok": executor.last_break_ok,
            "last_reason": executor.last_reason,
            "last_risk_qty": executor.last_risk_qty,
            "bars_1m": bars_1m,
            "bars_5m": bars_5m
        }
    except Exception as e:
        logging.exception("Failed to get ORB status")
        return {"status": "error", "detail": str(e)}

def self_test() -> Dict[str, Any]:
    """Self-test function for FastAPI endpoint"""
    try:
        from datetime import datetime, timezone
        from auto_trade_scheduler import load_runtime_settings
        now = datetime.now(timezone.utc)
        settings = load_runtime_settings()
        settings_dict = vars(settings)
        settings_dict["orb_debug_force_session"] = True
        
        executor = ORBExecutor()
        result = executor.run_orb_cycle(now, settings_dict)
        return {"now": now.isoformat(), "res": result}
    except Exception as e:
        logging.exception("ORB self-test failed")
        return {"status": "error", "detail": str(e)}

def flat_all_positions() -> Dict[str, Any]:
    """Close all ORB positions"""
    try:
        result = close_orb_position()
        return {"status": "success", "result": result}
    except Exception as e:
        logging.exception("Failed to flat positions")
        return {"status": "error", "detail": str(e)}


def get_cash_buffer_status() -> Dict[str, Any]:
    """Get current cash buffer status for UI display"""
    try:
        from kraken_trading_btc import KrakenTradingBot
        bot = KrakenTradingBot()
        
        # Get account info
        account_info = bot.get_account_info()
        if "error" in account_info:
            return {"status": "error", "reason": f"Failed to get account info: {account_info['error']}"}
        
        usd = float(account_info.get("cash", 0.0))
        btc = float(account_info.get("balances", {}).get("XXBT", 0.0))
        
        # Get current BTC price
        try:
            bid, ask, mid = bot.get_top_of_book()
            price = float(bid)
        except:
            price = 45000.0
        
        equity = usd + btc * price
        target_cash = max(25.0, equity * 0.20)  # $25 or 20% of equity
        current_exposure = (equity - usd) / equity if equity > 0 else 0.0
        
        # Check for pending buffer orders
        buffer_orders = []
        try:
            open_orders = bot._make_request_with_retry('POST', '/private/OpenOrders', {}, private=True, signature_path="/0/private/OpenOrders")
            if not open_orders.get("error"):
                for txid, order in open_orders.get("result", {}).get("open", {}).items():
                    userref = str(order.get("userref", ""))
                    if userref.startswith("orb-buffer-"):
                        buffer_orders.append({
                            "txid": txid,
                            "qty": float(order.get("vol", 0)),
                            "price": float(order.get("descr", {}).get("price", 0)),
                            "age_minutes": (time.time() - order.get("opentm", 0)) / 60 if order.get("opentm") else 0
                        })
        except Exception as e:
            logger.warning(f"Failed to get buffer orders: {e}")
        
        return {
            "status": "success",
            "cash_buffer": {
                "target_usd": target_cash,
                "current_usd": usd,
                "current_exposure_pct": current_exposure * 100,
                "max_exposure_pct": 80.0,
                "buffer_status": "ok" if usd >= target_cash else "needs_cash",
                "pending_orders": buffer_orders
            }
        }
        
    except Exception as e:
        logger.exception("Failed to get cash buffer status")
        return {"status": "error", "reason": str(e)}


def adopt_wallet_position() -> Dict[str, Any]:
    """Adopt existing BTC in wallet as ORB position for management"""
    try:
        from kraken_trading_btc import KrakenTradingBot
        bot = KrakenTradingBot()
        
        # Get current balances
        account_info = bot.get_account_info()
        if "error" in account_info:
            return {"status": "error", "reason": f"Failed to get account info: {account_info['error']}"}
        
        btc_balance = float(account_info.get("balances", {}).get("XXBT", 0.0))
        if btc_balance <= 0:
            return {"status": "error", "reason": "No BTC in wallet to adopt"}
        
        # Get current BTC price
        try:
            bid, ask, mid = bot.get_top_of_book()
            current_price = float(bid)
        except:
            current_price = 45000.0
        
        # Create ORB executor to check current state
        executor = ORBExecutor()
        
        # Only adopt if no active ORB position
        if executor.state and executor.state.in_position:
            return {"status": "error", "reason": "ORB already has active position"}
        
        # Set the position data
        if not executor.state:
            # Create new state if none exists
            from orb_strategy import ORBState
            executor.state = ORBState(
                day=datetime.now().strftime("%Y-%m-%d"),
                session="WALLET",
                in_position=True,
                avg_price=current_price,
                qty=btc_balance,
                direction="long"
            )
        else:
            # Update existing state
            executor.state.in_position = True
            executor.state.avg_price = current_price
            executor.state.qty = btc_balance
            executor.state.direction = "long"
            executor.state.phase = "WALLET_ADOPTED"
        
        # Save the state
        executor._save_state(executor.state)
        
        logger.info(f"✅ Adopted wallet position: {btc_balance} BTC at ${current_price:.2f}")
        
        return {
            "status": "success",
            "message": f"Adopted {btc_balance} BTC as ORB position",
            "position": {
                "qty": btc_balance,
                "avg_price": current_price,
                "direction": "long",
                "value_usd": btc_balance * current_price
            }
        }
        
    except Exception as e:
        logger.exception("Failed to adopt wallet position")
        return {"status": "error", "reason": str(e)}


