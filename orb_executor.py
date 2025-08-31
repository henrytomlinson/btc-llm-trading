#!/usr/bin/env python3
"""
ORB executor: fetches OHLC from Kraken public API, applies ORBStrategy,
and places orders via existing KrakenTradingBot with maker-first preference.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, time
from typing import List, Dict, Optional, Any, Tuple

import requests

from orb_strategy import ORBStrategy, ORBParams, ORBState, in_session
from db import get_setting
from utils_bool import parse_bool

logger = logging.getLogger(__name__)

# Environment variables for spot-only trading
SPOT_ONLY = parse_bool(os.getenv("SPOT_ONLY", "true"))
SAFETY_BAL_BUF = float(os.getenv("SAFETY_BAL_BUF", "0.995"))  # keep a tiny dust buffer
ALLOW_SHORT = parse_bool(os.getenv("ALLOW_SHORT", "false"))  # default to long-only

KRAKEN_PAIR_CODE = "XXBTZUSD"  # Kraken API pair code
PUBLIC_API = "https://api.kraken.com/0/public/OHLC"


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
        bid, ask, mid = bot.get_ticker("XBTUSD")
        spread_bps = (ask - bid) / mid * 1e4
        return bid, ask, mid, spread_bps
    except Exception as e:
        logger.warning(f"best_quotes REST failed: {e}; falling back to price_worker")
        # fallback: use price worker mid with a conservative spread so we don't trade blind
        try:
            from price_worker import get_cached_price
            mid_info = get_cached_price()  # existing price_worker hook
            mid = float(mid_info.price)
            # assume 8 bps spread to be safe; forces skip if MAX_SPREAD_BPS < 8
            return mid * 0.9996, mid * 1.0004, mid, 8.0
        except Exception as fallback_e:
            logger.warning(f"price_worker fallback also failed: {fallback_e}")
            return 0.0, 0.0, 0.0, 999.0

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
        from kraken_trading_btc import get_balances_normalized
        bals = get_balances_normalized()  # returns {"btc": float, "usd": float, ...}
        btc_bal = _safe_float(bals.get("btc", 0.0))
        usd_bal = _safe_float(bals.get("usd", 0.0))
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
            from kraken_trading_btc import get_pair_metadata
            pair_meta = get_pair_metadata(KRAKEN_PAIR_CODE)
            MIN_LOT_SIZE = float(pair_meta.get("lot_decimals", 0.0001))
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
            from kraken_trading_btc import get_pair_metadata
            pair_meta = get_pair_metadata(KRAKEN_PAIR_CODE)
            MIN_LOT_SIZE = float(pair_meta.get("lot_decimals", 0.0001))
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

def run_orb_cycle(now: datetime, settings: Dict[str, Any]) -> Dict[str, Any]:
    state = load_orb_state() or {}
    state.setdefault("phase", "PRE")
    day_key = now.astimezone(NY).date().isoformat()
    if state.get("day") != day_key:
        state = {"day": day_key, "phase": "PRE"}  # reset daily

    # idempotency: at most once per minute
    minute_bucket = now.replace(second=0, microsecond=0).isoformat()
    if state.get("last_minute_bucket") == minute_bucket:
        return {"status":"skipped","reason":"same_minute","phase":state.get("phase")}

    # Circuit breakers
    import os
    day_trades = int(get_setting("orb_day_trades", 0))
    if day_trades >= int(os.getenv("MAX_TRADES_PER_DAY", 6)):
        return {"status":"skipped","reason":"max_trades_reached","phase":state.get("phase")}

    # freshness + spread gates using best_quotes
    try:
        bid, ask, mid, spread_bps = best_quotes()
        if mid <= 0:
            state["last_minute_bucket"] = minute_bucket; save_orb_state(state)
            return {"status":"skipped","reason":"no_price","phase":state.get("phase")}
    except Exception as e:
        logger.warning(f"best_quotes failed: {e}")
        state["last_minute_bucket"] = minute_bucket; save_orb_state(state)
        return {"status":"skipped","reason":"market_data_failed","phase":state.get("phase")}

    if spread_bps > settings.get("max_spread_bps", 8):
        state["last_minute_bucket"] = minute_bucket; save_orb_state(state)
        return {"status":"skipped","reason":"wide_spread","spread_bps":round(spread_bps,1),"phase":state.get("phase")}

    # session gating
    if not is_ny_session_open(now) and not settings.get("orb_debug_force_session"):
        state["last_minute_bucket"] = minute_bucket; save_orb_state(state)
        return {"status":"skipped","reason":"session_closed","phase":state.get("phase")}

    # --- BEGIN PATCH: five-minute close detection ordering ---
    current_five = to_five_min_bucket(now).isoformat()
    prev_five = state.get("last_five_bucket")
    is_five_close = (prev_five != current_five)
    if is_five_close:
        logger.info("ORB_5M_ROLLOVER prev=%s -> curr=%s", prev_five, current_five)

    # ORB build window (09:30–09:45 ET) — unchanged
    if phase_is_pre_or_build(state) and is_orb_build_window(now):
        orb_hi, orb_lo, pct = compute_opening_range()
        if orb_hi and orb_lo:
            state["orb_high"] = float(orb_hi)
            state["orb_low"]  = float(orb_lo)
            state["phase"]    = "WAIT_CONFIRM"
            logger.info("ORB_BUILD hi=%.2f lo=%.2f range_pct=%.4f", orb_hi, orb_lo, pct or 0.0)
            state["last_minute_bucket"] = minute_bucket
            # DO NOT set last_five_bucket here; it will be set after confirm eval
            save_orb_state(state)
            return {"status":"ok","phase":state["phase"],"orb_high":orb_hi,"orb_low":orb_lo,"range_pct":pct}
        else:
            state["phase"] = "ORB_BUILD"
            state["last_minute_bucket"] = minute_bucket
            save_orb_state(state)
            return {"status":"waiting","reason":"building_orb","phase":"ORB_BUILD"}

    # Post-build: evaluate **only** on a new 5m close
    if state.get("phase") == "WAIT_CONFIRM" and is_five_close:
        # Get 5-minute bars for confirmation
        bars_5m = fetch_ohlc(KRAKEN_PAIR_CODE, interval_min=5)
        
        sig = compute_confirmation_signal(
            bars_5m=bars_5m,
            orb_hi=state["orb_high"],
            orb_lo=state["orb_low"],
            ema_fast=settings.get("orb_ema_fast", 9),
            ema_mid=settings.get("orb_ema_mid", 20),
            ema_slow=settings.get("orb_ema_slow", 50),
            buffer_bps=settings.get("orb_buffer_bps", 3),
            use_volume_filter=settings.get("orb_use_volume_filter", False)
        )
        
        # Store diagnostics in state
        state["last_close"] = sig.get("close")
        state["ema9"] = sig.get("ema9")
        state["ema20"] = sig.get("ema20")
        state["ema50"] = sig.get("ema50")
        state["confirm_reason"] = sig.get("reason")
        state["spread_bps"] = round(spread_bps, 1)
        # Store signal information for skip reason context
        state["signal"] = sig.get("dir")
        state["ema_ok"] = sig.get("ema_ok")
        state["break_ok"] = sig.get("break_ok")
        
        # Store last confirmation values for status endpoint
        state["last_signal"] = sig.get("dir")
        state["last_ema_ok"] = sig.get("ema_ok")
        state["last_break_ok"] = sig.get("break_ok")
        state["last_reason"] = sig.get("reason")
        
        logger.info(
            "ORB_DEBUG ema9=%.2f ema20=%.2f ema50=%.2f close=%.2f orb_hi=%.2f orb_lo=%.2f reason=%s",
            sig.get("ema9", float('nan')), sig.get("ema20", float('nan')), sig.get("ema50", float('nan')),
            sig.get("close", float('nan')), state["orb_high"], state["orb_low"], sig.get("reason")
        )
        
        logger.info("ORB_CONFIRM signal=%s ema_ok=%s", sig.get("dir"), sig.get("ema_ok"))
        if sig.get("dir") and sig.get("ema_ok"):
            entry = mid
            stop  = compute_stop(entry, state["orb_high"], state["orb_low"], settings.get("orb_buffer_frac", 0.10), sig["dir"])
            try:
                from kraken_trading_btc import KrakenTradingBot
                bot = KrakenTradingBot()
                acct = bot.get_current_exposure()
                equity = float(acct.get("equity", 0.0))
            except Exception as e:
                logger.warning(f"Failed to get equity: {e}")
                equity = 1000.0  # fallback
            qty   = size_by_risk(settings.get("orb_risk_per_trade", 0.005), equity, entry, stop)
            logger.info("ORB_ENTRY dir=%s entry=%.2f stop=%.2f qty=%.6f", sig["dir"], entry, stop, qty)
            
            # Store risk quantity for status endpoint
            state["last_risk_qty"] = qty
            
            # Use the new execute_orb_entry function with spot-only safety checks
            res = execute_orb_entry(sig["dir"], entry, stop, qty)

            if isinstance(res, dict) and res.get("status") == "success":
                state.update({"phase": sig["dir"].upper(), "avg_price": entry, "qty": qty, "stop": stop})
                # Clear skip reason on successful fill
                state["skip_reason"] = None
                state["risk_qty"] = qty
                state["final_qty"] = qty
                logger.info("ORB_FILLED mode=%s vwap=%.2f qty=%.6f", res.get("mode"), entry, qty)
                
                # Increment day trades counter
                current_day_trades = int(get_setting("orb_day_trades", 0))
                set_setting("orb_day_trades", str(current_day_trades + 1))
                
                # Send Telegram notification
                try:
                    from telemetry import notify_orb_filled
                    notify_orb_filled(sig["dir"], qty, entry, stop)
                except Exception as e:
                    logger.warning(f"Failed to send Telegram notification: {e}")
                
                state["last_minute_bucket"] = minute_bucket
                state["last_five_bucket"]   = current_five  # <-- update AFTER handling the close
                save_orb_state(state)
                return {"status":"filled","phase":state["phase"],"userref":res.get("userref")}
            elif isinstance(res, dict) and res.get("status") == "skipped":
                # Handle skipped orders (insufficient funds, no inventory, etc.)
                skip_reason = res.get("reason")
                logger.info("ORB_SKIP reason=%s", skip_reason)
                state["skip_reason"] = skip_reason
                state["risk_qty"] = qty
                state["final_qty"] = res.get("final_qty", 0.0)
                state["last_minute_bucket"] = minute_bucket
                state["last_five_bucket"]   = current_five
                save_orb_state(state)
                return {"status":"skipped","reason":skip_reason,"detail":res}
            else:
                logger.warning("ORB_ORDER_FAIL %s", res)
                state["skip_reason"] = "order_failed"
                state["risk_qty"] = qty
                state["final_qty"] = 0.0
                state["last_minute_bucket"] = minute_bucket
                state["last_five_bucket"]   = current_five
                save_orb_state(state)
                return {"status":"error","reason":"order_reject","detail":res}

        # no confirm on this 5m close
        state["last_minute_bucket"] = minute_bucket
        state["last_five_bucket"]   = current_five
        save_orb_state(state)
        return {"status":"waiting","reason":"no_confirm","phase":"WAIT_CONFIRM"}

    # Not a 5m close — carry state forward
    state["last_minute_bucket"] = minute_bucket
    # Keep prev_five until the real rollover; do not overwrite here.
    save_orb_state(state)
    return {"status":"ok","phase":state.get("phase")}
    # --- END PATCH ---


