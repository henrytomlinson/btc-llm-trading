ORB Scheduler Fix — Single Task Pack

Goal: ORB is enabled but not running. Make the scheduler call ORB first, add bullet-proof logging + idempotency, and add a /orb/self_test endpoint so we can validate without waiting for the session.

1) Replace execute_auto_trade so ORB definitely runs first

# --- PATCH: force ORB to run first, with robust logging and safe fallbacks.
from datetime import datetime, timezone
from typing import Any, Dict

from orb_executor import run_orb_cycle   # must return dict
# optional: from grid_executor import run_grid_cycle
# optional: from llm_trading_strategy import run_llm_cycle

import json
import logging
logger = logging.getLogger(__name__)

def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)

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
            res = run_orb_cycle(now, settings) or {"status": "error", "reason": "orb_cycle_returned_none"}
            logger.info("ORB_CYCLE %s", safe_json(res))
            return res
    except Exception as e:
        logger.exception("ORB_RUN_ERROR %s", e)

    # 2) (optional) Grid path
    try:
        if settings.get("grid_executor_enabled"):
            res = run_grid_cycle(now, settings) or {"status":"error","reason":"grid_cycle_returned_none"}
            logger.info("GRID_CYCLE %s", safe_json(res))
            return res
    except Exception as e:
        logger.exception("GRID_RUN_ERROR %s", e)

    # 3) (optional) LLM path
    try:
        if settings.get("llm_enabled"):
            res = run_llm_cycle(now, settings) or {"status":"error","reason":"llm_cycle_returned_none"}
            logger.info("LLM_CYCLE %s", safe_json(res))
            return res
    except Exception as e:
        logger.exception("LLM_RUN_ERROR %s", e)

    logger.info("SCHED_NOOP no strategy enabled")
    return {"status": "noop", "reason": "no_strategy_enabled"}


#Important: Remove/avoid any pre-ORB global gates (e.g., llm_min_confidence, min_expected_move) from the scheduler. ORB owns its own guards.

2) Harden run_orb_cycle (logging, guards, idempotency)

File: orb_executor.py
Add/merge the following helpers + wrapper. Keep your existing compute and order functions; this wrapper calls them and logs all outcomes.

# --- PATCH: add helpers and a robust run_orb_cycle wrapper ---
from datetime import datetime, time, timezone
from typing import Any, Dict, Tuple
import logging
import pytz

logger = logging.getLogger(__name__)
NY = pytz.timezone("America/New_York")

# Utilities you likely have already — include if missing:
def load_orb_state() -> Dict[str, Any]:
    return get_setting_json("orb_state") or {}  # your own storage helpers

def save_orb_state(state: Dict[str, Any]) -> None:
    set_setting_json("orb_state", state)

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

def get_market_data() -> Dict[str, Any]:
    """
    You already fetch best_bid/best_ask/mid + age_sec somewhere.
    Return: {'best_bid':float,'best_ask':float,'mid':float,'age_sec':float}
    """
    md = get_best_quotes_and_age()  # your existing function
    mid = 0.5 * (md["best_bid"] + md["best_ask"])
    return {"best_bid":md["best_bid"], "best_ask":md["best_ask"], "mid":mid, "age_sec":md["age_sec"]}

def compute_opening_range() -> Tuple[float,float,float] | Tuple[None,None,None]:
    """Your existing ORB build using 1-min bars between 09:30–09:45 ET; returns (hi, lo, pct_range)."""
    return build_orb_from_bars()

def compute_confirmation_signal(orb_hi: float, orb_lo: float) -> Dict[str, Any]:
    """
    Your existing logic using 5-min close + EMA alignment, returns:
    {'dir': 'long'|'short'|None, 'ema_ok': bool}
    """
    return confirm_breakout_signal(orb_hi, orb_lo)

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

    # freshness + spread gates
    md = get_market_data()
    if md["age_sec"] > settings.get("max_price_staleness_sec", 120):
        state["last_minute_bucket"] = minute_bucket; save_orb_state(state)
        return {"status":"skipped","reason":"stale_price","age_sec":md["age_sec"],"phase":state.get("phase")}

    spread_bps = (md["best_ask"]-md["best_bid"]) / max(1e-9, md["mid"]) * 1e4
    if spread_bps > settings.get("max_spread_bps", 8):
        state["last_minute_bucket"] = minute_bucket; save_orb_state(state)
        return {"status":"skipped","reason":"wide_spread","spread_bps":round(spread_bps,1),"phase":state.get("phase")}

    # session gating
    if not is_ny_session_open(now) and not settings.get("orb_debug_force_session"):
        state["last_minute_bucket"] = minute_bucket; save_orb_state(state)
        return {"status":"skipped","reason":"session_closed","phase":state.get("phase")}

    # mark new 5-minute bucket
    five_bucket_iso = to_five_min_bucket(now).isoformat()
    if state.get("last_five_bucket") != five_bucket_iso:
        logger.info("ORB_5M_ROLLOVER bucket=%s", five_bucket_iso)
        state["last_five_bucket"] = five_bucket_iso

    # ORB build window
    if phase_is_pre_or_build(state) and is_orb_build_window(now):
        orb_hi, orb_lo, pct = compute_opening_range()
        if orb_hi and orb_lo:
            state["orb_high"] = float(orb_hi)
            state["orb_low"]  = float(orb_lo)
            state["phase"]    = "WAIT_CONFIRM"
            logger.info("ORB_BUILD hi=%.2f lo=%.2f range_pct=%.4f", orb_hi, orb_lo, pct or 0.0)
            state["last_minute_bucket"] = minute_bucket; save_orb_state(state)
            return {"status":"ok","phase":state["phase"],"orb_high":orb_hi,"orb_low":orb_lo,"range_pct":pct}
        else:
            state["phase"] = "ORB_BUILD"
            state["last_minute_bucket"] = minute_bucket; save_orb_state(state)
            return {"status":"waiting","reason":"building_orb","phase":"ORB_BUILD"}

    # Post-build: wait for 5-min close confirmation
    if state.get("phase") == "WAIT_CONFIRM" and is_new_5m_close(now, state):
        sig = compute_confirmation_signal(state["orb_high"], state["orb_low"])
        logger.info("ORB_CONFIRM signal=%s ema_ok=%s", sig.get("dir"), sig.get("ema_ok"))
        if sig.get("dir") and sig.get("ema_ok"):
            # Entry + stop + sizing
            entry = md["mid"]
            stop  = compute_stop(entry, state["orb_high"], state["orb_low"], settings.get("orb_buffer_frac", 0.10), sig["dir"])
            qty   = size_by_risk(settings.get("orb_risk_per_trade", 0.005), get_equity(), entry, stop)

            side = "buy" if sig["dir"] == "long" else "sell"
            logger.info("ORB_ENTRY dir=%s entry=%.2f stop=%.2f qty=%.6f", sig["dir"], entry, stop, qty)
            ok, res = place_limit_post_only(side, qty, entry)
            if not ok:
                res = place_market_with_slippage_cap(side, qty, max_slippage_bps=10)

            if isinstance(res, dict) and res.get("status") == "success":
                state.update({"phase": sig["dir"].upper(), "avg_price": entry, "qty": qty, "stop": stop})
                logger.info("ORB_FILLED mode=%s vwap=%.2f qty=%.6f", res.get("mode"), entry, qty)
                state["last_minute_bucket"] = minute_bucket; save_orb_state(state)
                return {"status":"filled","phase":state["phase"],"userref":res.get("userref")}
            else:
                logger.warning("ORB_ORDER_FAIL %s", res)
                state["last_minute_bucket"] = minute_bucket; save_orb_state(state)
                return {"status":"error","reason":"order_reject","detail":res}

        state["last_minute_bucket"] = minute_bucket; save_orb_state(state)
        return {"status":"waiting","reason":"no_confirm","phase":"WAIT_CONFIRM"}

    # Nothing to do this minute
    state["last_minute_bucket"] = minute_bucket; save_orb_state(state)
    return {"status":"ok","phase":state.get("phase")}

#The wrapper always returns a dict and always saves state. Logs include: ORB_5M_ROLLOVER, ORB_BUILD, ORB_CONFIRM, ORB_ENTRY, ORB_FILLED, and explicit skip reasons.

⸻

3) Add a self-test endpoint (validate without waiting for NY)

File: main_btc.py (FastAPI)

# --- PATCH: self-test to simulate a pass at any UTC time ---
from datetime import datetime, timezone
from fastapi import FastAPI, Query
from auto_trade_scheduler import execute_auto_trade
from orb_executor import run_orb_cycle
app: FastAPI  # assume this exists

@app.get("/orb/self_test")
def orb_self_test(
    at: str | None = Query(default=None, description="ISO UTC time e.g. 2025-08-27T14:35:00Z"),
    force_session: bool = Query(default=True)
):
    now = datetime.now(timezone.utc)
    if at:
        try:
            # Accept both Z and +00:00
            iso = at.replace("Z", "+00:00")
            now = datetime.fromisoformat(iso)
        except Exception:
            return {"status":"error","reason":"bad_at_param"}

    settings = load_runtime_settings()
    if force_session:
        settings["orb_debug_force_session"] = True

    res = run_orb_cycle(now, settings)
    return {"now": now.isoformat(), "res": res}



--

4) Verification — run these now

# Rebuild/restart
docker compose up -d --build

# Sanity
curl -s https://henryt-btc.live/health ; echo
curl -s https://henryt-btc.live/diagnostics/price | jq '{source,age_sec}'

# Self-test inside build window (should report ORB_BUILD or WAIT_CONFIRM with hi/lo)
curl -s "https://henryt-btc.live/orb/self_test?at=2025-08-27T14:35:00Z" | jq .

# Self-test just after confirm window (should evaluate confirmation)
curl -s "https://henryt-btc.live/orb/self_test?at=2025-08-27T14:46:00Z" | jq .

# Hit scheduler once manually and tail logs
curl -s -X POST https://henryt-btc.live/auto_trade | jq .
docker compose logs -f --tail=200 | egrep -i "ORB_(SCHED|BUILD|CONFIRM|ENTRY|FILLED|ORDER_FAIL|RUN_ERROR)|SCHED_NOOP|wide_spread|stale_price"

Expected log progression during live session:

ORB_5M_ROLLOVER bucket=...
ORB_BUILD hi=... lo=... range_pct=...
ORB_CONFIRM signal=long ema_ok=True
ORB_ENTRY dir=long entry=... stop=... qty=...
ORB_FILLED mode=post_only vwap=... qty=...

5) Cron (minute tick)

Make sure the scheduler actually runs every minute:

crontab -l | grep auto_trade || (echo "* * * * * /usr/bin/curl -s -X POST https://henryt-btc.live/auto_trade >/dev/null 2>&1" | crontab -)

--

6) If self-test works but live doesn’t
	•	Session TZ math: confirm server clock UTC; ORB converts to America/New_York.
	•	Duplicate global gates: ensure scheduler no longer blocks ORB with LLM-specific checks.
	•	Spread/staleness: logs will say wide_spread or stale_price. Adjust MAX_SPREAD_BPS (8→10) or fix feed.

⸻

7) Done

After this patch, ORB will either trade or return a clear skip reason each minute, and the self-test lets you validate instantly.

⸻
