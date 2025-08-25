BTC LLM Trading — Grid & Execution Upgrades (one-shot task)

Goals
	•	Make many small, safe trades without ping-pong.
	•	Never exceed exposure caps or sell grid inventory at a loss.
	•	Keep trading only on fresh, tight-spread data.
	•	Get clear telemetry: why a trade fired or was skipped.

⸻

0) Assumptions
	•	Pair: XXBTZUSD (Kraken spot).
	•	Price feed: already implemented (WS + REST) with /diagnostics/price.
	•	Scheduler: 15-min cadence, “same candle” guard present.

1) Fix bool settings parsing (strings → real bools)

File: utils_bool.py

def parse_bool(v, default=False):
    if isinstance(v, bool): return v
    if v is None: return default
    s = str(v).strip().lower()
    return s in {"1","true","t","yes","y","on"}

2) Order placement: maker-first with safe fallbacks

File: kraken_trading_btc.py (inside KrakenTradingBot)

Helpers:

from math import floor
from datetime import datetime, timezone

KRAKEN_PAIR = "XXBTZUSD"
FALLBACK_LOT_STEP = 1e-5
FALLBACK_TICK     = 0.10
FALLBACK_OB_DEPTH = 10

def _now_iso(): return datetime.now(timezone.utc).isoformat()
def _round_qty(q, step): return floor(float(q) / step) * step
def _round_price(p, tick): return round(float(p) / tick) * tick if tick>0 else float(p)
def _slip_cap(best, side, bps): 
    m = (1 + bps/10_000) if side=="buy" else (1 - bps/10_000); return best*m

def _get_pair_meta(self, pair=KRAKEN_PAIR):
    try:
        meta = self.get_pair_info(pair)  # your existing wrapper if present
        lot_step = float(meta.get("lot_step", FALLBACK_LOT_STEP))
        tick     = float(meta.get("tick_size", FALLBACK_TICK))
        return lot_step, tick
    except Exception:
        return FALLBACK_LOT_STEP, FALLBACK_TICK

def _parse_ok(resp: dict) -> bool:
    if not isinstance(resp, dict): return False
    return resp.get("status")=="success" or bool(resp.get("txid"))


def place_limit_post_only(self, side: str, qty: float, price: float) -> tuple[bool, dict]:
    pair = KRAKEN_PAIR
    lot_step, tick = _get_pair_meta(self, pair)
    ob = self.get_orderbook_snapshot(pair, depth=FALLBACK_OB_DEPTH) or {}
    best_bid, best_ask = float(ob.get("best_bid",0)), float(ob.get("best_ask",0))
    if best_bid<=0 or best_ask<=0: return (False, {"status":"error","reason":"no_orderbook"})

    # ensure non-marketable post-only price
    if side=="buy":  po = min(best_bid, best_ask - tick, price)
    else:            po = max(best_ask, best_bid + tick, price)
    po = _round_price(po, tick)
    if side=="buy"  and po>=best_ask: po = best_ask - tick
    if side=="sell" and po<=best_bid: po = best_bid + tick

    vol = _round_qty(qty, lot_step)
    if vol<=0: return (False, {"status":"skipped","reason":"qty_rounding_zero"})
    userref = f"grid-po-{side}-{_now_iso()}"

    payload = {"pair":pair,"type":side,"ordertype":"limit","price":f"{po:.10f}",
               "volume":f"{vol:.10f}","oflags":"post","userref":userref}
    try:
        resp = self._kraken_add_order(payload)
        err = (resp or {}).get("error") or []
        if any("post" in str(e).lower() and "take" in str(e).lower() for e in err):
            return (False, {"status":"rejected","reason":"postonly_would_take","payload":payload})
        if _parse_ok(resp):
            resp.update({"status":"success","mode":"post_only","price_used":po,"userref":userref})
            return (True, resp)
        return (False, {"status":"error","reason":"kraken_reject","resp":resp,"payload":payload})
    except Exception as e:
        return (False, {"status":"error","reason":str(e)})

def place_market_with_slippage_cap(self, side: str, qty: float, max_slippage_bps: int = 10) -> dict:
    pair = KRAKEN_PAIR
    lot_step, tick = _get_pair_meta(self, pair)
    ob = self.get_orderbook_snapshot(pair, depth=FALLBACK_OB_DEPTH) or {}
    best_bid, best_ask = float(ob.get("best_bid",0)), float(ob.get("best_ask",0))
    if best_bid<=0 or best_ask<=0: return {"status":"error","reason":"no_orderbook"}

    cap = _slip_cap(best_ask if side=="buy" else best_bid, side, max_slippage_bps)
    px  = _round_price(cap, tick)
    vol = _round_qty(qty, lot_step)
    if vol<=0: return {"status":"skipped","reason":"qty_rounding_zero"}
    userref = f"grid-mkcap-{side}-{_now_iso()}"

    payload = {"pair":pair,"type":side,"ordertype":"limit","price":f"{px:.10f}",
               "volume":f"{vol:.10f}","timeinforce":"IOC","userref":userref}
    try:
        resp = self._kraken_add_order(payload)
        if _parse_ok(resp):
            resp.update({"status":"success","mode":"mk_slip_cap","price_cap":px,"userref":userref})
            return resp
        return {"status":"error","reason":"kraken_reject","resp":resp,"payload":payload}
    except Exception as e:
        return {"status":"error","reason":str(e)}

3) Grid state — inventory/VWAP + profit guard

File: grid_state.py (or where GridState lives)

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class GridState:
    last_grid_price: float
    bias_allocation: float           # target allocation from LLM in [-max_exp, +max_exp]
    last_update: datetime
    grid_step_pct: float = 0.25
    grid_order_usd: float = 12.0
    max_grid_exposure: float = 0.10  # band around bias
    grid_trades_count: int = 0
    total_grid_exposure: float = 0.0
    deadband_frac: float = 0.25
    last_action: Optional[str] = None
    # NEW
    grid_qty: float = 0.0            # BTC held by grid
    grid_avg_cost: float = 0.0       # USD VWAP
    min_profit_bps: int = 8          # min +0.08% over VWAP to sell

def _norm_bias(state: GridState, max_exposure: float) -> float:
    if max_exposure <= 0: return 0.0
    x = state.bias_allocation / max_exposure
    return max(-1.0, min(1.0, x))

def _update_grid_inventory_after_fill(state: GridState, side: str, fills: list[dict], equity: float):
    filled_qty = sum(float(f.get("qty",0)) for f in fills)
    notional   = sum(float(f.get("qty",0))*float(f.get("price",0)) for f in fills)
    if filled_qty<=0: return
    if side=="buy":
        new_qty = state.grid_qty + filled_qty
        if new_qty>0:
            state.grid_avg_cost = ((state.grid_qty*state.grid_avg_cost) + notional) / new_qty
        state.grid_qty = new_qty
        state.total_grid_exposure += (notional / equity)
    else:
        state.grid_qty = max(0.0, state.grid_qty - filled_qty)
        state.total_grid_exposure -= (notional / equity)
        if state.grid_qty==0:
            state.grid_avg_cost = 0.0
    state.grid_trades_count += 1

def _grid_can_sell_for_profit(state: GridState, price: float) -> bool:
    if state.grid_qty<=0: return True
    breakeven = state.grid_avg_cost * (1 + state.min_profit_bps/10_000)
    return price >= breakeven

4) should_grid_trade with bias tilt + hysteresis + spread guard

File: your grid executor

def should_grid_trade(self, price: float, state: GridState = None) -> Optional[str]:
    if not self.enabled: return None
    state = state or self.grid_state
    if state is None: 
        logger.warning("No grid state"); return None

    # time guard
    if (datetime.now(timezone.utc) - state.last_update).total_seconds() < self.min_grid_interval_sec:
        return None

    # spread guard
    ob = self.get_orderbook_snapshot("XXBTZUSD")
    best_bid, best_ask = float(ob["best_bid"]), float(ob["best_ask"])
    mid = 0.5*(best_bid+best_ask)
    spread_bps = (best_ask - best_bid)/mid * 1e4
    if spread_bps > getattr(self, "max_spread_bps", 8):
        logger.info("Grid skip: spread %.1f bps too wide", spread_bps); return None

    # adaptive step (optional simple version)
    base_step = max(0.01, float(state.grid_step_pct))
    b = _norm_bias(state, self.max_exposure)         # -1..+1
    tilt = getattr(self, "bias_tilt", 0.5)           # 0..1

    buy_step  = base_step * (1 - tilt * max(0.0, b))
    sell_step = base_step * (1 + tilt * max(0.0, b))
    if b < 0:  # bearish bias
        buy_step  = base_step * (1 + tilt * abs(b))
        sell_step = base_step * (1 - tilt * abs(b))

    # hysteresis
    deadband = max(0.0, float(state.deadband_frac))
    if state.last_action == "buy":  sell_step *= (1.0 + deadband)
    if state.last_action == "sell": buy_step  *= (1.0 + deadband)

    anchor = float(state.last_grid_price)
    if price <= anchor * (1 - buy_step/100):  return "buy"
    if price >= anchor * (1 + sell_step/100): return "sell"
    return None


5) Exposure-aware sizing + bias band caps

File: grid executor or scheduler

EXCHANGE_MIN_NOTIONAL_USD = 10.0

def _grid_band_room_usd(side: str, equity: float, price: float, btc_qty_now: float, 
                        state: GridState, max_exposure: float) -> float:
    btc_val   = btc_qty_now * price
    alloc_now = (btc_val / equity) if equity>0 else 0.0
    target = max(-max_exposure, min(max_exposure, state.bias_allocation))
    lo, hi = max(-max_exposure, target - state.max_grid_exposure), min(max_exposure, target + state.max_grid_exposure)
    lo_val, hi_val = max(0.0, lo*equity), max(0.0, hi*equity)  # spot only
    room = (hi_val - btc_val) if side=="buy" else (btc_val - lo_val)
    return max(0.0, room)

def _cap_by_exposure(side, qty, price, equity, max_exposure, btc_qty_now):
    btc_val, max_val = btc_qty_now*price, max_exposure*equity
    if side=="buy":
        return max(0.0, min(qty, (max_val-btc_val)/price))
    else:
        return max(0.0, min(qty, btc_qty_now))

6) Execute grid trade (profit guard + real fills + anchor integrity)

File: grid executor

def execute_grid_trade(self, action: str, equity: float, btc_qty_now: float, current_price: float):
    # Profit guard for sells
    if action=="sell" and not _grid_can_sell_for_profit(self.grid_state, current_price):
        logger.info("Grid skip: sell below breakeven (px=%.2f vwap=%.2f +%dbps)",
                    current_price, self.grid_state.grid_avg_cost, self.grid_state.min_profit_bps)
        return {"status":"skipped","reason":"grid_sell_below_breakeven"}

    room_usd = _grid_band_room_usd(action, equity, current_price, btc_qty_now, 
                                   self.grid_state, self.max_exposure)
    if room_usd < EXCHANGE_MIN_NOTIONAL_USD:
        return {"status":"skipped","reason":"no_room_in_band"}

    order_usd = min(self.grid_state.grid_order_usd, room_usd)
    qty_raw   = order_usd / current_price
    qty_cap   = _cap_by_exposure(action, qty_raw, current_price, equity, self.max_exposure, btc_qty_now)
    qty       = max(0.0, qty_cap)
    if qty <= 0: return {"status":"skipped","reason":"qty_after_caps_zero"}

    side = "buy" if action=="buy" else "sell"
    ok, res = self.place_limit_post_only(side, qty, current_price)
    if not ok:
        res = self.place_market_with_slippage_cap(side, qty, max_slippage_bps=10)

    if isinstance(res, dict) and res.get("status")=="success":
        fills = res.get("fills", []) or []
        requested = qty
        filled_qty = sum(float(f.get("qty",0)) for f in fills)
        if filled_qty <= 0: return {"status":"skipped","reason":"no_fills"}

        # Update inventory from real fills
        _update_grid_inventory_after_fill(self.grid_state, action, fills, equity)

        # Reset anchor only for meaningful fills
        fill_ratio = filled_qty / requested if requested>0 else 1.0
        if fill_ratio >= 0.25:
            self.grid_state.last_grid_price = current_price
            self.grid_state.last_action = action
            self.grid_state.last_update = datetime.now(timezone.utc)

        save_grid_state("BTC", self.grid_state)
        return res

    return {"status":"error","reason":"order_rejected"}

7) Allocation sells shouldn’t dump grid inventory at a loss

File: allocation/position code (where you size rebalance sells)

def size_rebalance_sell_without_losing_grid(state: GridState, desired_qty: float, price: float) -> float:
    if state.grid_qty <= 0: return desired_qty
    breakeven = state.grid_avg_cost * (1 + state.min_profit_bps/10_000)
    # protect grid if below breakeven
    return desired_qty if price >= breakeven else max(0.0, desired_qty - state.grid_qty)

8) Per-side same-candle guard

File: auto_trade_scheduler.py

def current_candle_ts(now, minutes=15):
    base = now.replace(second=0, microsecond=0)
    bucket = (base.minute // minutes) * minutes
    return base.replace(minute=bucket)

# before any grid/alloc place:
now = datetime.now(timezone.utc)
key = f"last_{side}_candle_ts"  # side = 'buy'|'sell'
cur = current_candle_ts(now).isoformat()
meta = read_settings() or {}
if meta.get(key) == cur:
    return {"status":"skipped","reason":f"same_candle_{side}"}
write_setting(key, cur)

9) Spread/slippage telemetry (Prometheus and/or logs)

Emit:
	•	grid_trades_total{side,mode}
	•	grid_skips_total{reason}
	•	maker_fills_total, market_fallback_total
	•	slippage_bps_histogram
	•	maker fill ratio and median slippage on the UI “Trade Quality” card.

10) Defaults (for “many small trades” mode)
	•	Grid step: 0.25%
	•	Deadband: 0.25
	•	Grid order: $12 (optionally σ-adaptive to $18–22)
	•	Bias tilt: 0.5
	•	Max spread guard: 6–8 bps
	•	Min interval: 15m
	•	Bias band: ±10% around bias
	•	Max exposure (global): 0.7
	•	LLM min confidence: 0.80–0.85
	•	Min expected move: 0.00–0.02% (spread guard does the heavy lifting)
	•	Daily PnL limit: $5 (start small)

11) Sanity tests (run on VPS)

export D="https://<your-domain>"

# Price freshness
curl -s $D/diagnostics/price | jq .

# PnL, exposure
curl -s $D/pnl_summary | jq '{status,equity,position_qty,exposure_pct,price_ts}'

# Two runs in same bar; ensure per-side guard works (<=1 trade per side)
( curl -s -X POST $D/auto_trade & sleep 5; curl -s -X POST $D/auto_trade ) ; wait

# Check trade quality metrics
curl -s $D/metrics | egrep 'grid_trades_total|grid_skips_total|maker_fills_total|market_fallback_total|slippage'

Pass: status ok, price age < 120s, exposure within bias ± grid band, trades do not exceed 1 per side per candle, metrics increment as expected.