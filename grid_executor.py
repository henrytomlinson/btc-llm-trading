#!/usr/bin/env python3
"""
Micro-Grid Executor for Bitcoin Trading
Trades every 15 minutes based on price movements with LLM bias direction.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)

@dataclass
class GridState:
    """Grid trading state"""
    last_grid_price: float
    bias_allocation: float   # -max_exposure..+max_exposure from LLM (e.g., +0.40 = +40% target exposure)
    last_update: datetime
    grid_step_pct: float = 0.25  # each 0.25% price move triggers a micro trade
    grid_order_usd: float = 12.0  # notional per micro trade
    max_grid_exposure: float = 0.1  # maximum 10% exposure from grid trades
    grid_trades_count: int = 0  # track number of grid trades
    total_grid_exposure: float = 0.0  # current grid exposure
    deadband_frac: float = 0.25    # 0.25 = require 25% extra move to flip side
    last_action: Optional[str] = None

# Default configuration
DEFAULT_GRID_STEP_PCT = 0.25   # each 0.25% price move triggers a micro trade
DEFAULT_GRID_ORDER_USD = 12.0  # notional per micro trade
DEFAULT_MAX_GRID_EXPOSURE = 0.1  # maximum 10% exposure from grid trades

# Helpers (put at top of the class/module)
from math import floor
from datetime import datetime, timezone

KRAKEN_PAIR = "XXBTZUSD"  # keep consistent everywhere

# Fallbacks if pair metadata is unavailable
FALLBACK_LOT_STEP = 1e-5     # 0.00001 BTC
FALLBACK_TICK     = 0.10     # $0.10
FALLBACK_OB_DEPTH = 10

def _round_qty(qty: float, lot_step: float) -> float:
    return floor(qty / lot_step) * lot_step

def _round_price(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(price / tick) * tick

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def _slip_cap(best: float, side: str, bps: int) -> float:
    mult = (1 + bps/10_000) if side == "buy" else (1 - bps/10_000)
    return best * mult

def _parse_boolish_ok(r: dict) -> bool:
    # Your wrapper may store {"status":"success"} or include "txid"
    if not isinstance(r, dict):
        return False
    if r.get("status") == "success":
        return True
    if r.get("txid"):
        return True
    return False

def _norm_bias(state: GridState, max_exposure: float) -> float:
    """Derive normalized bias in [-1, +1] from bias_allocation"""
    if max_exposure <= 0:
        return 0.0
    return max(-1.0, min(1.0, state.bias_allocation / max_exposure))

def _grid_notional_caps(side: str, *, equity: float, price: float, btc_qty_now: float,
                        state: GridState, max_exposure: float) -> float:
    """Calculate available notional for grid trades within bias band"""
    # Current BTC value & allocation
    btc_val   = btc_qty_now * price
    alloc_now = (btc_val / equity) if equity > 0 else 0.0

    # Bias band (e.g., bias_allocation=+0.40, max_grid_exposure=0.10 → [0.30, 0.50])
    target = max(-max_exposure, min(max_exposure, state.bias_allocation))
    band_lo = max(-max_exposure, min(max_exposure, target - state.max_grid_exposure))
    band_hi = max(-max_exposure, min(max_exposure, target + state.max_grid_exposure))

    # Convert to dollar caps
    lo_val = max(0.0, band_lo * equity)
    hi_val = max(0.0, band_hi * equity)  # negative allocs clamp to 0 for spot

    if side == "buy":
        room = max(0.0, hi_val - btc_val)
    else:  # sell
        room = max(0.0, btc_val - lo_val)

    return room  # USD notional room available for this side

class GridExecutor:
    """Micro-grid executor for automated scalping"""
    
    def __init__(self, trading_bot=None):
        """Initialize grid executor"""
        self.trading_bot = trading_bot
        self.grid_state = None
        self.enabled = bool(os.getenv("GRID_EXECUTOR_ENABLED", "True"))
        self.min_grid_interval_sec = int(os.getenv("MIN_GRID_INTERVAL_SEC", "900"))  # 15 minutes
        
        # Grid configuration
        self.grid_step_pct = float(os.getenv("GRID_STEP_PCT", str(DEFAULT_GRID_STEP_PCT)))
        self.grid_order_usd = float(os.getenv("GRID_ORDER_USD", str(DEFAULT_GRID_ORDER_USD)))
        self.max_grid_exposure = float(os.getenv("MAX_GRID_EXPOSURE", str(DEFAULT_MAX_GRID_EXPOSURE)))
        self.bias_tilt = float(os.getenv("BIAS_TILT", "0.5"))  # How much bias affects step sizes
        
        # Recommended defaults for "many small trades" mode
        self.grid_step_pct = float(os.getenv("GRID_STEP_PCT", "0.30"))  # 0.25-0.35%
        self.grid_order_usd = float(os.getenv("GRID_ORDER_USD", "12.0"))  # $12
        self.max_grid_exposure = float(os.getenv("MAX_GRID_EXPOSURE", "0.10"))  # 10% band around bias
        self.min_grid_interval_sec = int(os.getenv("MIN_GRID_INTERVAL_SEC", "900"))  # 15 minutes
        self.max_slippage_bps = int(os.getenv("MAX_SLIPPAGE_BPS", "10"))  # 0.10%
        
        logger.info(f"Grid executor initialized: enabled={self.enabled}, step_pct={self.grid_step_pct}%, order_usd=${self.grid_order_usd}, max_exposure={self.max_grid_exposure}")
    
    def _get_pair_meta(self, pair: str = KRAKEN_PAIR) -> tuple[float, float]:
        """
        Returns (lot_step, price_tick). Falls back to safe defaults.
        """
        try:
            meta = self.trading_bot.get_pair_info(pair) if self.trading_bot else None
            if meta:
                lot_step = float(meta.get("lot_step", FALLBACK_LOT_STEP))
                tick     = float(meta.get("tick_size", FALLBACK_TICK))
                return lot_step, tick
        except Exception:
            pass
        return FALLBACK_LOT_STEP, FALLBACK_TICK
    
    def initialize_grid(self, current_price: float, bias_allocation: float) -> GridState:
        """Initialize or reset grid state"""
        self.grid_state = GridState(
            last_grid_price=current_price,
            bias_allocation=bias_allocation,
            last_update=datetime.now(timezone.utc),
            grid_step_pct=self.grid_step_pct,
            grid_order_usd=self.grid_order_usd,
            max_grid_exposure=self.max_grid_exposure,
            deadband_frac=0.25,  # 25% extra move required to flip side
            last_action=None
        )
        logger.info(f"Grid initialized: price=${current_price:.2f}, bias_allocation={bias_allocation:.3f}")
        return self.grid_state
    
    def should_grid_trade(self, price: float, state: GridState = None) -> Optional[str]:
        """
        Decide if a grid trade should execute based on price movement,
        with hysteresis to avoid ping-pong and bias-aware thresholds.
        """
        if not self.enabled:
            logger.info("Grid skip: disabled")
            return None

        if state is None:
            state = self.grid_state
        if state is None:
            logger.warning("Grid skip: no grid state available")
            return None

        # --- time guard (same as you have) ---
        time_since_last = (datetime.now(timezone.utc) - state.last_update).total_seconds()
        if time_since_last < self.min_grid_interval_sec:
            logger.info("Grid skip: time guard (staleness=%ds, min_interval=%ds)", 
                       int(time_since_last), self.min_grid_interval_sec)
            return None

        # --- base step ---
        base_step = max(0.01, float(state.grid_step_pct))  # in percent; floor at 0.01%

        # --- bias-aware tilt ---
        # bias in [-1, +1]. Positive = prefer buys (trigger sooner), negative = prefer sells.
        # 'tilt' scales how much bias changes the step (0.0 = off, 0.5 = moderate).
        tilt = getattr(self, "bias_tilt", 0.5)
        b = _norm_bias(state, state.max_grid_exposure)

        # Smaller step on the preferred side, larger on the opposite side.
        # e.g. with base_step=0.25% and bias=+0.8:
        #   buy_step ≈ 0.25 * (1 - 0.5*0.8) = 0.15%
        #   sell_step ≈ 0.25 * (1 + 0.5*0.8) = 0.35%
        buy_step  = base_step * (1 - tilt * max(0.0, b))
        sell_step = base_step * (1 + tilt * max(0.0, b))
        if b < 0:  # invert if bearish bias
            buy_step  = base_step * (1 + tilt * abs(b))   # harder to buy
            sell_step = base_step * (1 - tilt * abs(b))   # easier to sell

        # --- hysteresis (deadband) ---
        # After a 'buy', require extra % to trigger a 'sell' (and vice versa)
        # to avoid immediate flip-flop from tiny mean-reversions.
        deadband_frac = max(0.0, float(getattr(state, "deadband_frac", 0.25)))
        if state.last_action == "buy":
            sell_step *= (1.0 + deadband_frac)
        elif state.last_action == "sell":
            buy_step  *= (1.0 + deadband_frac)

        # --- trigger levels from the current anchor ---
        anchor = float(state.last_grid_price)
        buy_trigger  = anchor * (1.0 - buy_step / 100.0)
        sell_trigger = anchor * (1.0 + sell_step / 100.0)

        # --- decisions ---
        if price <= buy_trigger:
            return "buy"
        if price >= sell_trigger:
            return "sell"
        
        # Log why no trade was triggered
        logger.info("Grid skip: no trigger (price=%.2f, buy_trigger=%.2f, sell_trigger=%.2f, step=%.3f%%, last_action=%s, staleness=%ds)", 
                   price, buy_trigger, sell_trigger, state.grid_step_pct, state.last_action, int(time_since_last))
        return None
    
    def next_grid_anchor(self, price: float) -> float:
        """Reset grid anchor after each fill to keep steps relative"""
        return price
    
    def update_bias(self, bias_allocation: float):
        """Update the grid bias based on new LLM signal"""
        if self.grid_state:
            self.grid_state.bias_allocation = bias_allocation
            logger.info(f"Grid bias_allocation updated: {bias_allocation:.3f}")
    
    def update_last_action(self, action: str):
        """Update the last action for hysteresis tracking"""
        if self.grid_state:
            self.grid_state.last_action = action
            logger.debug(f"Grid last_action updated: {action}")
    
    def execute_grid_trade(self, action: str, equity: float, btc_qty_now: float, current_price: float):
        """
        Execute a grid trade with actual fills tracking and maker-first orders.
        
        Args:
            action: "buy" or "sell"
            equity: Current portfolio equity
            btc_qty_now: Current BTC quantity
            current_price: Current market price
            
        Returns:
            Trade result dictionary
        """
        if not self.trading_bot:
            return {"status": "error", "reason": "No trading bot available"}
        
        if not self.grid_state:
            return {"status": "error", "reason": "No grid state available"}
        
        # 1) Cap order size to band room
        room_usd = _grid_notional_caps(action, equity=equity, price=current_price,
                                       btc_qty_now=btc_qty_now, state=self.grid_state,
                                       max_exposure=self.max_grid_exposure)

        if room_usd < 10.0:  # EXCHANGE_MIN_NOTIONAL_USD
            logger.info("Grid skip: no room in band (room_usd=%.2f, min_notional=%.2f, step=%.3f%%, last_action=%s)", 
                       room_usd, 10.0, self.grid_state.grid_step_pct, self.grid_state.last_action)
            return {"status":"skipped","reason":"no_room_in_band"}

        order_usd = min(self.grid_state.grid_order_usd, room_usd)
        qty_raw   = order_usd / current_price
        lot_step, _ = self._get_pair_meta()
        qty       = _round_qty(qty_raw, lot_step)
        if qty <= 0:
            logger.info("Grid skip: quantity rounding zero (qty_raw=%.6f, qty=%.6f, order_usd=%.2f, price=%.2f)", 
                       qty_raw, qty, order_usd, current_price)
            return {"status":"skipped","reason":"qty_rounding_zero"}

        # 2) Maker-first
        side = "buy" if action=="buy" else "sell"
        ok, res = self.place_limit_post_only(side, qty, current_price)  # implement best_bid/ask snap inside
        if not ok:
            res = self.place_market_with_slippage_cap(side, qty, max_slippage_bps=10)

        if isinstance(res, dict) and res.get("status") == "success":
            # 3) Use *actual* fills to update exposure
            fills = res.get("fills", []) or []
            filled_qty = sum(float(f.get("qty", 0)) for f in fills)
            notional   = sum(float(f.get("qty", 0))*float(f.get("price", current_price)) for f in fills)
            if filled_qty <= 0:
                logger.info("Grid skip: no fills (action=%s, step=%.3f%%, last_action=%s)", 
                           action, self.grid_state.grid_step_pct, self.grid_state.last_action)
                return {"status":"skipped","reason":"no_fills"}

            # 4) Update grid state (anchor, time, last_action, exposure)
            st = self.grid_state
            st.last_grid_price = current_price
            st.last_update = datetime.now(timezone.utc)
            st.last_action = action
            st.grid_trades_count += 1

            # exposure updated from *actual* notional
            exp_delta = (notional / equity)
            if action == "buy":
                st.total_grid_exposure += exp_delta
            else:
                st.total_grid_exposure -= exp_delta

            # Save grid state to database
            try:
                from auto_trade_scheduler import save_grid_state
                save_grid_state("BTC", st)
            except Exception as e:
                logger.warning(f"Failed to save grid state: {e}")
            
            logger.info("Grid %s: fills=%.6f BTC, $%.2f, exposure=%.3f (bias=%.3f)",
                        action, filled_qty, notional, st.total_grid_exposure, st.bias_allocation)
            return res

        logger.info("Grid skip: order rejected (action=%s, step=%.3f%%, last_action=%s)", 
                   action, self.grid_state.grid_step_pct, self.grid_state.last_action)
        return {"status":"error","reason":"order_rejected"}
    
    def place_limit_post_only(self, side: str, qty: float, price: float) -> tuple[bool, dict]:
        """
        Maker-first order: place a post-only limit that is guaranteed non-marketable.
        Returns (ok_flag, result). ok_flag=False means "do fallback" (e.g., post-only would take).
        """
        pair = KRAKEN_PAIR
        lot_step, tick = self._get_pair_meta(pair)

        # 1) Snapshot book and compute a *non-marketable* price
        ob = self.trading_bot.get_orderbook_snapshot(pair, depth=FALLBACK_OB_DEPTH) or {}
        best_bid = float(ob.get("best_bid", 0.0))
        best_ask = float(ob.get("best_ask", 0.0))
        if best_bid <= 0 or best_ask <= 0:
            return (False, {"status":"error","reason":"no_orderbook"})

        # For post-only:
        #  - BUY must be strictly below best_ask
        #  - SELL must be strictly above best_bid
        if side == "buy":
            po_price = min(best_bid, best_ask - tick)
        else:
            po_price = max(best_ask, best_bid + tick)

        # Honor caller's reference price as a ceiling/floor
        if side == "buy":
            po_price = min(po_price, price)
        else:
            po_price = max(po_price, price)

        po_price = _round_price(po_price, tick)
        if side == "buy" and po_price >= best_ask:
            po_price = best_ask - tick
        if side == "sell" and po_price <= best_bid:
            po_price = best_bid + tick

        # 2) Round qty to lot step
        vol = _round_qty(float(qty), lot_step)
        if vol <= 0:
            return (False, {"status":"skipped","reason":"qty_rounding_zero"})

        # 3) Client ref for idempotency
        userref = f"grid-po-{side}-{_now_iso()}"

        payload = {
            "pair": pair,
            "type": side,                # "buy" | "sell"
            "ordertype": "limit",
            "price": f"{po_price:.10f}",
            "volume": f"{vol:.10f}",
            "oflags": "post",            # <-- post-only
            "userref": userref,
            # "timeinforce": "GTC",      # default
            # "validate": True,          # turn on if you want dry-run checks
        }

        try:
            resp = self.trading_bot._kraken_add_order(payload)    # your REST wrapper
            # If Kraken says "EOrder:PostOnly" or similar, treat as not-ok so caller can fallback
            err = (resp or {}).get("error") or []
            if any("postonly" in str(e).lower() for e in err):
                return (False, {"status":"rejected","reason":"postonly_would_take","payload":payload})

            if _parse_boolish_ok(resp):
                resp["status"] = "success"
                resp["mode"] = "post_only"
                resp["price_used"] = po_price
                resp["userref"] = userref
                return (True, resp)

            return (False, {"status":"error","reason":"kraken_reject","resp":resp,"payload":payload})

        except Exception as e:
            return (False, {"status":"error","reason":str(e)})
    
    def place_market_with_slippage_cap(self, side: str, qty: float, max_slippage_bps: int = 10) -> dict:
        """
        Slippage-capped 'market': submit a marketable LIMIT at best±cap with IOC.
        """
        pair = KRAKEN_PAIR
        lot_step, tick = self._get_pair_meta(pair)

        ob = self.trading_bot.get_orderbook_snapshot(pair, depth=FALLBACK_OB_DEPTH) or {}
        best_bid = float(ob.get("best_bid", 0.0))
        best_ask = float(ob.get("best_ask", 0.0))
        if best_bid <= 0 or best_ask <= 0:
            return {"status":"error","reason":"no_orderbook"}

        cap_price = _slip_cap(best_ask if side=="buy" else best_bid, side, max_slippage_bps)
        limit_px  = _round_price(cap_price, tick)

        vol = _round_qty(float(qty), lot_step)
        if vol <= 0:
            return {"status":"skipped","reason":"qty_rounding_zero"}

        userref = f"grid-mkcap-{side}-{_now_iso()}"

        payload = {
            "pair": pair,
            "type": side,               # "buy" | "sell"
            "ordertype": "limit",       # marketable limit
            "price": f"{limit_px:.10f}",
            "volume": f"{vol:.10f}",
            "timeinforce": "IOC",       # fill what you can up to the cap, then cancel
            "userref": userref,
        }

        try:
            resp = self.trading_bot._kraken_add_order(payload)
            if _parse_boolish_ok(resp):
                resp["status"] = "success"
                resp["mode"] = "mk_slip_cap"
                resp["price_cap"] = limit_px
                resp["userref"] = userref
                return resp

            return {"status":"error","reason":"kraken_reject","resp":resp,"payload":payload}

        except Exception as e:
            return {"status":"error","reason":str(e)}
    
    def update_bias(self, bias_allocation: float):
        """Update the LLM bias allocation"""
        if self.grid_state:
            self.grid_state.bias_allocation = bias_allocation
            logger.info(f"Grid bias_allocation updated: {bias_allocation:.3f}")
    
    def get_grid_status(self) -> Dict[str, Any]:
        """Get current grid status"""
        if not self.grid_state:
            return {"enabled": self.enabled, "initialized": False}
        
        return {
            "enabled": self.enabled,
            "initialized": True,
            "last_grid_price": self.grid_state.last_grid_price,
            "bias_allocation": self.grid_state.bias_allocation,
            "last_update": self.grid_state.last_update.isoformat(),
            "grid_step_pct": self.grid_state.grid_step_pct,
            "grid_order_usd": self.grid_state.grid_order_usd,
            "max_grid_exposure": self.grid_state.max_grid_exposure,
            "grid_trades_count": self.grid_state.grid_trades_count,
            "total_grid_exposure": self.grid_state.total_grid_exposure,
            "time_since_last": (datetime.now(timezone.utc) - self.grid_state.last_update).total_seconds()
        }
    
    def reset_grid(self, current_price: float, bias_allocation: float = 0.0):
        """Reset grid state"""
        self.initialize_grid(current_price, bias_allocation)
        logger.info("Grid state reset")
    
    def self_test_order_placement(self) -> dict:
        """
        Quick self-test (no live keys) to validate order placement logic.
        Tests both sides when the book is thin (tick rounding matters).
        """
        logger.info("🧪 Starting grid order placement self-test...")
        
        test_results = {
            "pair_metadata": {},
            "post_only_orders": {},
            "market_orders": {},
            "overall_status": "unknown"
        }
        
        try:
            # Test 1: Pair metadata retrieval
            logger.info("📊 Testing pair metadata retrieval...")
            lot_step, tick = self._get_pair_meta(KRAKEN_PAIR)
            test_results["pair_metadata"] = {
                "lot_step": lot_step,
                "tick": tick,
                "status": "success"
            }
            logger.info(f"✅ Pair metadata: lot_step={lot_step}, tick={tick}")
            
            # Test 2: Post-only order validation (with validate=True)
            logger.info("🔒 Testing post-only order validation...")
            
            # Test buy side
            buy_test_qty = 0.0001  # Small test quantity
            buy_test_price = 45000.0  # Test price
            
            buy_payload = {
                "pair": KRAKEN_PAIR,
                "type": "buy",
                "ordertype": "limit",
                "price": f"{buy_test_price:.10f}",
                "volume": f"{buy_test_qty:.10f}",
                "oflags": "post",
                "userref": f"test-buy-{_now_iso()}",
                "validate": True  # Dry-run validation
            }
            
            try:
                buy_resp = self.trading_bot._kraken_add_order(buy_payload)
                test_results["post_only_orders"]["buy"] = {
                    "payload": buy_payload,
                    "response": buy_resp,
                    "status": "success" if _parse_boolish_ok(buy_resp) else "error"
                }
                logger.info(f"✅ Buy post-only validation: {buy_resp}")
            except Exception as e:
                test_results["post_only_orders"]["buy"] = {
                    "payload": buy_payload,
                    "error": str(e),
                    "status": "error"
                }
                logger.warning(f"❌ Buy post-only validation failed: {e}")
            
            # Test sell side
            sell_test_qty = 0.0001  # Small test quantity
            sell_test_price = 45000.0  # Test price
            
            sell_payload = {
                "pair": KRAKEN_PAIR,
                "type": "sell",
                "ordertype": "limit",
                "price": f"{sell_test_price:.10f}",
                "volume": f"{sell_test_qty:.10f}",
                "oflags": "post",
                "userref": f"test-sell-{_now_iso()}",
                "validate": True  # Dry-run validation
            }
            
            try:
                sell_resp = self.trading_bot._kraken_add_order(sell_payload)
                test_results["post_only_orders"]["sell"] = {
                    "payload": sell_payload,
                    "response": sell_resp,
                    "status": "success" if _parse_boolish_ok(sell_resp) else "error"
                }
                logger.info(f"✅ Sell post-only validation: {sell_resp}")
            except Exception as e:
                test_results["post_only_orders"]["sell"] = {
                    "payload": sell_payload,
                    "error": str(e),
                    "status": "error"
                }
                logger.warning(f"❌ Sell post-only validation failed: {e}")
            
            # Test 3: Market order validation (with validate=True)
            logger.info("📈 Testing market order validation...")
            
            market_payload = {
                "pair": KRAKEN_PAIR,
                "type": "buy",
                "ordertype": "limit",
                "price": f"{buy_test_price:.10f}",
                "volume": f"{buy_test_qty:.10f}",
                "timeinforce": "IOC",
                "userref": f"test-market-{_now_iso()}",
                "validate": True  # Dry-run validation
            }
            
            try:
                market_resp = self.trading_bot._kraken_add_order(market_payload)
                test_results["market_orders"] = {
                    "payload": market_payload,
                    "response": market_resp,
                    "status": "success" if _parse_boolish_ok(market_resp) else "error"
                }
                logger.info(f"✅ Market order validation: {market_resp}")
            except Exception as e:
                test_results["market_orders"] = {
                    "payload": market_payload,
                    "error": str(e),
                    "status": "error"
                }
                logger.warning(f"❌ Market order validation failed: {e}")
            
            # Determine overall status
            all_tests = [
                test_results["post_only_orders"].get("buy", {}).get("status"),
                test_results["post_only_orders"].get("sell", {}).get("status"),
                test_results["market_orders"].get("status")
            ]
            
            if all(status == "success" for status in all_tests if status):
                test_results["overall_status"] = "success"
                logger.info("🎉 All order placement tests passed!")
            else:
                test_results["overall_status"] = "partial_failure"
                logger.warning("⚠️ Some order placement tests failed")
            
        except Exception as e:
            test_results["overall_status"] = "error"
            test_results["error"] = str(e)
            logger.error(f"❌ Self-test failed: {e}")
        
        return test_results
