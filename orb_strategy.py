#!/usr/bin/env python3
"""
Opening Range Breakout (ORB) deterministic strategy module.

Multi-session support for BTC/USD using London and New York session times.
Keeps state minimal and deterministic; integrates with existing Kraken plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, time, timedelta, timezone
from typing import List, Dict, Optional, Tuple
import pytz

NY = pytz.timezone("America/New_York")
LDN = pytz.timezone("Europe/London")


@dataclass
class SessionDef:
    label: str                 # "LONDON" | "NY"
    tz: str                    # "Europe/London" | "America/New_York"
    open_t: time               # local open
    close_t: time              # local close
    orb_minutes: int = 15
    confirm_minutes: int = 5


@dataclass
class ORBParams:
    sessions: list[SessionDef]
    ema_fast: int = 9
    ema_mid: int = 20
    ema_slow: int = 50
    min_orb_pct: float = 0.002
    buffer_bps: int = 2
    risk_per_trade: float = 0.005
    buffer_frac: float = 0.10
    max_adds: int = 2
    trail_atr_mult: float = 2.0


@dataclass
class ORBState:
    day: str
    session: str               # "LONDON" or "NY"
    orb_high: float | None = None
    orb_low: float | None = None
    direction: str | None = None
    adds_done: int = 0
    in_position: bool = False
    avg_price: float = 0.0
    qty: float = 0.0
    stop: float = 0.0
    last_five_bucket: datetime | None = None
    phase: str = "PRE"


def localize(dt_utc: datetime, tzname: str):
    return dt_utc.astimezone(pytz.timezone(tzname))


def in_local_window(dt_utc: datetime, tzname: str, start: time, end: time) -> bool:
    loc = localize(dt_utc, tzname)
    return start <= loc.time() <= end


def build_orb_window(now_utc: datetime, s: SessionDef) -> tuple[datetime, datetime]:
    loc = localize(now_utc, s.tz)
    d = loc.date()
    start = pytz.timezone(s.tz).localize(datetime(d.year, d.month, d.day, s.open_t.hour, s.open_t.minute))
    orb_end = start + timedelta(minutes=s.orb_minutes)
    return start.astimezone(timezone.utc), orb_end.astimezone(timezone.utc)


def is_orb_build(now_utc: datetime, s: SessionDef) -> bool:
    start_utc, orb_end_utc = build_orb_window(now_utc, s)
    return start_utc <= now_utc < orb_end_utc


def is_confirm_window(now_utc: datetime, s: SessionDef) -> bool:
    _, orb_end_utc = build_orb_window(now_utc, s)
    close_loc = localize(now_utc, s.tz).replace(hour=s.close_t.hour, minute=s.close_t.minute, second=0, microsecond=0)
    close_utc = close_loc.astimezone(timezone.utc)
    return orb_end_utc <= now_utc <= close_utc


def weekday_allowed(now_utc: datetime, allowed: set[int]) -> bool:
    # Python weekday: Mon=0..Sun=6. We map to 1..7 to match env
    return (now_utc.weekday() + 1) in allowed


def _ema(values: List[float], length: int) -> List[float]:
    if length <= 1 or not values:
        return values[:]
    k = 2.0 / (length + 1.0)
    ema_vals: List[float] = []
    last: Optional[float] = None
    for v in values:
        if last is None:
            last = v
        else:
            last = (v - last) * k + last
        ema_vals.append(last)
    return ema_vals


class ORBStrategy:
    def __init__(self, params: Optional[ORBParams] = None):
        self.p = params or ORBParams(sessions=[])

    def build_orb(self, bars_1m: List[Dict], now_utc: datetime, session: SessionDef) -> Optional[Tuple[float, float]]:
        """Compute ORB high/low from session window using 1m bars.

        bars_1m: list of {"ts": datetime (UTC), "high": float, "low": float}
        """
        if not is_orb_build(now_utc, session):
            return None
        start_utc, orb_end_utc = build_orb_window(now_utc, session)
        highs: List[float] = []
        lows: List[float] = []
        for b in bars_1m:
            ts: datetime = b.get("ts")
            if ts and start_utc <= ts < orb_end_utc:
                try:
                    highs.append(float(b["high"]))
                    lows.append(float(b["low"]))
                except Exception:
                    pass
        if not highs or not lows:
            return None
        return (max(highs), min(lows))

    def orb_tight_filter(self, orb_high: float, orb_low: float, ref_price: float) -> bool:
        """Return True if the ORB range is too tight and we should skip the day."""
        rng = max(0.0, float(orb_high) - float(orb_low))
        if ref_price <= 0:
            return True
        return (rng / ref_price) < self.p.min_orb_pct

    def confirm_breakout(self, bars_5m: List[Dict], orb_high: float, orb_low: float) -> Optional[str]:
        """Return "long" or "short" if last 5m close confirms a breakout with EMA alignment."""
        if not bars_5m:
            return None
        closes = [float(b.get("close", 0.0)) for b in bars_5m]
        if not closes:
            return None
        ema_fast = _ema(closes, self.p.ema_fast)[-1]
        ema_mid = _ema(closes, self.p.ema_mid)[-1]
        ema_slow = _ema(closes, self.p.ema_slow)[-1]
        last_close = closes[-1]
        if last_close > orb_high and (ema_fast > ema_mid > ema_slow):
            return "long"
        if last_close < orb_low and (ema_fast < ema_mid < ema_slow):
            return "short"
        return None

    def compute_initial_stop(self, side: str, orb_high: float, orb_low: float) -> float:
        rng = max(0.0, float(orb_high) - float(orb_low))
        if side == "long":
            return float(orb_low) - self.p.buffer_frac * rng
        else:
            return float(orb_high) + self.p.buffer_frac * rng

    def compute_qty_from_risk(self, equity_usd: float, entry: float, stop: float, lot_step: float) -> float:
        risk_usd = max(0.0, self.p.risk_per_trade * float(equity_usd))
        per_unit_risk = max(1e-9, abs(float(entry) - float(stop)))
        raw_qty = risk_usd / per_unit_risk
        # round down to lot step
        if lot_step <= 0:
            return 0.0
        return (raw_qty // lot_step) * lot_step


