#!/usr/bin/env python3
"""
Telemetry module for tracking trade quality metrics
"""

import time
import logging
from datetime import datetime, timezone
from collections import defaultdict, deque
from typing import Dict, List, Optional
import threading

logger = logging.getLogger(__name__)

class TradeTelemetry:
    """Track trade quality metrics for monitoring and tuning"""
    
    def __init__(self):
        self.lock = threading.Lock()
        
        # Counters
        self.grid_trades_total = defaultdict(int)  # {side: count}
        self.grid_skips_total = defaultdict(int)   # {reason: count}
        self.maker_fills_total = 0
        self.market_fallback_total = 0
        
        # Slippage tracking (EWMA)
        self.avg_slippage_bps = 0.0
        self.slippage_alpha = 0.1  # EWMA smoothing factor
        self.slippage_samples = 0
        
        # Daily tracking
        self.daily_trades = 0
        self.daily_reset_time = self._get_daily_reset_time()
        
        # Recent trades for UI
        self.recent_trades = deque(maxlen=100)
        
    def _get_daily_reset_time(self) -> datetime:
        """Get next daily reset time (midnight UTC)"""
        now = datetime.now(timezone.utc)
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _check_daily_reset(self):
        """Reset daily counters if needed"""
        now = datetime.now(timezone.utc)
        if now >= self.daily_reset_time:
            with self.lock:
                self.daily_trades = 0
                self.daily_reset_time = self._get_daily_reset_time()
                logger.info("Daily telemetry counters reset")
    
    def record_grid_trade(self, side: str, order_type: str, slippage_bps: Optional[float] = None):
        """Record a completed grid trade"""
        with self.lock:
            self._check_daily_reset()
            
            # Update counters
            self.grid_trades_total[side] += 1
            self.daily_trades += 1
            
            if order_type == "maker":
                self.maker_fills_total += 1
            elif order_type == "market":
                self.market_fallback_total += 1
            
            # Update slippage EWMA
            if slippage_bps is not None:
                if self.slippage_samples == 0:
                    self.avg_slippage_bps = slippage_bps
                else:
                    self.avg_slippage_bps = (self.slippage_alpha * slippage_bps + 
                                           (1 - self.slippage_alpha) * self.avg_slippage_bps)
                self.slippage_samples += 1
            
            # Record recent trade
            trade_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "side": side,
                "order_type": order_type,
                "slippage_bps": slippage_bps,
                "daily_trades": self.daily_trades
            }
            self.recent_trades.append(trade_record)
            
            logger.debug(f"Telemetry: {side} {order_type} trade recorded, slippage={slippage_bps:.2f}bps")
    
    def record_grid_skip(self, reason: str):
        """Record a skipped grid trade"""
        with self.lock:
            self.grid_skips_total[reason] += 1
            logger.debug(f"Telemetry: grid skip recorded, reason={reason}")
    
    def get_maker_ratio(self) -> float:
        """Calculate maker fill ratio"""
        total_fills = self.maker_fills_total + self.market_fallback_total
        if total_fills == 0:
            return 0.0
        return self.maker_fills_total / total_fills
    
    def get_median_slippage(self) -> float:
        """Calculate median slippage from recent trades"""
        slippages = [t.get("slippage_bps", 0) for t in self.recent_trades 
                    if t.get("slippage_bps") is not None]
        if not slippages:
            return 0.0
        
        slippages.sort()
        mid = len(slippages) // 2
        if len(slippages) % 2 == 0:
            return (slippages[mid - 1] + slippages[mid]) / 2
        else:
            return slippages[mid]
    
    def get_top_skip_reasons(self, limit: int = 3) -> List[tuple]:
        """Get top reasons for trade skips"""
        with self.lock:
            sorted_reasons = sorted(self.grid_skips_total.items(), 
                                  key=lambda x: x[1], reverse=True)
            return sorted_reasons[:limit]
    
    def get_metrics(self) -> Dict:
        """Get all telemetry metrics for API/UI"""
        with self.lock:
            self._check_daily_reset()
            
            return {
                "grid_trades_total": dict(self.grid_trades_total),
                "grid_skips_total": dict(self.grid_skips_total),
                "maker_fills_total": self.maker_fills_total,
                "market_fallback_total": self.market_fallback_total,
                "maker_ratio": self.get_maker_ratio(),
                "avg_slippage_bps": self.avg_slippage_bps,
                "median_slippage_bps": self.get_median_slippage(),
                "daily_trades": self.daily_trades,
                "top_skip_reasons": self.get_top_skip_reasons(),
                "slippage_samples": self.slippage_samples,
                "recent_trades_count": len(self.recent_trades)
            }

# Global telemetry instance
TELEMETRY = TradeTelemetry()

def record_grid_trade(side: str, order_type: str, slippage_bps: Optional[float] = None):
    """Record a grid trade (convenience function)"""
    TELEMETRY.record_grid_trade(side, order_type, slippage_bps)

def record_grid_skip(reason: str):
    """Record a grid skip (convenience function)"""
    TELEMETRY.record_grid_skip(reason)

def get_telemetry_metrics() -> Dict:
    """Get telemetry metrics (convenience function)"""
    return TELEMETRY.get_metrics()
