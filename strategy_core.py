#!/usr/bin/env python3
"""
Core strategy utilities.
Provides a pure decision function that maps normalized inputs to a target allocation.
"""
from typing import Optional


def decide_target_allocation(
    recommended_action: str,
    confidence: float,
    max_exposure: float = 0.8,
) -> float:
    """Pure mapping from action/confidence to target allocation.

    Args:
        recommended_action: 'buy' | 'sell' | 'hold'
        confidence: float in [0, 1]
        max_exposure: cap for allocation magnitude in [0, 1]

    Returns:
        target_exposure in [-max_exposure, +max_exposure]
    """
    try:
        action = (recommended_action or "hold").lower()
        clamped_conf = max(0.0, min(1.0, float(confidence or 0.0)))
        cap = max(0.0, min(1.0, float(max_exposure or 0.0)))
        if action == "buy":
            return cap * clamped_conf
        if action == "sell":
            return -cap * clamped_conf
        return 0.0
    except Exception:
        return 0.0
