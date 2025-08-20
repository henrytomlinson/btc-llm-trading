#!/usr/bin/env python3
"""
Backtest the allocation strategy on BTCUSD historical data.
Requires a CSV with columns: timestamp, close.
"""
import argparse
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from strategy_core import decide_target_allocation


class BacktestConfig:
    def __init__(self):
        self.initial_equity = 10000.0
        self.fee_bps = 10.0  # 0.10% per trade
        self.max_exposure = 0.8
        self.min_trade_threshold = 0.05  # 5% of equity
        self.cooldown_bars = 3  # hours if data is hourly
        self.min_order_usd = 10.0


def run_backtest(
    df: pd.DataFrame,
    confidence_threshold: float = 0.7,
    max_exposure: float = 0.8,
    cooldown_bars: int = 3,
    fee_bps: float = 10.0,
) -> Tuple[pd.DataFrame, dict]:
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Dummy sentiment/action: use momentum as proxy for LLM
    # momentum > 0 => buy; < 0 => sell; confidence ~ scaled |momentum|
    df["ret"] = df["close"].pct_change().fillna(0.0)
    df["mom"] = df["close"].pct_change(24).fillna(0.0)
    df["action"] = np.where(df["mom"] > 0, "buy", np.where(df["mom"] < 0, "sell", "hold"))
    conf = np.clip(np.abs(df["mom"]) / (df["mom"].abs().quantile(0.95) + 1e-9), 0.0, 1.0)
    df["confidence"] = conf

    # Decide targets via pure function
    df["target"] = [
        decide_target_allocation(a, c, max_exposure) if c >= confidence_threshold else 0.0
        for a, c in zip(df["action"], df["confidence"])
    ]

    # Simulate portfolio
    config = BacktestConfig()
    equity = config.initial_equity
    btc_qty = 0.0
    last_trade_idx = -9999
    fee_rate = fee_bps / 10000.0

    equities = []
    targets = []
    exposures = []
    trades = 0

    for i, row in df.iterrows():
        price = float(row["close"])
        target = float(row["target"])  # [-cap, +cap]

        # Current exposure
        exposure_usd = btc_qty * price
        current_exposure_frac = (exposure_usd / equity) if equity > 0 else 0.0

        # Cooldown
        if i - last_trade_idx < cooldown_bars:
            target = current_exposure_frac

        # Delta
        delta = target - current_exposure_frac
        notional = abs(delta) * equity

        # Threshold and min order
        if notional >= config.min_order_usd and abs(delta) >= config.min_trade_threshold:
            # Execute trade: buy/sell
            if delta > 0:
                usd_to_buy = notional
                fee = usd_to_buy * fee_rate
                btc_qty += (usd_to_buy - fee) / price
                equity -= fee
            else:
                usd_to_sell = notional
                btc_to_sell = usd_to_sell / price
                fee = usd_to_sell * fee_rate
                btc_qty -= btc_to_sell
                equity += (usd_to_sell - fee) - fee  # deduct fee from proceeds
            trades += 1
            last_trade_idx = i

        # Mark-to-market
        equity = (equity - (btc_qty * price)) + (btc_qty * price)

        equities.append(equity)
        targets.append(target)
        exposures.append((btc_qty * price) / equity if equity > 0 else 0.0)

    df["equity"] = equities
    df["target_exposure"] = targets
    df["current_exposure"] = exposures

    # Metrics
    total_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1.0
    num_years = (pd.to_datetime(df["timestamp"]).iloc[-1] - pd.to_datetime(df["timestamp"]).iloc[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / max(num_years, 1e-9)) - 1.0

    # Daily (or bar) returns for Sharpe
    port_rets = pd.Series(df["equity"]).pct_change().fillna(0.0)
    sharpe = np.sqrt(252) * (port_rets.mean() / (port_rets.std() + 1e-9))

    # Max drawdown
    cummax = df["equity"].cummax()
    drawdown = df["equity"] / cummax - 1.0
    max_dd = drawdown.min()

    # Buy and Hold benchmark
    start_price = float(df["close"].iloc[0])
    end_price = float(df["close"].iloc[-1])
    bh_return = end_price / start_price - 1.0
    bh_cagr = (1 + bh_return) ** (1 / max(num_years, 1e-9)) - 1.0

    metrics = {
        "final_equity": float(df["equity"].iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "num_trades": int(trades),
        "benchmark_return": float(bh_return),
        "benchmark_cagr": float(bh_cagr),
    }

    return df, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to BTCUSD CSV with timestamp,close")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--max_exposure", type=float, default=0.8)
    parser.add_argument("--cooldown", type=int, default=3)
    parser.add_argument("--fee_bps", type=float, default=10.0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("CSV must have columns: timestamp, close")

    _, metrics = run_backtest(
        df,
        confidence_threshold=args.confidence,
        max_exposure=args.max_exposure,
        cooldown_bars=args.cooldown,
        fee_bps=args.fee_bps,
    )

    print("Backtest Metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
