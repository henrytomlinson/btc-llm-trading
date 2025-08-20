#!/usr/bin/env python3
"""
Optimize strategy parameters through comprehensive backtesting.
Tests multiple combinations of confidence, exposure, and cooldown settings.
"""
import argparse
import pandas as pd
import numpy as np
from itertools import product
from backtest import run_backtest

def optimize_strategy(csv_file, output_file="optimization_results.csv"):
    """Run backtests with different parameter combinations"""
    
    print("Loading data...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} data points")
    
    # Parameter ranges to test
    confidence_levels = [0.6, 0.7, 0.8, 0.9]
    max_exposures = [0.6, 0.7, 0.8, 0.9]
    cooldowns = [1, 3, 6, 12]  # hours
    fee_rates = [5, 10, 15]  # basis points
    
    results = []
    total_combinations = len(confidence_levels) * len(max_exposures) * len(cooldowns) * len(fee_rates)
    current = 0
    
    print(f"Testing {total_combinations} parameter combinations...")
    
    for conf, exp, cool, fee in product(confidence_levels, max_exposures, cooldowns, fee_rates):
        current += 1
        print(f"Progress: {current}/{total_combinations} - Testing conf={conf}, exp={exp}, cool={cool}, fee={fee}")
        
        try:
            _, metrics = run_backtest(
                df,
                confidence_threshold=conf,
                max_exposure=exp,
                cooldown_bars=cool,
                fee_bps=fee
            )
            
            # Add parameters to results
            result = {
                'confidence': conf,
                'max_exposure': exp,
                'cooldown_hours': cool,
                'fee_bps': fee,
                'final_equity': metrics['final_equity'],
                'total_return': metrics['total_return'],
                'cagr': metrics['cagr'],
                'sharpe': metrics['sharpe'],
                'max_drawdown': metrics['max_drawdown'],
                'num_trades': metrics['num_trades'],
                'benchmark_return': metrics['benchmark_return'],
                'benchmark_cagr': metrics['benchmark_cagr'],
                'excess_return': metrics['total_return'] - metrics['benchmark_return'],
                'excess_cagr': metrics['cagr'] - metrics['benchmark_cagr']
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error testing conf={conf}, exp={exp}, cool={cool}, fee={fee}: {e}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by different metrics
    print("\n=== TOP 10 BY SHARPE RATIO ===")
    print(results_df.nlargest(10, 'sharpe')[['confidence', 'max_exposure', 'cooldown_hours', 'fee_bps', 'sharpe', 'cagr', 'max_drawdown', 'num_trades']])
    
    print("\n=== TOP 10 BY EXCESS RETURN ===")
    print(results_df.nlargest(10, 'excess_return')[['confidence', 'max_exposure', 'cooldown_hours', 'fee_bps', 'excess_return', 'sharpe', 'max_drawdown', 'num_trades']])
    
    print("\n=== TOP 10 BY CAGR ===")
    print(results_df.nlargest(10, 'cagr')[['confidence', 'max_exposure', 'cooldown_hours', 'fee_bps', 'cagr', 'sharpe', 'max_drawdown', 'num_trades']])
    
    print("\n=== LOWEST MAX DRAWDOWN ===")
    print(results_df.nsmallest(10, 'max_drawdown')[['confidence', 'max_exposure', 'cooldown_hours', 'fee_bps', 'max_drawdown', 'cagr', 'sharpe', 'num_trades']])
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"\nFull results saved to {output_file}")
    
    # Summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Average Sharpe: {results_df['sharpe'].mean():.3f}")
    print(f"Best Sharpe: {results_df['sharpe'].max():.3f}")
    print(f"Average CAGR: {results_df['cagr'].mean():.3f}")
    print(f"Best CAGR: {results_df['cagr'].max():.3f}")
    print(f"Average Max Drawdown: {results_df['max_drawdown'].mean():.3f}")
    print(f"Best Max Drawdown: {results_df['max_drawdown'].min():.3f}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument("--csv", required=True, help="Path to BTCUSD CSV file")
    parser.add_argument("--output", default="optimization_results.csv", help="Output file for results")
    
    args = parser.parse_args()
    
    results = optimize_strategy(args.csv, args.output)
    
    # Find best overall strategy (balanced approach)
    # Weight: 40% Sharpe, 30% CAGR, 20% Max Drawdown, 10% Trade Count
    results['score'] = (
        0.4 * results['sharpe'] + 
        0.3 * results['cagr'] * 100 +  # Scale CAGR to similar range as Sharpe
        0.2 * (1 + results['max_drawdown']) +  # Convert drawdown to positive (lower is better)
        0.1 * (1 / (1 + results['num_trades'] / 100))  # Fewer trades is better
    )
    
    best_overall = results.loc[results['score'].idxmax()]
    print(f"\n=== BEST OVERALL STRATEGY (BALANCED) ===")
    print(f"Confidence: {best_overall['confidence']}")
    print(f"Max Exposure: {best_overall['max_exposure']}")
    print(f"Cooldown: {best_overall['cooldown_hours']} hours")
    print(f"Fee Rate: {best_overall['fee_bps']} bps")
    print(f"Sharpe: {best_overall['sharpe']:.3f}")
    print(f"CAGR: {best_overall['cagr']:.3f}")
    print(f"Max Drawdown: {best_overall['max_drawdown']:.3f}")
    print(f"Number of Trades: {best_overall['num_trades']}")

if __name__ == "__main__":
    main()
