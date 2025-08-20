#!/usr/bin/env python3
"""
Fetch BTCUSD historical data for backtesting.
Uses yfinance to get real Bitcoin price data.
"""
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def fetch_btc_data(symbol="BTC-USD", period="2y", interval="1h"):
    """Fetch BTCUSD data from Yahoo Finance"""
    print(f"Fetching {symbol} data for {period} with {interval} intervals...")
    
    try:
        # Get data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            print("No data received. Trying alternative symbol...")
            # Try alternative symbol
            ticker = yf.Ticker("BTCUSD=X")
            data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            raise ValueError("Could not fetch BTC data")
        
        # Reset index to get timestamp as column
        data = data.reset_index()
        
        # Rename columns to match our backtest format
        data = data.rename(columns={
            'Datetime': 'timestamp',
            'Date': 'timestamp',
            'Close': 'close'
        })
        
        # Ensure we have required columns
        if 'timestamp' not in data.columns or 'close' not in data.columns:
            raise ValueError("Data missing required columns")
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Successfully fetched {len(data)} data points")
        print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
        
        return data
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Fetch BTCUSD historical data")
    parser.add_argument("--output", default="btcusd_data.csv", help="Output CSV file")
    parser.add_argument("--period", default="2y", help="Data period (1y, 2y, 5y, max)")
    parser.add_argument("--interval", default="1h", help="Data interval (1h, 1d)")
    parser.add_argument("--symbol", default="BTC-USD", help="Symbol to fetch")
    
    args = parser.parse_args()
    
    # Fetch data
    data = fetch_btc_data(args.symbol, args.period, args.interval)
    
    if data is not None:
        # Save to CSV
        data.to_csv(args.output, index=False)
        print(f"Data saved to {args.output}")
        
        # Show sample
        print("\nSample data:")
        print(data.head())
        
        # Show basic stats
        print(f"\nBasic statistics:")
        print(f"Total data points: {len(data)}")
        print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
        print(f"Average price: ${data['close'].mean():.2f}")
        print(f"Price volatility: {data['close'].std():.2f}")
    else:
        print("Failed to fetch data")

if __name__ == "__main__":
    main()
