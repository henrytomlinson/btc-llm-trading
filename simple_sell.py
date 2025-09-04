#!/usr/bin/env python3
"""Simple script to sell $20 worth of BTC"""

import os
import time
import hmac
import hashlib
import base64
import requests
from urllib.parse import urlencode

# Kraken API credentials
API_KEY = os.getenv('KRAKEN_API_KEY')
API_SECRET = os.getenv('KRAKEN_PRIVATE_KEY')

def kraken_request(uri_path, data, api_key, api_secret):
    """Make authenticated request to Kraken"""
    api_url = "https://api.kraken.com"
    
    # Get API nonce
    nonce = str(int(time.time() * 1000))
    
    # Post data
    postdata = urlencode(data)
    encoded = (str(data['nonce']) + postdata).encode()
    message = uri_path.encode() + hashlib.sha256(encoded).digest()
    
    # Create signature
    signature = hmac.new(base64.b64decode(api_secret), message, hashlib.sha256)
    sigdigest = base64.b64encode(signature.digest())
    
    # Headers
    headers = {
        'API-Key': api_key,
        'API-Signature': sigdigest.decode()
    }
    
    # Make request
    url = api_url + uri_path
    response = requests.post(url, headers=headers, data=data)
    return response.json()

def get_balance():
    """Get current balance"""
    data = {
        'nonce': str(int(time.time() * 1000))
    }
    
    result = kraken_request('/0/private/Balance', data, API_KEY, API_SECRET)
    return result

def get_ticker():
    """Get current BTC price"""
    response = requests.get('https://api.kraken.com/0/public/Ticker?pair=XXBTZUSD')
    data = response.json()
    if 'result' in data and 'XXBTZUSD' in data['result']:
        ticker = data['result']['XXBTZUSD']
        return float(ticker['c'][0])  # Current price
    return None

def place_sell_order(volume, price):
    """Place a sell order"""
    data = {
        'nonce': str(int(time.time() * 1000)),
        'pair': 'XXBTZUSD',
        'type': 'sell',
        'ordertype': 'limit',
        'volume': str(volume),
        'price': str(price),
        'oflags': 'post'
    }
    
    result = kraken_request('/0/private/AddOrder', data, API_KEY, API_SECRET)
    return result

def main():
    print("🔄 Checking current balances...")
    
    # Get balance
    balance = get_balance()
    if 'error' in balance:
        print(f"❌ Error getting balance: {balance['error']}")
        return
    
    print(f"📊 Balance result: {balance}")
    
    # Get current BTC price
    btc_price = get_ticker()
    if not btc_price:
        print("❌ Could not get BTC price")
        return
    
    print(f"💰 Current BTC price: ${btc_price:,.2f}")
    
    # Calculate how much BTC to sell for $20
    target_usd = 20.0
    btc_to_sell = target_usd / btc_price
    
    print(f"🔢 Need to sell {btc_to_sell:.8f} BTC for ${target_usd:.2f}")
    
    # Place sell order
    print("\n🚀 Placing sell order...")
    result = place_sell_order(btc_to_sell, btc_price)
    
    if 'error' in result:
        print(f"❌ Error placing order: {result['error']}")
    else:
        print(f"✅ Order placed successfully: {result}")
        print(f"📋 Order details: {result.get('result', {})}")

if __name__ == "__main__":
    main()
