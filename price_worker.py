import threading
import time
from datetime import datetime, timezone
from collections import deque

PRICE_CACHE = {"price": None, "ts": None, "source": None}

class VolBucket:
    """Volatility bucket for tracking rolling price volatility"""
    def __init__(self, n=48):  # ~12h of 15m bars
        self.returns = deque(maxlen=n)
        self.last_mid = None
    
    def update(self, mid):
        if self.last_mid:
            r = (mid / self.last_mid) - 1.0
            self.returns.append(r)
        self.last_mid = mid
    
    def sigma(self):
        if len(self.returns) < 5:
            return None
        m = sum(self.returns) / len(self.returns)
        var = sum((x - m) * (x - m) for x in self.returns) / len(self.returns)
        return var ** 0.5

# Global volatility tracker
VOL_BUCKET = VolBucket()

def _set_price(p, source):
    price = float(p)
    PRICE_CACHE["price"] = price
    PRICE_CACHE["ts"] = datetime.now(timezone.utc)
    PRICE_CACHE["source"] = source
    
    # Update volatility tracker with new price
    VOL_BUCKET.update(price)

def kraken_ws_loop():
    import websocket
    import json
    url = "wss://ws.kraken.com"
    while True:
        try:
            ws = websocket.create_connection(url, timeout=15)
            ws.send(json.dumps({"event":"subscribe","pair":["XBT/USD"],"subscription":{"name":"ticker"}}))
            while True:
                msg = json.loads(ws.recv())
                if isinstance(msg, list) and len(msg) > 1 and "c" in msg[1]:  # last trade price
                    _set_price(msg[1]["c"][0], "kraken_ws")
        except Exception:
            time.sleep(2)  # reconnect backoff

def rest_fallback_loop():
    import requests
    import math
    while True:
        time.sleep(5)
        try:
            # try Kraken REST
            r = requests.get("https://api.kraken.com/0/public/Ticker?pair=XXBTZUSD", timeout=5)
            j = r.json()
            p = j["result"]["XXBTZUSD"]["c"][0]
            _set_price(p, "kraken_rest")
        except Exception:
            # try a second source (e.g., Coinbase)
            try:
                r = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/ticker", timeout=5)
                p = r.json()["price"]
                _set_price(p, "coinbase_rest")
            except Exception:
                pass

def start_price_worker():
    threading.Thread(target=kraken_ws_loop, daemon=True).start()
    threading.Thread(target=rest_fallback_loop, daemon=True).start()

def get_last_price():
    p, ts = PRICE_CACHE["price"], PRICE_CACHE["ts"]
    if p is None or ts is None:
        raise RuntimeError("price unavailable")
    return float(p), ts

def get_current_volatility():
    """Get current volatility (sigma) from the volatility tracker"""
    return VOL_BUCKET.sigma()
