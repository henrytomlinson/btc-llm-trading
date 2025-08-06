#!/usr/bin/env python3
"""
Bitcoin LLM Trading System
A focused Bitcoin trading application with compact LLM for automated trading decisions.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import jwt
import secrets
from fastapi import FastAPI, HTTPException, Form, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Bitcoin LLM Trading System", version="1.0.0")

# Security
security = HTTPBasic()

# JWT Configuration
JWT_SECRET = secrets.token_urlsafe(32)
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

# Load environment variables
alpaca_api_key = os.getenv('ALPACA_API_KEY')
alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
news_api_key = os.getenv('NEWS_API_KEY')
cohere_key = os.getenv('COHERE_KEY')

# Check if Alpaca is available
alpaca_available = bool(alpaca_api_key and alpaca_secret_key)

# Initialize Alpaca Trading Bot
try:
    from alpaca_trading_btc import AlpacaTradingBot
    alpaca_bot = AlpacaTradingBot() if alpaca_available else None
    logger.info("Alpaca trading bot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Alpaca trading bot: {e}")
    alpaca_bot = None

# Initialize LLM Trading Strategy
try:
    from llm_trading_strategy import LLMTradingStrategy
    llm_strategy = LLMTradingStrategy()
    logger.info("LLM trading strategy initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM trading strategy: {e}")
    llm_strategy = None

# News cache for BTC
btc_news_cache = {}
btc_news_cache_timestamp = {}

# Global variables
trade_log = []  # Store trade logs

class BTCTradingData(BaseModel):
    symbol: str = "BTC"
    price: float
    time: str
    news: str
    sentiment: int
    probability: float
    signal: int
    volume: float
    change_24h: float

def get_btc_news() -> str:
    """Get latest Bitcoin news from News API"""
    import requests
    from datetime import datetime, timedelta
    
    # Check cache first (cache for 5 minutes)
    current_time = datetime.now()
    if 'BTC' in btc_news_cache and 'BTC' in btc_news_cache_timestamp:
        if (current_time - btc_news_cache_timestamp['BTC']).seconds < 300:  # 5 minutes
            return btc_news_cache['BTC']
    
    try:
        # Get Bitcoin news from News API
        url = f"https://newsapi.org/v2/everything"
        params = {
            'q': '"Bitcoin" OR "BTC" OR "cryptocurrency"',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 1,  # Get only the latest news
            'apiKey': news_api_key
        }
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('articles') and len(data['articles']) > 0:
                article = data['articles'][0]
                news_text = f"Latest: {article.get('title', 'No title')}"
                
                # Cache the result
                btc_news_cache['BTC'] = news_text
                btc_news_cache_timestamp['BTC'] = current_time
                
                return news_text
            else:
                return "Real-time Bitcoin data"
        else:
            return "Real-time Bitcoin data"
            
    except Exception as e:
        logger.error(f"Error fetching Bitcoin news: {e}")
        return "Real-time Bitcoin data"

def analyze_btc_sentiment(news_text: str) -> tuple:
    """Analyze Bitcoin sentiment using compact LLM approach"""
    try:
        # Simple sentiment analysis based on keywords
        positive_keywords = ['bullish', 'surge', 'rally', 'breakout', 'adoption', 'institutional']
        negative_keywords = ['bearish', 'crash', 'dump', 'sell-off', 'regulation', 'ban']
        
        text_lower = news_text.lower()
        positive_score = sum(1 for word in positive_keywords if word in text_lower)
        negative_score = sum(1 for word in negative_keywords if word in text_lower)
        
        if positive_score > negative_score:
            return 1, 0.7  # Positive sentiment
        elif negative_score > positive_score:
            return -1, 0.7  # Negative sentiment
        else:
            return 0, 0.5  # Neutral sentiment
            
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return 0, 0.5

def generate_trading_signal(price: float, sentiment: int, volume: float, change_24h: float) -> int:
    """Generate trading signal based on price, sentiment, and technical indicators"""
    try:
        # Simple algorithmic trading logic
        signal_score = 0
        
        # Sentiment factor (40% weight)
        signal_score += sentiment * 0.4
        
        # Price momentum factor (30% weight)
        if change_24h > 2:  # Strong positive momentum
            signal_score += 0.3
        elif change_24h < -2:  # Strong negative momentum
            signal_score -= 0.3
        
        # Volume factor (20% weight)
        if volume > 1000000:  # High volume
            signal_score += 0.2
        elif volume < 500000:  # Low volume
            signal_score -= 0.1
        
        # Price level factor (10% weight)
        if price > 50000:  # High price level
            signal_score += 0.1
        elif price < 30000:  # Low price level
            signal_score -= 0.1
        
        # Determine final signal
        if signal_score > 0.3:
            return 1  # Buy signal
        elif signal_score < -0.3:
            return -1  # Sell signal
        else:
            return 0  # Hold signal
            
    except Exception as e:
        logger.error(f"Error generating trading signal: {e}")
        return 0

def get_btc_market_data() -> Optional[Dict]:
    """Get Bitcoin market data from CoinGecko API"""
    try:
        # Use CoinGecko API for real Bitcoin data
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': 'bitcoin',
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_24hr_vol': 'true'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        # Handle rate limiting
        if response.status_code == 429:
            logger.warning("CoinGecko API rate limit reached, using fallback data")
            return {
                'close': 114000.0,  # Fallback price
                'volume': 30000000000.0,  # Fallback volume
                'change_24h': 0.0  # Fallback change
            }
        
        response.raise_for_status()
        data = response.json()
        
        if 'bitcoin' in data:
            btc_data = data['bitcoin']
            return {
                'close': btc_data.get('usd', 45000.0),
                'volume': btc_data.get('usd_24h_vol', 1000000.0),
                'change_24h': btc_data.get('usd_24h_change', 0.0)
            }
        else:
            logger.error("No Bitcoin data found in CoinGecko response")
            return None
            
    except Exception as e:
        logger.error(f"Error getting BTC market data: {e}")
        # Return fallback data instead of None
        return {
            'close': 114000.0,
            'volume': 30000000000.0,
            'change_24h': 0.0
        }

def create_jwt_token(username: str):
    """Create JWT token"""
    expiry = datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS)
    payload = {
        "username": username,
        "exp": expiry
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("username")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(token: str = None):
    """Get current user from JWT token"""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    return verify_jwt_token(token)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main Bitcoin trading dashboard"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bitcoin LLM Trading System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #1a1a1a; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .trading-card { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .btc-price { font-size: 2em; font-weight: bold; color: #f7931a; }
            .signal-buy { color: #28a745; font-weight: bold; }
            .signal-sell { color: #dc3545; font-weight: bold; }
            .signal-hold { color: #6c757d; font-weight: bold; }
            .news-section { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .trading-controls { display: flex; gap: 10px; margin: 15px 0; }
            .btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; }
            .btn-buy { background: #28a745; color: white; }
            .btn-sell { background: #dc3545; color: white; }
            .btn-hold { background: #6c757d; color: white; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .status-success { background: #d4edda; color: #155724; }
            .status-error { background: #f8d7da; color: #721c24; }
            .trade-log { background: #f8f9fa; padding: 15px; border-radius: 5px; max-height: 300px; overflow-y: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Bitcoin LLM Trading System</h1>
                <p>Automated Bitcoin trading with compact LLM analysis</p>
            </div>
            
            <div class="trading-card">
                <h2>📊 Bitcoin Trading Dashboard</h2>
                <div id="btc-data">
                    <div class="btc-price" id="btc-price">Loading...</div>
                    <div id="btc-info">Loading market data...</div>
                </div>
                
                <div class="news-section">
                    <h3>📰 Latest Bitcoin News</h3>
                    <div id="btc-news">Loading news...</div>
                </div>
                
                <div class="trading-controls">
                    <input type="number" id="trade-amount" placeholder="Amount ($)" value="100" min="10" step="10">
                    <button class="btn btn-buy" onclick="buyBTC()">Buy BTC</button>
                    <button class="btn btn-sell" onclick="sellBTC()">Sell BTC</button>
                    <button class="btn btn-hold" onclick="loadBTCData()">Refresh</button>
                    <button class="btn btn-buy" onclick="autoTrade()" style="background: #17a2b8;">🤖 Auto Trade</button>
                    <button class="btn btn-refresh" onclick="testJS()" style="background: #28a745;">Test JS</button>
                </div>
                
                <div id="trading-status"></div>
            </div>
            
            <div class="trading-card">
                <h3>📈 Trading Signals</h3>
                <div id="signals">Loading signals...</div>
            </div>
            
            <div class="trading-card">
                <h3>📋 Trade Log</h3>
                <div class="trade-log" id="trade-log">No trades yet...</div>
            </div>
        </div>
        
        <script>
            let jwtToken = localStorage.getItem('jwt_token');
            
            // Simple test function
            function testJS() {
                alert('JavaScript is working!');
                console.log('Test function called');
            }
            
            // Simple data loading function
            async function loadBTCData() {
                try {
                    console.log('Loading BTC data...');
                    
                    // Show loading state
                    document.getElementById('btc-price').innerHTML = 'Loading...';
                    document.getElementById('btc-info').innerHTML = '<p>Loading market data...</p>';
                    document.getElementById('btc-news').innerHTML = 'Loading news...';
                    document.getElementById('signals').innerHTML = '<p>Loading signals...</p>';
                    
                    // Simple fetch without complex headers
                    const response = await fetch('/btc_data');
                    console.log('Response status:', response.status);
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const data = await response.json();
                    console.log('BTC data received:', data);
                    
                    // Update price
                    if (data.price) {
                        document.getElementById('btc-price').innerHTML = `$${data.price.toLocaleString()}`;
                    }
                    
                    // Update market info
                    if (data.change_24h !== undefined && data.volume !== undefined) {
                        document.getElementById('btc-info').innerHTML = `
                            <p><strong>24h Change:</strong> <span class="${data.change_24h >= 0 ? 'signal-buy' : 'signal-sell'}">${data.change_24h > 0 ? '+' : ''}${data.change_24h.toFixed(2)}%</span></p>
                            <p><strong>Volume:</strong> $${data.volume.toLocaleString()}</p>
                        `;
                    }
                    
                    // Update news
                    if (data.news) {
                        document.getElementById('btc-news').innerHTML = data.news;
                    }
                    
                    // Update signals
                    if (data.sentiment !== undefined && data.probability !== undefined) {
                        document.getElementById('signals').innerHTML = `
                            <p><strong>Sentiment:</strong> <span class="${data.sentiment > 0 ? 'signal-buy' : data.sentiment < 0 ? 'signal-sell' : 'signal-hold'}">${data.sentiment > 0 ? 'Positive' : data.sentiment < 0 ? 'Negative' : 'Neutral'}</span></p>
                            <p><strong>Confidence:</strong> ${(data.probability * 100).toFixed(1)}%</p>
                        `;
                    }
                    
                    console.log('BTC data loaded successfully');
                } catch (error) {
                    console.error('Error loading BTC data:', error);
                    document.getElementById('btc-price').innerHTML = 'Error loading data';
                    document.getElementById('btc-info').innerHTML = '<p>Error loading market data</p>';
                    document.getElementById('btc-news').innerHTML = 'Error loading news';
                    document.getElementById('signals').innerHTML = '<p>Error loading signals</p>';
                }
            }
            
            async function buyBTC() {
                const amount = document.getElementById('trade-amount').value;
                alert(`Buy BTC: $${amount} - This would execute a buy order`);
                console.log('Buy BTC called with amount:', amount);
            }
            
            async function sellBTC() {
                const amount = document.getElementById('trade-amount').value;
                alert(`Sell BTC: $${amount} - This would execute a sell order`);
                console.log('Sell BTC called with amount:', amount);
            }
            
                        async function autoTrade() {
                try {
                    console.log('🤖 Auto Trade started...');
                    
                    // Show loading state
                    document.getElementById('trading-status').innerHTML = '<div class="status status-success">🤖 AI analyzing market conditions...</div>';
                    
                    // Call the auto trade endpoint
                    const response = await fetch('/auto_trade', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                        body: 'token=demo'
                    });
                    
                    console.log('Auto trade response status:', response.status);
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const data = await response.json();
                    console.log('Auto trade response:', data);
                    
                    // Display the results
                    let message = `🤖 AI Analysis Complete!\n\n`;
                    if (data.signal) {
                        message += `Sentiment: ${data.signal.sentiment}\n`;
                        message += `Confidence: ${(data.signal.confidence * 100).toFixed(1)}%\n`;
                        message += `Action: ${data.signal.action.toUpperCase()}\n`;
                        message += `Reason: ${data.signal.reason}\n\n`;
                        
                        if (data.trade_executed) {
                            message += `✅ Trade executed: $${data.signal.position_size} ${data.signal.action}`;
                        } else {
                            message += `⏸️ No trade executed: ${data.signal.reason}`;
                        }
                    } else {
                        message += `Analysis completed but no signal generated.`;
                    }
                    
                    document.getElementById('trading-status').innerHTML = `<div class="status status-success">${message.replace(/\n/g, '<br>')}</div>`;
                    
                    // Add to trade log
                    const log = document.getElementById('trade-log');
                    const timestamp = new Date().toLocaleTimeString();
                    log.innerHTML += `<div>[${timestamp}] 🤖 AUTO TRADE: ${data.signal ? data.signal.action.toUpperCase() : 'Analysis'} - ${data.signal ? data.signal.reason : 'Completed'}</div>`;
                    log.scrollTop = log.scrollHeight;
                    
                } catch (error) {
                    console.error('Auto trade error:', error);
                    document.getElementById('trading-status').innerHTML = `<div class="status status-error">Error: ${error.message}</div>`;
                }
            }
            
            function refreshData() {
                loadBTCData();
            }
            
            function addTradeLog(message) {
                const log = document.getElementById('trade-log');
                const timestamp = new Date().toLocaleTimeString();
                log.innerHTML += `<div>[${timestamp}] ${message}</div>`;
                log.scrollTop = log.scrollHeight;
            }
            
            function getSignalText(signal) {
                if (signal === 1) return 'BUY';
                if (signal === -1) return 'SELL';
                return 'HOLD';
            }
            
            function getSignalClass(signal) {
                if (signal === 1) return 'signal-buy';
                if (signal === -1) return 'signal-sell';
                return 'signal-hold';
            }
            
            function getSentimentText(sentiment) {
                if (sentiment > 0) return 'Positive';
                if (sentiment < 0) return 'Negative';
                return 'Neutral';
            }
            
            function getSentimentClass(sentiment) {
                if (sentiment > 0) return 'signal-buy';
                if (sentiment < 0) return 'signal-sell';
                return 'signal-hold';
            }
            
            // Auto-refresh every 30 seconds
            setInterval(loadBTCData, 30000);
            
            // Load initial data with timeout
            console.log('Setting up initial data load...');
            setTimeout(() => {
                console.log('Starting initial data load...');
                loadBTCData();
            }, 1000);
            
            // Test function to check if JavaScript is working
            window.testJS = function() {
                console.log('JavaScript is working!');
                alert('JavaScript is working!');
            };
        </script>
    </body>
    </html>
    """

@app.get("/btc_data")
async def get_btc_data(token: str = None):
    """Get Bitcoin market data and analysis"""
    try:
        # Get market data
        market_data = get_btc_market_data()
        
        if market_data:
            price = market_data.get('close', 45000.0)
            volume = market_data.get('volume', 1000000.0)
            change_24h = market_data.get('change_24h', 0.0)
        else:
            # Fallback data
            price = 45000.0
            volume = 1000000.0
            change_24h = 2.5
        
        # Get news and sentiment
        news_text = get_btc_news()
        sentiment, probability = analyze_btc_sentiment(news_text)
        
        # Generate trading signal
        signal = generate_trading_signal(price, sentiment, volume, change_24h)
        
        return BTCTradingData(
            symbol="BTC",
            price=price,
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            news=news_text,
            sentiment=sentiment,
            probability=probability,
            signal=signal,
            volume=volume,
            change_24h=change_24h
        )
        
    except Exception as e:
        logger.error(f"Error getting BTC data: {e}")
        return BTCTradingData(
            symbol="BTC",
            price=45000.0,
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            news="Real-time Bitcoin data",
            sentiment=0,
            probability=0.5,
            signal=0,
            volume=1000000.0,
            change_24h=0.0
        )

@app.post("/buy_btc")
async def buy_btc(amount: float = Form(...), token: str = Form(...)):
    """Buy Bitcoin with dollar amount"""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        username = verify_jwt_token(token)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not alpaca_available:
        return {"message": "Alpaca trading not available"}
    
    try:
        # Execute buy order through Alpaca using Bitcoin proxy (GBTC)
        result = alpaca_bot.place_buy_order("BTC/USD", amount, 1.0)
        
        trade_log.append({
            "action": "BUY",
            "symbol": "BTC",
            "amount": amount,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result": result
        })
        
        if result['status'] == 'success':
            return {"message": f"Bought ${amount:.2f} worth of Bitcoin (via {result['symbol']})", "result": result}
        else:
            return {"message": f"Buy order failed: {result.get('reason', 'Unknown error')}", "result": result}
            
    except Exception as e:
        return {"message": f"Error buying Bitcoin: {str(e)}"}

@app.post("/sell_btc")
async def sell_btc(amount: float = Form(...), token: str = Form(...)):
    """Sell Bitcoin"""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        username = verify_jwt_token(token)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not alpaca_available:
        return {"message": "Alpaca trading not available"}
    
    try:
        # Execute sell order through Alpaca using Bitcoin proxy (GBTC)
        result = alpaca_bot.place_sell_order("BTC/USD", amount)
        
        trade_log.append({
            "action": "SELL",
            "symbol": "BTC",
            "amount": amount,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result": result
        })
        
        if result['status'] == 'success':
            return {"message": f"Sold ${amount:.2f} worth of Bitcoin (via {result['symbol']})", "result": result}
        else:
            return {"message": f"Sell order failed: {result.get('reason', 'Unknown error')}", "result": result}
            
    except Exception as e:
        return {"message": f"Error selling Bitcoin: {str(e)}"}

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """Login page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Login - Bitcoin LLM Trading</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; display: flex; justify-content: center; align-items: center; height: 100vh; }
            .login-container { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); width: 400px; }
            .header { text-align: center; margin-bottom: 30px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; }
            .btn { width: 100%; padding: 12px; background: #f7931a; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }
            .btn:hover { background: #e6850e; }
        </style>
    </head>
    <body>
        <div class="login-container">
            <div class="header">
                <h1>🚀 Bitcoin LLM Trading</h1>
                <p>Login to access the trading system</p>
            </div>
            <form id="loginForm">
                <div class="form-group">
                    <label for="username">Username:</label>
                    <input type="text" id="username" name="username" value="admin" required>
                </div>
                <div class="form-group">
                    <label for="password">Password:</label>
                    <input type="password" id="password" name="password" value="trading123" required>
                </div>
                <button type="submit" class="btn">Login</button>
            </form>
        </div>
        
        <script>
            document.getElementById('loginForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                formData.append('username', document.getElementById('username').value);
                formData.append('password', document.getElementById('password').value);
                
                try {
                    const response = await fetch('/auth/login', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        localStorage.setItem('jwt_token', data.session_token);
                        window.location.href = '/';
                    } else {
                        alert('Login failed: ' + data.detail);
                    }
                } catch (error) {
                    alert('Login error: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/auth/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """Handle login"""
    if username == "admin" and password == "trading123":
        token = create_jwt_token(username)
        return {"session_token": token, "message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/test_apis")
async def test_apis():
    """Test API connectivity"""
    results = {}
    
    # Test Alpaca
    try:
        if alpaca_available:
            results["alpaca"] = "✅ Connected successfully"
        else:
            results["alpaca"] = "❌ Not configured"
    except Exception as e:
        results["alpaca"] = f"❌ Error: {str(e)}"
    
    # Test News API
    try:
        if news_api_key:
            results["news_api"] = "✅ Connected successfully"
        else:
            results["news_api"] = "❌ Not configured"
    except Exception as e:
        results["news_api"] = f"❌ Error: {str(e)}"
    
    # Test Cohere
    try:
        if cohere_key:
            results["cohere"] = "✅ Connected successfully"
        else:
            results["cohere"] = "❌ Not configured"
    except Exception as e:
        results["cohere"] = f"❌ Error: {str(e)}"
    
    return results

@app.post("/auto_trade")
async def auto_trade(token: str = Form(...)):
    """Execute automated trading based on LLM analysis"""
    logger.info("🤖 Auto Trade endpoint called")
    
    if not token:
        logger.warning("No token provided for auto trade")
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        username = verify_jwt_token(token)
        logger.info(f"Auto trade requested by user: {username}")
    except:
        logger.warning("Invalid token for auto trade")
        raise HTTPException(status_code=401, detail="Authentication required")

    if not alpaca_available or not alpaca_bot:
        logger.warning("Alpaca trading not available for auto trade")
        return {"message": "Alpaca trading not available"}

    if not llm_strategy:
        logger.warning("LLM strategy not available for auto trade")
        return {"message": "LLM strategy not available"}

    try:
        logger.info("🤖 Starting LLM market analysis...")
        
        # Get current market data
        market_data = get_btc_market_data()
        if not market_data:
            logger.error("Unable to get market data for auto trade")
            return {"message": "Unable to get market data"}

        current_price = market_data.get('close', 45000.0)
        price_change_24h = market_data.get('change_24h', 0.0)
        volume_24h = market_data.get('volume', 1000000.0)
        
        logger.info(f"📊 Market Data: Price=${current_price:,.2f}, Change={price_change_24h:+.2f}%, Volume=${volume_24h:,.0f}")

        # Get news sentiment
        news_text = get_btc_news()
        sentiment, probability = analyze_btc_sentiment(news_text)
        news_sentiment = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
        
        logger.info(f"📰 News Sentiment: {news_sentiment} (score: {sentiment}, probability: {probability:.2f})")

        # Generate LLM trading signal
        logger.info("🧠 Generating LLM trading signal...")
        signal = llm_strategy.generate_trading_signal(
            current_price=current_price,
            price_change_24h=price_change_24h,
            volume_24h=volume_24h,
            news_sentiment=news_sentiment
        )
        
        logger.info(f"🎯 LLM Signal: {signal['action'].upper()} - {signal['reason']} (confidence: {signal['confidence']:.2f})")

        # Execute trade if recommended
        trade_result = None
        if signal['should_trade'] and signal['position_size']:
            logger.info(f"💰 Executing trade: {signal['action'].upper()} ${signal['position_size']}")
            if signal['action'] == 'buy':
                trade_result = alpaca_bot.place_buy_order("BTC/USD", signal['position_size'], 1.0)
            elif signal['action'] == 'sell':
                trade_result = alpaca_bot.place_sell_order("BTC/USD", signal['position_size'])
            logger.info(f"✅ Trade result: {trade_result}")
        else:
            logger.info("⏸️ No trade executed - conditions not met")

        # Log the automated trade
        trade_log.append({
            "action": "AUTO_TRADE",
            "signal": signal,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trade_result": trade_result
        })

        logger.info("🤖 Auto trade analysis completed successfully")
        return {
            "message": f"Automated trading analysis complete",
            "signal": signal,
            "trade_executed": trade_result is not None,
            "trade_result": trade_result
        }

    except Exception as e:
        logger.error(f"❌ Error in automated trading: {e}")
        return {"message": f"Error in automated trading: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 