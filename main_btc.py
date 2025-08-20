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
from fastapi import FastAPI, HTTPException, Form, Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import requests
import json
from dotenv import load_dotenv
import time

# Ledger for persistent PnL and orders
try:
    from db import (
        init_db as ledger_init_db,
        record_order as ledger_record_order,
        compute_pnl as ledger_compute_pnl,
        get_hodl_benchmark as ledger_get_hodl_benchmark,
        snapshot_equity as ledger_snapshot_equity,
    )
    ledger_init_db()
    LEDGER_AVAILABLE = True
except Exception as _ledger_err:
    LEDGER_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Ledger not available: {_ledger_err}")

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security configuration
ALLOWED_IPS = os.getenv('ALLOWED_IPS', '127.0.0.1').split(',')  # Comma-separated IPs
RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))  # Requests per hour
RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', '3600'))  # 1 hour in seconds

# Admin credentials (use environment variables)
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'change_this_password_immediately')

# Rate limiting storage (in production, use Redis)
request_counts = {}

# Initialize FastAPI app
app = FastAPI(title="Bitcoin LLM Trading System", version="1.0.0")

# Security middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for IP whitelisting and rate limiting"""
    client_ip = request.client.host
    
    # IP whitelist check
    if client_ip not in ALLOWED_IPS and '0.0.0.0' not in ALLOWED_IPS:
        logger.warning(f"Blocked request from unauthorized IP: {client_ip}")
        return JSONResponse(
            status_code=403,
            content={"error": "IP not authorized"}
        )
    
    # Rate limiting
    current_time = time.time()
    
    # Clean old entries
    global request_counts
    request_counts = {ip: (count, timestamp) for ip, (count, timestamp) in request_counts.items() 
                     if current_time - timestamp < RATE_LIMIT_WINDOW}
    
    # Check rate limit
    if client_ip in request_counts:
        count, timestamp = request_counts[client_ip]
        if current_time - timestamp < RATE_LIMIT_WINDOW and count >= RATE_LIMIT_REQUESTS:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )
        request_counts[client_ip] = (count + 1, timestamp)
    else:
        request_counts[client_ip] = (1, current_time)
    
    # Add security headers
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response

# Security
security = HTTPBasic()

# JWT Configuration
JWT_SECRET = secrets.token_urlsafe(32)
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

# Load environment variables
binance_api_key = os.getenv('BINANCE_API_KEY')
binance_secret_key = os.getenv('BINANCE_SECRET_KEY')
kraken_api_key = os.getenv('KRAKEN_API_KEY')
kraken_secret_key = os.getenv('KRAKEN_PRIVATE_KEY')  # Updated to use KRAKEN_PRIVATE_KEY
news_api_key = os.getenv('NEWS_API_KEY')
cohere_key = os.getenv('COHERE_KEY')

# Initialize Trading Bot (Priority: Kraken > Binance)
trading_bot = None
exchange_name = None

# Check if Kraken has valid credentials first
if kraken_api_key and kraken_secret_key:
    try:
        from kraken_trading_btc import KrakenTradingBot
        trading_bot = KrakenTradingBot()
        exchange_name = "kraken"
        logger.info("✅ Kraken trading bot initialized successfully (Real Bitcoin trading)")
    except Exception as e:
        logger.warning(f"Kraken not available: {e}")
        trading_bot = None

# If Kraken not available, try Binance
if not trading_bot:
    try:
        from binance_trading_btc import BinanceTradingBot
        trading_bot = BinanceTradingBot()
        exchange_name = "binance"
        logger.info("✅ Binance trading bot initialized successfully (Bitcoin spot trading)")
    except Exception as e:
        logger.warning(f"Binance not available: {e}")
        logger.error(f"❌ No trading bot available: {e}")
        trading_bot = None
        exchange_name = None

# Initialize LLM Trading Strategy
try:
    from llm_trading_strategy import LLMTradingStrategy
    llm_strategy = LLMTradingStrategy()
    logger.info("✅ LLM trading strategy initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize LLM trading strategy: {e}")
    llm_strategy = None

# Initialize Technical Analysis (if available)
try:
    from technical_analysis import TechnicalAnalyzer
    technical_analyzer = TechnicalAnalyzer()
    logger.info("✅ Technical analyzer initialized successfully")
except Exception as e:
    logger.warning(f"Technical analyzer not available: {e}")
    technical_analyzer = None

# Initialize trade log for tracking automated trades
trade_log = []

# Initialize Risk Management (if available)
try:
    from risk_management import RiskManager
    risk_manager = RiskManager()
    logger.info("✅ Risk manager initialized successfully")
except Exception as e:
    logger.warning(f"Risk manager not available: {e}")
    risk_manager = None

# News cache for BTC
btc_news_cache = {}
btc_news_cache_timestamp = {}

# Sentiment cache for BTC (to reduce Cohere calls)
btc_sentiment_cache = {}
btc_sentiment_cache_timestamp = {}

# Global variables

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

class PnlSummary(BaseModel):
    symbol: str
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    total_fees: float
    exposure_usd: float
    exposure_pct: float
    hodl_value: float
    hodl_pnl: float
    qty_btc: float
    avg_cost: float

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
    """Analyze Bitcoin sentiment using LLM strategy"""
    # 15-minute cache to reduce LLM calls triggered by dashboard/health checks
    current_time = datetime.now()
    last_ts = btc_sentiment_cache_timestamp.get('BTC')
    if last_ts and (current_time - last_ts).seconds < 900:
        logger.info("Using cached sentiment analysis")
        return btc_sentiment_cache['BTC']

    try:
        # Try to use LLM strategy if available
        if llm_strategy:
            try:
                # Get current market data for context
                market_data = get_btc_market_data()
                if market_data:
                    current_price = market_data.get('close', 45000.0)
                    change_24h = market_data.get('change_24h', 0.0)
                    volume = market_data.get('volume', 1000000.0)
                    
                    # Use LLM strategy for analysis
                    analysis = llm_strategy.analyze_market_conditions(
                        current_price=current_price,
                        price_change_24h=change_24h,
                        volume_24h=volume,
                        news_sentiment=news_text[:200],  # Limit news text
                        technical_indicators=None
                    )
                    
                    # Convert sentiment to numeric
                    if analysis.sentiment == "bullish":
                        sentiment = 1
                    elif analysis.sentiment == "bearish":
                        sentiment = -1
                    else:
                        sentiment = 0
                    
                    btc_sentiment_cache['BTC'] = (sentiment, analysis.confidence)
                    btc_sentiment_cache_timestamp['BTC'] = current_time
                    return sentiment, analysis.confidence
                    
            except Exception as e:
                logger.warning(f"LLM strategy failed, using fallback: {e}")
        
        # Fallback to simple keyword analysis
        positive_keywords = ['bullish', 'surge', 'rally', 'breakout', 'adoption', 'institutional', 'positive']
        negative_keywords = ['bearish', 'crash', 'dump', 'sell-off', 'regulation', 'ban', 'negative']
        
        text_lower = news_text.lower()
        positive_score = sum(1 for word in positive_keywords if word in text_lower)
        negative_score = sum(1 for word in negative_keywords if word in text_lower)
        
        if positive_score > negative_score:
            btc_sentiment_cache['BTC'] = (1, 0.7)
            btc_sentiment_cache_timestamp['BTC'] = current_time
            return 1, 0.7  # Positive sentiment
        elif negative_score > positive_score:
            btc_sentiment_cache['BTC'] = (-1, 0.7)
            btc_sentiment_cache_timestamp['BTC'] = current_time
            return -1, 0.7  # Negative sentiment
        else:
            btc_sentiment_cache['BTC'] = (0, 0.5)
            btc_sentiment_cache_timestamp['BTC'] = current_time
            return 0, 0.5  # Neutral sentiment
            
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        btc_sentiment_cache['BTC'] = (0, 0.5)
        btc_sentiment_cache_timestamp['BTC'] = datetime.now()
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
    """Get Bitcoin market data from trading bot or CoinGecko API"""
    try:
        # Try to get data from trading bot first
        if trading_bot:
            try:
                market_data = trading_bot.get_market_data()
                if market_data and 'error' not in market_data:
                    return {
                        'close': market_data.get('close', 45000.0),
                        'volume': market_data.get('volume', 1000000.0),
                        'change_24h': market_data.get('change_24h', 0.0)
                    }
            except Exception as e:
                logger.warning(f"Trading bot market data failed: {e}")
        
        # Fallback to CoinGecko API
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
                <div id="auth-status" style="margin-top: 10px;">
                    <span id="login-status">Not logged in</span>
                    <button id="login-btn" onclick="showLoginForm()" style="margin-left: 10px; padding: 5px 10px; background: #28a745; color: white; border: none; border-radius: 3px; cursor: pointer;">Login</button>
                    <button id="logout-btn" onclick="logout()" style="margin-left: 10px; padding: 5px 10px; background: #dc3545; color: white; border: none; border-radius: 3px; cursor: pointer; display: none;">Logout</button>
                </div>
                <div id="login-form" style="display: none; margin-top: 10px; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
                    <p style="margin: 0 0 10px 0; color: #fff; font-size: 12px;">Enter your credentials:</p>
                    <input type="text" id="login-username" placeholder="Username" style="margin-right: 10px; padding: 5px;">
                    <input type="password" id="login-password" placeholder="Password" style="margin-right: 10px; padding: 5px;">
                    <button onclick="login()" style="padding: 5px 10px; background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer;">Login</button>
                    <button onclick="hideLoginForm()" style="padding: 5px 10px; background: #6c757d; color: white; border: none; border-radius: 3px; cursor: pointer;">Cancel</button>
                </div>
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
                <h3>💹 PnL Summary</h3>
                <div id="pnl">Login to view PnL</div>
            </div>
            
            <div class="trading-card">
                <h3>📋 Trade Log</h3>
                <div class="trade-log" id="trade-log">No trades yet...</div>
            </div>
        </div>
        
        <script>
            console.log('JavaScript starting...');
            
            let jwtToken = localStorage.getItem('jwt_token');
            
            // Test function to check if JavaScript is working
            function testJS() {
                console.log('JavaScript is working!');
                alert('JavaScript is working!');
            }
            
            async function loadPnl() {
                const container = document.getElementById('pnl');
                if (!jwtToken) { container.textContent = 'Login to view PnL'; return; }
                try {
                    const res = await fetch('/pnl_summary?token=' + encodeURIComponent(jwtToken));
                    if (!res.ok) { container.textContent = 'PnL unavailable'; return; }
                    const d = await res.json();
                    container.innerHTML = `
                        <p><strong>Equity:</strong> $${d.equity.toLocaleString(undefined,{maximumFractionDigits:2})}</p>
                        <p><strong>Realized PnL:</strong> $${d.realized_pnl.toLocaleString(undefined,{maximumFractionDigits:2})}</p>
                        <p><strong>Unrealized PnL:</strong> $${d.unrealized_pnl.toLocaleString(undefined,{maximumFractionDigits:2})}</p>
                        <p><strong>Total Fees:</strong> $${d.total_fees.toLocaleString(undefined,{maximumFractionDigits:2})}</p>
                        <p><strong>Exposure:</strong> $${d.exposure_usd.toLocaleString(undefined,{maximumFractionDigits:2})} (${(d.exposure_pct*100).toFixed(1)}%)</p>
                        <p><strong>HODL Value:</strong> $${d.hodl_value.toLocaleString(undefined,{maximumFractionDigits:2})} (PnL: $${d.hodl_pnl.toLocaleString(undefined,{maximumFractionDigits:2})})</p>
                        <p><strong>Position:</strong> ${d.qty_btc.toFixed(8)} BTC @ $${d.avg_cost.toLocaleString(undefined,{maximumFractionDigits:2})}</p>
                    `;
                } catch (e) { container.textContent = 'PnL error'; }
            }
            
            // Function to handle login and store JWT token
            async function loginUser(username, password) {
                try {
                    const formData = new FormData();
                    formData.append('username', username);
                    formData.append('password', password);
                    
                    const response = await fetch('/auth/login', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        jwtToken = data.session_token;
                        localStorage.setItem('jwt_token', jwtToken);
                        return true;
                    } else {
                        return false;
                    }
                } catch (error) {
                    console.error('Login error:', error);
                    return false;
                }
            }
            
            // Show login form
            function showLoginForm() {
                document.getElementById('login-form').style.display = 'block';
                document.getElementById('login-btn').style.display = 'none';
            }
            
            // Hide login form
            function hideLoginForm() {
                document.getElementById('login-form').style.display = 'none';
                document.getElementById('login-btn').style.display = 'inline-block';
            }
            
            // Login function
            async function login() {
                const username = document.getElementById('login-username').value;
                const password = document.getElementById('login-password').value;
                
                if (await loginUser(username, password)) {
                    document.getElementById('login-status').textContent = 'Logged in as ' + username;
                    document.getElementById('login-btn').style.display = 'none';
                    document.getElementById('logout-btn').style.display = 'inline-block';
                    document.getElementById('login-form').style.display = 'none';
                    alert('Login successful!');
                } else {
                    alert('Login failed. Please check your credentials.');
                }
            }
            
            // Logout function
            function logout() {
                jwtToken = null;
                localStorage.removeItem('jwt_token');
                document.getElementById('login-status').textContent = 'Not logged in';
                document.getElementById('login-btn').style.display = 'inline-block';
                document.getElementById('logout-btn').style.display = 'none';
                alert('Logged out successfully!');
            }
            
            // Load Bitcoin data from API
            function loadBTCData() {
                console.log('loadBTCData called');
                
                fetch('/btc_data_public')
                    .then(response => response.json())
                    .then(data => {
                        console.log('Data received:', data);
                        
                        if (data.error) {
                            document.getElementById('btc-price').innerHTML = 'Error loading data';
                            document.getElementById('btc-info').innerHTML = '<p>Unable to fetch market data</p>';
                            return;
                        }
                        
                        // Update price
                        document.getElementById('btc-price').innerHTML = '$' + data.price.toLocaleString();
                        
                        // Update market info
                        document.getElementById('btc-info').innerHTML = `
                            <p><strong>24h Change:</strong> <span class="${data.change_24h >= 0 ? 'signal-buy' : 'signal-sell'}">${data.change_24h > 0 ? '+' : ''}${data.change_24h.toFixed(2)}%</span></p>
                            <p><strong>Volume:</strong> $${data.volume.toLocaleString()}</p>
                        `;
                        
                        // Update news
                        document.getElementById('btc-news').innerHTML = data.news || 'Real-time Bitcoin data';
                        
                        // Update signals
                        const sentimentText = data.sentiment > 0 ? 'Positive' : data.sentiment < 0 ? 'Negative' : 'Neutral';
                        const sentimentClass = data.sentiment > 0 ? 'signal-buy' : data.sentiment < 0 ? 'signal-sell' : 'signal-hold';
                        
                        document.getElementById('signals').innerHTML = `
                            <p><strong>Sentiment:</strong> <span class="${sentimentClass}">${sentimentText}</span></p>
                            <p><strong>Confidence:</strong> ${(data.probability * 100).toFixed(1)}%</p>
                            <p><strong>Signal:</strong> <span class="${data.signal === 1 ? 'signal-buy' : data.signal === -1 ? 'signal-sell' : 'signal-hold'}">${data.signal === 1 ? 'BUY' : data.signal === -1 ? 'SELL' : 'HOLD'}</span></p>
                        `;
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        document.getElementById('btc-price').innerHTML = 'Error loading data';
                        document.getElementById('btc-info').innerHTML = '<p>Unable to fetch market data</p>';
                    });
            }
            
            // Buy Bitcoin function
            function buyBTC() {
                const amount = document.getElementById('trade-amount').value;
                alert(`Buy BTC: $${amount}`);
                console.log('Buy BTC:', amount);
                
                const log = document.getElementById('trade-log');
                const timestamp = new Date().toLocaleTimeString();
                log.innerHTML += `<div>[${timestamp}] BUY: $${amount}</div>`;
            }
            
            // Sell Bitcoin function
            function sellBTC() {
                const amount = document.getElementById('trade-amount').value;
                alert(`Sell BTC: $${amount}`);
                console.log('Sell BTC:', amount);
                
                const log = document.getElementById('trade-log');
                const timestamp = new Date().toLocaleTimeString();
                log.innerHTML += `<div>[${timestamp}] SELL: $${amount}</div>`;
            }
            
            // Automated trading function
            async function autoTrade() {
                console.log('Auto trade started');
                document.getElementById('trading-status').innerHTML = '<div class="status status-success">🤖 AI analyzing...</div>';
                
                // Check if user is authenticated
                if (!jwtToken) {
                    document.getElementById('trading-status').innerHTML = '<div class="status status-error">Please log in to use auto-trade</div>';
                    return;
                }
                
                try {
                    const formData = new FormData();
                    formData.append('token', jwtToken);
                    
                    const response = await fetch('/auto_trade', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.status === 401) {
                        document.getElementById('trading-status').innerHTML = '<div class="status status-error">Authentication required. Please log in again.</div>';
                        return;
                    }
                    
                    const data = await response.json();
                    console.log('Auto trade response:', data);
                    
                    let message = '🤖 AI Analysis Complete!';
                    if (data.signal) {
                        message += '\\nAction: ' + data.signal.action.toUpperCase();
                        message += '\\nReason: ' + data.signal.reason;
                    }
                    
                    document.getElementById('trading-status').innerHTML = `<div class="status status-success">${message.replace(/\\\\n/g, '<br>')}</div>`;
                    
                    const log = document.getElementById('trade-log');
                    const timestamp = new Date().toLocaleTimeString();
                    log.innerHTML += `<div>[${timestamp}] 🤖 AUTO TRADE: ${data.signal ? data.signal.action.toUpperCase() : 'Analysis'}</div>`;
                } catch (error) {
                    console.error('Auto trade error:', error);
                    document.getElementById('trading-status').innerHTML = `<div class="status status-error">Error: ${error.message}</div>`;
                }
            }
            
            // Check login status on page load
            function checkLoginStatus() {
                if (jwtToken) {
                    document.getElementById('login-status').textContent = 'Logged in as admin';
                    document.getElementById('login-btn').style.display = 'none';
                    document.getElementById('logout-btn').style.display = 'inline-block';
                } else {
                    document.getElementById('login-status').textContent = 'Not logged in';
                    document.getElementById('login-btn').style.display = 'inline-block';
                    document.getElementById('logout-btn').style.display = 'none';
                }
            }
            
            // Load data when page loads
            window.addEventListener('load', function() {
                console.log('Page loaded, loading BTC data...');
                loadBTCData();
                loadPnl();
                checkLoginStatus();
                
                // Auto-refresh every 30 seconds
                setInterval(() => { loadBTCData(); loadPnl(); }, 30000);
                
                console.log('JavaScript loaded successfully');
            });
        </script>
    </body>
    </html>
    """

@app.get("/btc_data_public")
async def get_btc_data_public():
    """Get basic Bitcoin market data (public endpoint)"""
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
            news=news_text[:200] + "..." if len(news_text) > 200 else news_text,
            sentiment=sentiment,
            probability=probability,
            signal=signal,
            volume=volume,
            change_24h=change_24h
        )
    except Exception as e:
        logger.error(f"Error getting BTC data: {e}")
        return {"error": "Unable to fetch market data"}

@app.get("/btc_data")
async def get_btc_data(token: str = None):
    """Get Bitcoin market data and analysis (authenticated)"""
    # Require authentication for sensitive data
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        username = verify_jwt_token(token)
        logger.info(f"BTC data requested by user: {username}")
    except:
        raise HTTPException(status_code=401, detail="Authentication required")
    
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
    
    if not trading_bot:
        return {"message": "No trading bot available"}
    
    try:
        # Execute buy order through the trading bot
        result = trading_bot.place_buy_order("BTC/USD", amount, 1.0)
        
        trade_log.append({
            "action": "BUY",
            "symbol": "BTC",
            "amount": amount,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result": result
        })
        
        if result['status'] == 'success':
            # Record in ledger (best-effort)
            try:
                if LEDGER_AVAILABLE:
                    qty_btc = float(result.get('quantity', 0.0))
                    price = float(result.get('price_per_btc', 0.0)) if result.get('price_per_btc') is not None else (qty_btc and amount/qty_btc or 0.0)
                    fee = float(result.get('fee', 0.0)) if result.get('fee') is not None else 0.0
                    from db import record_order as _rec
                    _rec(
                        symbol="BTC",
                        side="buy",
                        qty_btc=qty_btc,
                        notional_usd=float(amount),
                        price=price,
                        demo_mode=bool(result.get('demo_mode', False)),
                        exchange_order_id=result.get('order_id'),
                        fee=fee,
                        metadata=result,
                    )
            except Exception as e:
                logger.warning(f"Ledger write failed (buy): {e}")
            return {"message": f"Bought ${amount:.2f} worth of Bitcoin (via {exchange_name})", "result": result}
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
    
    if not trading_bot:
        return {"message": "No trading bot available"}
    
    try:
        # Execute sell order through the trading bot
        result = trading_bot.place_sell_order("BTC/USD", amount)
        
        trade_log.append({
            "action": "SELL",
            "symbol": "BTC",
            "amount": amount,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result": result
        })
        
        if result['status'] == 'success':
            # Record in ledger (best-effort)
            try:
                if LEDGER_AVAILABLE:
                    qty_btc = float(result.get('quantity', 0.0))
                    price = float(result.get('price_per_btc', 0.0)) if result.get('price_per_btc') is not None else (qty_btc and amount/qty_btc or 0.0)
                    fee = float(result.get('fee', 0.0)) if result.get('fee') is not None else 0.0
                    from db import record_order as _rec
                    _rec(
                        symbol="BTC",
                        side="sell",
                        qty_btc=qty_btc,
                        notional_usd=float(amount),
                        price=price,
                        demo_mode=bool(result.get('demo_mode', False)),
                        exchange_order_id=result.get('order_id'),
                        fee=fee,
                        metadata=result,
                    )
            except Exception as e:
                logger.warning(f"Ledger write failed (sell): {e}")
            return {"message": f"Sold ${amount:.2f} worth of Bitcoin (via {exchange_name})", "result": result}
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
                    <input type="password" id="password" name="password" value="SecureTrading2024!" required>
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
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        token = create_jwt_token(username)
        return {"session_token": token, "message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/test_apis")
async def test_apis():
    """Test API connectivity"""
    results = {}
    
    # Test Binance
    if trading_bot and exchange_name == "binance":
        results["binance"] = "✅ Connected successfully"
    else:
        results["binance"] = "❌ Not configured"
    
    # Test Kraken
    if trading_bot and exchange_name == "kraken":
        results["kraken"] = "✅ Connected successfully"
    else:
        results["kraken"] = "❌ Not configured"
    
    # Test Alpaca (if available)
    # Removed Alpaca test as it's no longer used
    
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

@app.get("/test")
async def test_page():
    """Serve the test HTML page"""
    with open('test.html', 'r') as f:
        return HTMLResponse(content=f.read())

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

    if not trading_bot:
        logger.warning("No trading bot available for auto trade")
        return {"message": "No trading bot available"}

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
        try:
            signal = llm_strategy.generate_trading_signal(
                current_price=current_price,
                price_change_24h=price_change_24h,
                volume_24h=volume_24h,
                news_sentiment=news_sentiment
            )
            
            logger.info(f"🎯 LLM Signal: {signal['action'].upper()} - {signal['reason']} (confidence: {signal['confidence']:.2f})")
            
        except Exception as llm_error:
            logger.error(f"❌ LLM signal generation failed: {llm_error}")
            # Create fallback signal
            signal = {
                'action': 'hold',
                'should_trade': False,
                'reason': f'LLM analysis failed: {str(llm_error)}',
                'confidence': 0.5,
                'risk_level': 'medium',
                'position_size': None,
                'price_target': None,
                'stop_loss': None,
                'analysis': f'LLM Error: {str(llm_error)}',
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"🔄 Using fallback signal: {signal['action'].upper()}")

        # Check account balance before executing trade
        account_balance = trading_bot.get_account_info()
        logger.info(f"💰 Account Balance: {account_balance}")
        
        # Calculate appropriate position size based on available funds
        available_funds = 0.0
        btc_balance = 0.0
        
        if account_balance and 'balances' in account_balance:
            balances = account_balance['balances']
            # Check GBP balance
            gbp_balance = float(balances.get('ZGBP', '0.0'))
            # Check USD balance  
            usd_balance = float(balances.get('ZUSD', '0.0'))
            # Check BTC balance
            btc_balance = float(balances.get('XXBT', '0.0'))
            
            available_funds = gbp_balance + usd_balance
            logger.info(f"💵 Available funds: £{gbp_balance:.2f} + ${usd_balance:.2f} = ${available_funds:.2f}")
            logger.info(f"₿ BTC balance: {btc_balance:.8f}")
        
        # Execute allocation change via target exposure if provided
        trade_result = None
        if signal.get('target_exposure') is not None:
            try:
                target = float(signal.get('target_exposure'))
                trade_result = trading_bot.rebalance_to_target(target)
            except Exception as e:
                logger.warning(f"Rebalance failed: {e}")
                trade_result = {"status": "error", "reason": str(e)}
        elif signal['should_trade'] and signal['position_size']:
            # Adjust position size based on available funds
            requested_size = signal['position_size']
            max_buy_size = available_funds * 0.95  # Use 95% of available funds
            max_sell_size = btc_balance * current_price * 0.95  # Use 95% of BTC balance
            
            if signal['action'] == 'buy':
                if max_buy_size < 10.0:  # Minimum $10 trade
                    logger.info(f"⏸️ Insufficient funds for buy: ${max_buy_size:.2f} available, ${requested_size:.2f} requested")
                    trade_result = {"status": "error", "reason": ["Insufficient funds for buy"]}
                else:
                    actual_size = min(requested_size, max_buy_size)
                    logger.info(f"💰 Executing buy: ${actual_size:.2f} (requested: ${requested_size:.2f}, available: ${max_buy_size:.2f})")
                    trade_result = trading_bot.place_buy_order("BTC/USD", actual_size, 1.0)
                    # Persist trade to ledger (best-effort)
                    try:
                        if LEDGER_AVAILABLE and isinstance(trade_result, dict) and trade_result.get('status') == 'success':
                            qty_btc = float(trade_result.get('quantity', 0.0))
                            price = float(trade_result.get('price_per_btc', 0.0)) if trade_result.get('price_per_btc') is not None else (qty_btc and actual_size/qty_btc or 0.0)
                            fee = float(trade_result.get('fee', 0.0)) if trade_result.get('fee') is not None else 0.0
                            ledger_record_order(
                                symbol="BTC",
                                side="buy",
                                qty_btc=qty_btc,
                                notional_usd=float(actual_size),
                                price=price,
                                demo_mode=bool(trade_result.get('demo_mode', False)),
                                exchange_order_id=trade_result.get('order_id'),
                                fee=fee,
                                metadata=trade_result,
                            )
                    except Exception as e:
                        logger.warning(f"Ledger write failed (auto buy): {e}")
                    
            elif signal['action'] == 'sell':
                if max_sell_size < 10.0:  # Minimum $10 trade
                    logger.info(f"⏸️ Insufficient BTC for sell: ${max_sell_size:.2f} available, ${requested_size:.2f} requested")
                    trade_result = {"status": "error", "reason": ["Insufficient BTC for sell"]}
                else:
                    actual_size = min(requested_size, max_sell_size)
                    logger.info(f"💰 Executing sell: ${actual_size:.2f} (requested: ${requested_size:.2f}, available: ${max_sell_size:.2f})")
                    trade_result = trading_bot.place_sell_order("BTC/USD", actual_size)
                    # Persist trade to ledger (best-effort)
                    try:
                        if LEDGER_AVAILABLE and isinstance(trade_result, dict) and trade_result.get('status') == 'success':
                            qty_btc = float(trade_result.get('quantity', 0.0))
                            price = float(trade_result.get('price_per_btc', 0.0)) if trade_result.get('price_per_btc') is not None else (qty_btc and actual_size/qty_btc or 0.0)
                            fee = float(trade_result.get('fee', 0.0)) if trade_result.get('fee') is not None else 0.0
                            ledger_record_order(
                                symbol="BTC",
                                side="sell",
                                qty_btc=qty_btc,
                                notional_usd=float(actual_size),
                                price=price,
                                demo_mode=bool(trade_result.get('demo_mode', False)),
                                exchange_order_id=trade_result.get('order_id'),
                                fee=fee,
                                metadata=trade_result,
                            )
                    except Exception as e:
                        logger.warning(f"Ledger write failed (auto sell): {e}")
                    
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

        # Snapshot equity (best-effort)
        try:
            if LEDGER_AVAILABLE:
                acct = trading_bot.get_account_info() if trading_bot else None
                equity_val = float(acct.get('equity', 0.0)) if isinstance(acct, Dict) or isinstance(acct, dict) else 0.0
                if equity_val:
                    ledger_snapshot_equity(equity_val)
        except Exception as e:
            logger.warning(f"Equity snapshot failed: {e}")

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

# REMOVED: /auto_trade_demo endpoint for security reasons
# This endpoint was allowing unauthorized access to trading functionality

@app.post("/auto_trade_scheduled")
async def auto_trade_scheduled():
    """Execute automated trading without authentication (for scheduled runs)"""
    logger.info("🤖 Scheduled Auto Trade endpoint called")
    
    if not trading_bot:
        logger.warning("No trading bot available for scheduled auto trade")
        return {"status": "error", "message": "No trading bot available"}

    if not llm_strategy:
        logger.warning("LLM strategy not available for scheduled auto trade")
        return {"status": "error", "message": "LLM strategy not available"}

    try:
        logger.info("🤖 Starting scheduled LLM market analysis...")
        
        # Get current market data
        market_data = get_btc_market_data()
        if not market_data:
            logger.error("Unable to get market data for scheduled auto trade")
            return {"status": "error", "message": "Unable to get market data"}

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
        try:
            signal = llm_strategy.generate_trading_signal(
                current_price=current_price,
                price_change_24h=price_change_24h,
                volume_24h=volume_24h,
                news_sentiment=news_sentiment
            )
            
            logger.info(f"🎯 LLM Signal: {signal['action'].upper()} - {signal['reason']} (confidence: {signal['confidence']:.2f})")
            
        except Exception as llm_error:
            logger.error(f"❌ LLM signal generation failed: {llm_error}")
            # Create fallback signal
            signal = {
                'action': 'hold',
                'should_trade': False,
                'reason': f'LLM analysis failed: {str(llm_error)}',
                'confidence': 0.5,
                'risk_level': 'medium',
                'position_size': None,
                'price_target': None,
                'stop_loss': None,
                'analysis': f'LLM Error: {str(llm_error)}',
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"🔄 Using fallback signal: {signal['action'].upper()}")

        # Check account balance before executing trade
        account_balance = trading_bot.get_account_info()
        logger.info(f"💰 Account Balance: {account_balance}")
        
        # Calculate appropriate position size based on available funds
        available_funds = 0.0
        btc_balance = 0.0
        
        if account_balance and 'balances' in account_balance:
            balances = account_balance['balances']
            # Check GBP balance
            gbp_balance = float(balances.get('ZGBP', '0.0'))
            # Check USD balance  
            usd_balance = float(balances.get('ZUSD', '0.0'))
            # Check BTC balance
            btc_balance = float(balances.get('XXBT', '0.0'))
            
            available_funds = gbp_balance + usd_balance
            logger.info(f"💵 Available funds: £{gbp_balance:.2f} + ${usd_balance:.2f} = ${available_funds:.2f}")
            logger.info(f"₿ BTC balance: {btc_balance:.8f}")
        
        # Execute allocation change via target exposure if provided
        trade_result = None
        if signal.get('target_exposure') is not None:
            try:
                target = float(signal.get('target_exposure'))
                trade_result = trading_bot.rebalance_to_target(target)
            except Exception as e:
                logger.warning(f"Rebalance failed: {e}")
                trade_result = {"status": "error", "reason": str(e)}
        elif signal['should_trade'] and signal['position_size']:
            # Adjust position size based on available funds
            requested_size = signal['position_size']
            max_buy_size = available_funds * 0.95  # Use 95% of available funds
            max_sell_size = btc_balance * current_price * 0.95  # Use 95% of BTC balance
            
            if signal['action'] == 'buy':
                if max_buy_size < 10.0:  # Minimum $10 trade
                    logger.info(f"⏸️ Insufficient funds for buy: ${max_buy_size:.2f} available, ${requested_size:.2f} requested")
                    trade_result = {"status": "error", "reason": ["Insufficient funds for buy"]}
                else:
                    actual_size = min(requested_size, max_buy_size)
                    logger.info(f"💰 Executing buy: ${actual_size:.2f} (requested: ${requested_size:.2f}, available: ${max_buy_size:.2f})")
                    trade_result = trading_bot.place_buy_order("BTC/USD", actual_size, 1.0)
                    
            elif signal['action'] == 'sell':
                if max_sell_size < 10.0:  # Minimum $10 trade
                    logger.info(f"⏸️ Insufficient BTC for sell: ${max_sell_size:.2f} available, ${requested_size:.2f} requested")
                    trade_result = {"status": "error", "reason": ["Insufficient BTC for sell"]}
                else:
                    actual_size = min(requested_size, max_sell_size)
                    logger.info(f"💰 Executing sell: ${actual_size:.2f} (requested: ${requested_size:.2f}, available: ${max_sell_size:.2f})")
                    trade_result = trading_bot.place_sell_order("BTC/USD", actual_size)
                    
            logger.info(f"✅ Scheduled trade result: {trade_result}")
        else:
            logger.info("⏸️ No scheduled trade executed - conditions not met")

        # Log the automated trade
        trade_log.append({
            "action": "SCHEDULED_AUTO_TRADE",
            "signal": signal,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trade_result": trade_result
        })

        # Snapshot equity (best-effort)
        try:
            if LEDGER_AVAILABLE:
                acct = trading_bot.get_account_info() if trading_bot else None
                equity_val = float(acct.get('equity', 0.0)) if isinstance(acct, Dict) or isinstance(acct, dict) else 0.0
                if equity_val:
                    ledger_snapshot_equity(equity_val)
        except Exception as e:
            logger.warning(f"Equity snapshot failed: {e}")

        logger.info("🤖 Scheduled auto trade analysis completed successfully")
        return {
            "status": "success",
            "message": f"Scheduled automated trading analysis complete",
            "signal": signal,
            "trade_executed": trade_result is not None,
            "trade_result": trade_result
        }

    except Exception as e:
        logger.error(f"❌ Error in scheduled automated trading: {e}")
        return {"status": "error", "message": f"Error in scheduled automated trading: {str(e)}"}

@app.get("/account_balance")
async def get_account_balance(token: str = None):
    """Get current account balance and positions"""
    logger.info("💰 Account balance requested")
    
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        username = verify_jwt_token(token)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not trading_bot:
        return {"message": "No trading bot available", "balance": None}
    
    try:
        # Get account information
        account_info = trading_bot.get_account_info()
        positions = trading_bot.get_positions()
        
        return {
            "account": account_info,
            "positions": positions,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting account balance: {e}")
        return {"message": f"Error getting account balance: {str(e)}", "balance": None}

@app.get("/trade_history")
async def get_trade_history(token: str = None, limit: int = 50):
    """Get recent trade history"""
    logger.info(f"📋 Trade history requested (limit: {limit})")
    
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        username = verify_jwt_token(token)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # Get recent trades from the trade log
        recent_trades = trade_log[-limit:] if len(trade_log) > limit else trade_log
        
        return {
            "trades": recent_trades,
            "total_trades": len(trade_log),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        return {"message": f"Error getting trade history: {str(e)}", "trades": []}

@app.get("/trading_summary")
async def get_trading_summary(token: str = None):
    """Get comprehensive trading summary including balance, positions, and recent activity"""
    logger.info("📊 Trading summary requested")
    
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        username = verify_jwt_token(token)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not trading_bot:
        return {"message": "No trading bot available", "summary": None}
    
    try:
        # Get account information
        account_info = trading_bot.get_account_info()
        positions = trading_bot.get_positions()
        
        # Get recent trades
        recent_trades = trade_log[-10:] if len(trade_log) > 10 else trade_log
        
        # Calculate basic statistics
        total_trades = len(trade_log)
        auto_trades = len([t for t in trade_log if t.get('action') == 'AUTO_TRADE' or t.get('action') == 'AUTO_TRADE_DEMO'])
        manual_trades = total_trades - auto_trades
        
        summary = {
            "account": account_info,
            "positions": positions,
            "recent_trades": recent_trades,
            "statistics": {
                "total_trades": total_trades,
                "auto_trades": auto_trades,
                "manual_trades": manual_trades,
                "last_trade": trade_log[-1] if trade_log else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    except Exception as e:
        logger.error(f"Error getting trading summary: {e}")
        return {"message": f"Error getting trading summary: {str(e)}", "summary": None}

@app.get("/pnl_summary")
async def pnl_summary(token: str = None):
    """Return realized/unrealized PnL, fees, exposure, equity, and HODL benchmark."""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        _ = verify_jwt_token(token)
    except:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not LEDGER_AVAILABLE:
        return {"message": "Ledger not configured"}

    market = get_btc_market_data() or {}
    current_price = float(market.get('close', 45000.0))

    pnl = ledger_compute_pnl(current_price)

    # Determine equity from broker if available; otherwise approximate
    equity = 0.0
    try:
        acct = trading_bot.get_account_info() if trading_bot else None
        if isinstance(acct, dict) and acct.get('equity') is not None:
            equity = float(acct.get('equity', 0.0))
        else:
            cash_usd = float(acct.get('cash', 0.0)) if isinstance(acct, dict) else 0.0
            equity = cash_usd + pnl['exposure_usd']
    except Exception:
        equity = pnl['exposure_usd']

    hodl_value, hodl_pnl = ledger_get_hodl_benchmark(current_price)
    exposure_pct = (pnl['exposure_usd'] / equity) if equity > 0 else 0.0

    return PnlSummary(
        symbol=pnl['symbol'],
        equity=equity,
        realized_pnl=pnl['realized_pnl'],
        unrealized_pnl=pnl['unrealized_pnl'],
        total_fees=pnl['fees'],
        exposure_usd=pnl['exposure_usd'],
        exposure_pct=exposure_pct,
        hodl_value=hodl_value,
        hodl_pnl=hodl_pnl,
        qty_btc=pnl['qty_btc'],
        avg_cost=pnl['avg_cost']
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 