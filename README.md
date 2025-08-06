# 🚀 Bitcoin LLM Trading System

A focused Bitcoin trading platform that combines real-time market data, sentiment analysis, and automated trading through Alpaca Trading API with compact LLM integration.

## ✨ Key Features

### 🔐 **Security**
- **JWT Authentication**: Secure stateless authentication
- **Protected Endpoints**: All trading operations require authentication
- **Session Management**: Cross-pod compatible authentication

### 💰 **Bitcoin Trading**
- **BTC-Only Focus**: Dedicated Bitcoin trading interface
- **Dollar-Based Trading**: Buy $X worth of Bitcoin
- **Real-Time Market Data**: Live Bitcoin price feeds
- **Automated Trading**: Algorithmic buy/sell decisions
- **Position Management**: Track Bitcoin positions
- **Paper Trading**: Safe testing environment

### 🤖 **Compact LLM Integration**
- **Lightweight Sentiment Analysis**: Keyword-based Bitcoin sentiment
- **Trading Signal Generation**: Algorithmic decision making
- **News Integration**: Real-time Bitcoin news analysis
- **Technical Indicators**: Price momentum and volume analysis

### 📊 **Data Integration**
- **News API**: Real-time Bitcoin news sentiment analysis
- **Alpaca Trading API**: Live Bitcoin trading execution
- **Market Data**: Real-time Bitcoin price and volume data

### 🏗️ **Infrastructure**
- **Kubernetes Deployment**: Scalable container orchestration
- **Google Cloud Platform**: Production-ready infrastructure
- **Docker Containerization**: Consistent deployment
- **Load Balancing**: High availability setup

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker
- Google Cloud Platform account
- Alpaca Trading API keys

### Local Development
```bash
# Clone the repository
git clone https://github.com/henrytomlinson/btc-llm-trading.git
cd btc-llm-trading

# Install dependencies
pip install -r requirements_btc.txt

# Set up environment variables
export ALPACA_API_KEY=your_alpaca_key
export ALPACA_SECRET_KEY=your_alpaca_secret
export NEWS_API_KEY=your_news_api_key
export COHERE_KEY=your_cohere_key

# Run locally
python main_btc.py
```

### Docker Deployment
```bash
# Build the image
docker build -f Dockerfile.btc -t btc-trading-app .

# Run container
docker run -p 8000:8000 btc-trading-app
```

### Kubernetes Deployment
```bash
# Deploy to GKE
kubectl apply -f deployment_btc.yaml

# Scale as needed
kubectl scale deployment btc-trading-app --replicas=2
```

## 📋 API Endpoints

### Authentication
- `POST /auth/login` - Login with username/password
- `GET /login` - Login page

### Bitcoin Trading Operations
- `POST /buy_btc` - Buy Bitcoin with dollar amount
- `POST /sell_btc` - Sell Bitcoin
- `GET /btc_data` - Get Bitcoin market data and analysis

### System Health
- `GET /test_apis` - API connectivity status
- `GET /` - Main Bitcoin trading dashboard

## 🔧 Configuration

### Environment Variables
```bash
# Alpaca Trading API
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret

# News API
NEWS_API_KEY=your_news_api_key

# Cohere API (for future LLM features)
COHERE_KEY=your_cohere_key
```

### Default Credentials
- **Username**: `admin`
- **Password**: `trading123`

## 💡 Usage Examples

### Bitcoin Trading
```bash
# Buy $100 worth of Bitcoin
curl -X POST "http://your-app/buy_btc" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "amount=100&token=your_jwt_token"
```

### Get Bitcoin Data
```bash
curl "http://your-app/btc_data?token=your_jwt_token"
```

## 🛠️ Development

### Project Structure
```
├── main_btc.py              # Main FastAPI application
├── alpaca_trading_btc.py    # Bitcoin Alpaca trading bot
├── deployment_btc.yaml      # Kubernetes deployment
├── Dockerfile.btc          # Docker configuration
├── requirements_btc.txt     # Python dependencies
└── README.md               # Documentation
```

### Key Components
- **FastAPI Backend**: RESTful API with async support
- **Bitcoin Alpaca Integration**: Real-time Bitcoin trading execution
- **JWT Authentication**: Secure stateless auth
- **Compact LLM**: Lightweight sentiment analysis
- **Kubernetes**: Scalable deployment
- **Docker**: Containerized application

## 🤖 Compact LLM Features

### Sentiment Analysis
- **Keyword-based**: Analyzes Bitcoin news for sentiment
- **Real-time**: Processes latest Bitcoin news
- **Caching**: 5-minute cache to respect API limits

### Trading Signals
- **Multi-factor**: Combines sentiment, price, volume, and momentum
- **Weighted Scoring**: 40% sentiment, 30% momentum, 20% volume, 10% price level
- **Real-time**: Updates every 30 seconds

### Technical Indicators
- **Price Momentum**: 24-hour price change analysis
- **Volume Analysis**: Trading volume impact
- **Price Levels**: Support/resistance considerations

## 📈 Performance

- **Response Time**: < 200ms for API calls
- **Scalability**: Horizontal pod autoscaling
- **Availability**: Multi-pod deployment
- **Security**: JWT-based authentication
- **Resource Usage**: Lightweight (1-2GB memory)

## 🔒 Security Features

- **JWT Tokens**: Stateless authentication
- **HTTPS**: Encrypted communication
- **API Rate Limiting**: Respect API limits
- **Input Validation**: Sanitized user inputs

## 🚀 Deployment Status

- ✅ **Bitcoin-Only Interface**
- ✅ **Compact LLM Integration**
- ✅ **JWT Authentication**
- ✅ **Dollar-Based Bitcoin Trading**
- ✅ **Kubernetes Deployment**
- ✅ **Real-Time Bitcoin Data**
- ✅ **News Integration**
- ✅ **Position Management**

## 📞 Support

For issues and questions:
- Check the documentation in `/docs`
- Review API status at `/test_apis`
- Check logs: `kubectl logs -l app=btc-trading-app`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for Bitcoin trading enthusiasts**

