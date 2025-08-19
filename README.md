## Bitcoin LLM Trading System (Kraken + Cohere)

Production-grade FastAPI service that automates Bitcoin trading on Kraken with LLM (Cohere) analysis and a minimal web dashboard behind nginx/HTTPS. Designed to run 24/7 on a VPS using Docker Compose and a cron scheduler.

### Highlights
- Kraken trading with balance checks and dynamic position sizing
- Cohere LLM analysis with a 15‑minute cache to control costs
- Public JSON data endpoint and simple dashboard
- nginx TLS termination and security headers
- Hourly automated trading via cron (tunable)

---

## Architecture
- `nginx` → reverse proxy + TLS
- `btc-trading-app` (FastAPI) → endpoints, LLM/TA analysis, Kraken integration
- `cron` → calls `auto_trade_scheduled` hourly

Key files:
- `main_btc.py` — API/UI/caching/security
- `llm_trading_strategy.py` — Cohere integration and response parsing
- `kraken_trading_btc.py` — Kraken REST (auth, ticker, orders, balances)
- `Dockerfile.btc`, `docker-compose.yml`, `nginx.conf`
- `.env` — secrets/config (on server)
- `auto_trade_scheduler.py` — cron entry

---

## Prerequisites
- VPS with Docker + Docker Compose
- Domain + TLS (Let’s Encrypt)
- Kraken API key + private key
- Cohere Production key (prevents 429s)
- News API key (optional; has free-tier limits)

---

## Configure Environment (on VPS)
Create `/opt/btc-trading/.env`:
```
KRAKEN_API_KEY=your_kraken_key
KRAKEN_PRIVATE_KEY=your_kraken_private_secret
COHERE_KEY=your_cohere_prod_key
NEWS_API_KEY=your_news_api_key
ALLOWED_IPS=0.0.0.0
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
ADMIN_USERNAME=admin
ADMIN_PASSWORD=change_me
```
Restart after any `.env` change.

---

## Deploy / Operate
```
cd /opt/btc-trading
docker compose up -d --build
docker compose ps
docker compose logs --tail=100 btc-trading | cat
```
Site: `https://your-domain`

Quick checks:
```
curl -s https://your-domain/test_apis | jq .
curl -s https://your-domain/btc_data_public | jq .
```

---

## Scheduler (Auto Trade)
Hourly (recommended):
```
echo "0 * * * * /usr/bin/python3 /opt/btc-trading/auto_trade_scheduler.py >> /opt/btc-trading/cron.log 2>&1" | crontab -
crontab -l
```
Every 15 minutes (optional):
```
echo "*/15 * * * * /usr/bin/python3 /opt/btc-trading/auto_trade_scheduler.py >> /opt/btc-trading/cron.log 2>&1" | crontab -
```
View runs:
```
tail -f /opt/btc-trading/cron.log | cat
```
Stop scheduler:
```
crontab -r
```

---

## API Endpoints
- `GET /` — dashboard (simple UI)
- `GET /btc_data_public` — public JSON (price/change/sentiment/signal)
- `POST /auth/login` — returns JWT (use `ADMIN_USERNAME`/`ADMIN_PASSWORD`)
- `GET /btc_data?token=...` — authenticated data
- `POST /auto_trade` — authenticated auto trade
- `POST /auto_trade_scheduled` — cron path (no auth)
- `GET /account_balance?token=...`, `GET /trade_history?token=...`, `GET /trading_summary?token=...`
- `GET /test_apis` — connectivity

Example auth:
```
TOKEN=$(curl -s -X POST -F username=$ADMIN_USERNAME -F password=$ADMIN_PASSWORD https://your-domain/auth/login | jq -r .session_token)
curl -s "https://your-domain/btc_data?token=$TOKEN" | jq .
```

---

## Trading Logic (summary)
1) Market data from Kraken → price, 24h change, volume
2) Cohere LLM analysis (15‑min cache; keyword fallback on error/limits)
3) Signal from sentiment + momentum + volume
4) Before orders: balance check, 95% caps, min $10 trade

Default: execute only if confidence ≥ 0.7.

---

## Decision Logic: Buy / Sell / Hold (details)
- **Inputs combined**
  - **Market**: current price, 24h % change, 24h volume (from Kraken or CoinGecko fallback)
  - **News/Sentiment**: latest headline turned into a sentiment label (positive/neutral/negative)
  - **Optional TA**: placeholders for RSI/MACD/MAs/Bollinger; not required for execution

- **LLM-driven decision (used for live trading via `/auto_trade` and the scheduler)**
  - Build a compact market context and prompt Cohere for a strict-JSON response including `sentiment`, `recommended_action` (buy/sell/hold), `confidence` \/ `risk_level`, and optional `price_target` \/ `stop_loss`.
  - A trade is considered only when ALL are true:
    - **Confidence ≥ 0.7** and **risk_level ≤ medium**
    - `recommended_action` is `buy` or `sell` (if `hold`, no trade)
  - Position size: start from a **$1000 base**, then scale by risk and confidence
    - Risk multipliers: low = 1.0, medium = 0.7, high = 0.5
    - Final size = base × risk_multiplier × confidence, with a **$10 minimum**
  - Execution guardrails (per order):
    - Buys use up to **95% of available USD+GBP** funds; sells use up to **95% of BTC value**
    - Enforce **minimum $10** notional for both buy and sell

- **Dashboard signal (display-only)**
  - A lightweight, explainable score is shown on `/btc_data_public` and the UI:
    - Sentiment 40% + momentum (24h change) 30% + volume 20% + price level 10%
    - Thresholds: score > 0.3 → BUY, score < -0.3 → SELL, otherwise HOLD
  - This display signal does not place orders when the LLM strategy is available.

- **Fallbacks and cost control**
  - If the LLM call fails or returns malformed JSON, the system defaults to a neutral HOLD.
  - Sentiment is cached for **15 minutes** to limit LLM usage triggered by page refreshes and health checks.

- **Where to tune**
  - Confidence threshold, risk cap, and sizing multipliers: see `llm_trading_strategy.py`
  - Display signal weights and thresholds: see `generate_trading_signal` in `main_btc.py`
  - Base order sizing for LLM decisions: `$1000` inside `LLMTradingStrategy.generate_trading_signal`

References:
```149:180:llm_trading_strategy.py
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with the given prompt"""
        
        headers = {
            "Authorization": f"Bearer {self.cohere_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.3,
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods": "NONE"
        }
```

```309:350:llm_trading_strategy.py
    def should_execute_trade(self, analysis: MarketAnalysis) -> Tuple[bool, str]:
        """Determine if a trade should be executed based on LLM analysis"""
        
        # Check confidence threshold
        if analysis.confidence < self.min_confidence:
            return False, f"Confidence too low: {analysis.confidence:.2f} < {self.min_confidence}"
        
        # Check risk level
        risk_levels = {"low": 1, "medium": 2, "high": 3}
        max_risk = risk_levels.get(self.max_risk_level, 2)
        current_risk = risk_levels.get(analysis.risk_level, 2)
        
        if current_risk > max_risk:
            return False, f"Risk level too high: {analysis.risk_level}"
        
        # Don't execute if recommendation is "hold"
        if analysis.recommended_action == "hold":
            return False, "LLM recommends holding"
```

```301:334:main_btc.py
def generate_trading_signal(price: float, sentiment: int, volume: float, change_24h: float) -> int:
    # Sentiment factor (40% weight)
    # Price momentum factor (30% weight)
    # Volume factor (20% weight)
    # Price level factor (10% weight)
    # score > 0.3 → BUY, score < -0.3 → SELL, else HOLD
```

## Cost Controls (Cohere)
- Hourly schedule ≈ 24 calls/day
- 15‑minute cache prevents dashboard-triggered extra calls
- Rule of thumb at $0.15/1K tokens, 300–500 tokens/call → ~$30–$60/month
- Reduce to every 2 hours for ~$15–$30/month

Change cache TTL inside `analyze_btc_sentiment` in `main_btc.py`.

---

## Security
- IP allowlist (`ALLOWED_IPS`), rate limiting (`RATE_LIMIT_*`), JWT auth
- nginx with HSTS and security headers
- Never commit `.env` or keys

---

## Operations Cheatsheet
```
# Start/stop/status
docker compose up -d --build
docker compose ps
docker compose down

# Logs
docker compose logs --tail=100 btc-trading | cat
docker compose logs --tail=100 nginx | cat
tail -f /opt/btc-trading/cron.log | cat

# After editing .env
docker compose down && docker compose up -d --build
```

---

## Troubleshooting
- Site down → `docker compose ps`, then `docker compose logs --tail=200 btc-trading | cat`
- App restarting / SyntaxError → review latest `main_btc.py` edits and rebuild
- Cohere 429 → production key in `.env`, hourly schedule + cache
- Kraken errors → verify keys/balances/min trade size

Test Cohere from container:
```
docker compose exec -T btc-trading python3 - << 'PY'
import os,requests
key=os.getenv('COHERE_KEY')
print('Key starts with:', key[:6] if key else None)
print(requests.post('https://api.cohere.ai/v1/generate',
  headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},
  json={'model':'command','prompt':'Return JSON: {"status": "ok"}','max_tokens':20,'temperature':0},timeout=30).status_code)
PY
```

---

## File Map
- `main_btc.py` — API/UI/caching/auth/rate-limit/signals/scheduler endpoints
- `llm_trading_strategy.py` — Cohere calls + JSON parsing
- `kraken_trading_btc.py` — Kraken auth, ticker, orders, balances
- `auto_trade_scheduler.py` — cron entry
- `nginx.conf` — TLS + reverse proxy
- `Dockerfile.btc`, `docker-compose.yml` — build/runtime

---

## Notes
- Dashboard refreshes every 30s; caching prevents LLM overuse
- HEAD on `/` may 405; use GET or `/btc_data_public`
- Keep `.env` synced and restart after changes

Happy trading! 🚀
