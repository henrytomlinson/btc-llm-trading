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
