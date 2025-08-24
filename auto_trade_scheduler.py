#!/usr/bin/env python3
"""
Automated Bitcoin Trading Scheduler
This script can be run via cron to automatically execute trading decisions
"""

import requests
import json
import logging
from datetime import datetime, timezone
import time
import os
import sys

# Add the current directory to Python path to import db module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

load_env_file()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/btc-trading/auto_trade.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Configuration
TRADING_URL = "https://henryt-btc.live/auto_trade_scheduled"
MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds

# Max staleness used by both summary and trading gates (same as main_btc.py)
MAX_PRICE_STALENESS_SEC = int(os.getenv("MAX_PRICE_STALENESS_SEC", "120"))

def load_runtime_settings():
    """Load runtime settings from database."""
    try:
        from db import read_settings
        settings = read_settings()
        return {
            "max_exposure": float(settings.get("max_exposure", 0.8)),
            "cooldown_hours": float(settings.get("trade_cooldown_hours", 3)),
            "min_confidence": float(settings.get("min_confidence", 0.7)),
            "min_trade_delta": float(settings.get("min_trade_delta", 0.05)),
            "min_trade_delta_usd": float(settings.get("min_trade_delta_usd", 30.0)),
        }
    except Exception as e:
        logger.warning(f"Failed to load settings from DB: {e}")
        # Fallback to environment variables
        return {
            "max_exposure": float(os.getenv("MAX_EXPOSURE", "0.8")),
            "cooldown_hours": float(os.getenv("TRADE_COOLDOWN_HOURS", "3")),
            "min_confidence": float(os.getenv("MIN_CONFIDENCE", "0.7")),
            "min_trade_delta": float(os.getenv("MIN_TRADE_DELTA", "0.05")),
            "min_trade_delta_usd": float(os.getenv("MIN_TRADE_DELTA_USD", "30.0")),
        }

def check_price_staleness():
    """Check if price data is stale before executing trades"""
    try:
        # Get current price data from the public endpoint
        response = requests.get(
            "https://henryt-btc.live/btc_data_public",
            timeout=10,
            verify=False
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'ok' and data.get('last_update'):
                try:
                    # Parse the timestamp
                    price_ts = datetime.fromisoformat(data['last_update'].replace('Z', '+00:00'))
                    staleness = (datetime.now(timezone.utc) - price_ts).total_seconds()
                    
                    if staleness > MAX_PRICE_STALENESS_SEC:
                        logger.warning(f"🛡️ Skip trade: price stale ({int(staleness)}s > {MAX_PRICE_STALENESS_SEC}s)")
                        return False, f"price_stale_{int(staleness)}s"
                    else:
                        logger.info(f"✅ Price data is fresh ({int(staleness)}s old)")
                        return True, None
                except Exception as e:
                    logger.warning(f"🛡️ Skip trade: unable to parse price timestamp: {e}")
                    return False, "timestamp_parse_error"
            else:
                logger.warning(f"🛡️ Skip trade: data status is {data.get('status', 'unknown')}")
                return False, "data_degraded"
        else:
            logger.warning(f"🛡️ Skip trade: unable to fetch price data (status {response.status_code})")
            return False, "data_unavailable"
            
    except Exception as e:
        logger.warning(f"🛡️ Skip trade: price staleness check failed: {e}")
        return False, "staleness_check_failed"

def execute_auto_trade():
    """Execute automated trading via the API endpoint"""
    logger.info("🤖 Starting automated trading execution...")
    
    # Check price staleness before proceeding
    price_ok, reason = check_price_staleness()
    if not price_ok:
        logger.warning(f"🛡️ Skipping trade due to stale data: {reason}")
        return {"status": "skipped", "reason": reason}
    
    # Load current settings from database
    settings = load_runtime_settings()
    logger.info(f"⚙️ Current settings: max_exposure={settings['max_exposure']}, cooldown={settings['cooldown_hours']}h, min_confidence={settings['min_confidence']}, min_delta={settings['min_trade_delta']}")
    
    # Get authentication token
    try:
        auth_response = requests.post(
            "https://henryt-btc.live/auth/login",
            data={
                'username': os.getenv('ADMIN_USERNAME', 'admin'),
                'password': os.getenv('ADMIN_PASSWORD', 'change_this_password_immediately')
            },
            timeout=10,
            verify=False
        )
        
        if auth_response.status_code == 200:
            auth_data = auth_response.json()
            token = auth_data.get('session_token')
            if not token:
                logger.error("❌ No session token in auth response")
                return {"status": "error", "reason": "auth_failed"}
        else:
            logger.error(f"❌ Auth failed with status {auth_response.status_code}")
            return {"status": "error", "reason": "auth_failed"}
            
    except Exception as e:
        logger.error(f"❌ Auth request failed: {e}")
        return {"status": "error", "reason": "auth_failed"}
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"📡 Attempt {attempt + 1}/{MAX_RETRIES}: Calling auto-trade endpoint...")
            
            # Make POST request to the scheduled auto-trade endpoint with token
            response = requests.post(
                TRADING_URL,
                data={'token': token},
                timeout=60,  # 60 second timeout
                verify=False  # Skip SSL verification for self-signed certs
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Auto-trade executed successfully!")
                logger.info(f"📊 Result: {json.dumps(result, indent=2)}")
                return result
            else:
                logger.error(f"❌ Auto-trade failed with status {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error on attempt {attempt + 1}: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error on attempt {attempt + 1}: {e}")
        
        if attempt < MAX_RETRIES - 1:
            logger.info(f"⏳ Waiting {RETRY_DELAY} seconds before retry...")
            time.sleep(RETRY_DELAY)
    
    logger.error("❌ All retry attempts failed")
    return None

def main():
    """Main function to execute automated trading"""
    logger.info("🚀 Automated Bitcoin Trading Scheduler Started")
    logger.info(f"🕐 Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        result = execute_auto_trade()
        if result:
            logger.info("🎉 Automated trading completed successfully")
        else:
            logger.error("💥 Automated trading failed")
            exit(1)
            
    except Exception as e:
        logger.error(f"💥 Fatal error in automated trading: {e}")
        exit(1)

if __name__ == "__main__":
    main()
