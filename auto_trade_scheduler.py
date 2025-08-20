#!/usr/bin/env python3
"""
Automated Bitcoin Trading Scheduler
This script can be run via cron to automatically execute trading decisions
"""

import requests
import json
import logging
from datetime import datetime
import time
import os
import sys

# Add the current directory to Python path to import db module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
        }
    except Exception as e:
        logger.warning(f"Failed to load settings from DB: {e}")
        # Fallback to environment variables
        return {
            "max_exposure": float(os.getenv("MAX_EXPOSURE", "0.8")),
            "cooldown_hours": float(os.getenv("TRADE_COOLDOWN_HOURS", "3")),
            "min_confidence": float(os.getenv("MIN_CONFIDENCE", "0.7")),
            "min_trade_delta": float(os.getenv("MIN_TRADE_DELTA", "0.05")),
        }

def execute_auto_trade():
    """Execute automated trading via the API endpoint"""
    logger.info("🤖 Starting automated trading execution...")
    
    # Load current settings from database
    settings = load_runtime_settings()
    logger.info(f"⚙️ Current settings: max_exposure={settings['max_exposure']}, cooldown={settings['cooldown_hours']}h, min_confidence={settings['min_confidence']}, min_delta={settings['min_trade_delta']}")
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"📡 Attempt {attempt + 1}/{MAX_RETRIES}: Calling auto-trade endpoint...")
            
            # Make POST request to the scheduled auto-trade endpoint
            response = requests.post(
                TRADING_URL,
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
