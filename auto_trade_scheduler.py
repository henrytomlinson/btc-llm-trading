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

def execute_auto_trade():
    """Execute automated trading via the API endpoint"""
    logger.info("🤖 Starting automated trading execution...")
    
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
