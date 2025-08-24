#!/usr/bin/env python3
"""Check database for recent trades and positions"""

from db import _connect

def check_database():
    with _connect() as conn:
        cursor = conn.cursor()
        
        # Check positions
        cursor.execute("SELECT COUNT(*) FROM positions")
        pos_count = cursor.fetchone()[0]
        print(f"Total positions: {pos_count}")
        
        # Check recent positions
        cursor.execute("SELECT * FROM positions ORDER BY updated_at DESC LIMIT 3")
        positions = cursor.fetchall()
        print(f"Recent positions: {len(positions)}")
        for pos in positions:
            print(f"  Position: {dict(pos)}")
        
        # Check settings
        cursor.execute("SELECT * FROM settings WHERE key LIKE '%bias%' OR key LIKE '%grid%'")
        bias_settings = cursor.fetchall()
        print(f"Bias/Grid settings: {len(bias_settings)}")
        for setting in bias_settings:
            print(f"  Setting: {dict(setting)}")

if __name__ == "__main__":
    check_database()
