#!/usr/bin/env python3
"""
Lightweight trading ledger and PnL accounting using SQLite.

Tables
- orders: one row per submitted order (assumes immediate/full fill for market orders)
- fills: optional granular fills per order (not used by default, but available)
- positions: rolling average cost and open quantity per symbol
- equity_curve: optional snapshots of equity over time (for charts)

This module is intentionally dependency-light, relying only on Python's sqlite3.
"""

import os
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List


DB_PATH = os.getenv("TRADING_DB_PATH", "/opt/btc-trading/trades.db")


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist."""
    global DB_PATH
    # Ensure directory exists, with safe fallbacks if permission denied
    target_dir = os.path.dirname(DB_PATH) or "."
    try:
        os.makedirs(target_dir, exist_ok=True)
    except Exception:
        # Fallback to /data, then /tmp
        for fallback in ("/data", "/tmp"):
            try:
                os.makedirs(fallback, exist_ok=True)
                DB_PATH = os.path.join(fallback, "trades.db")
                break
            except Exception:
                continue

    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exchange_order_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL CHECK(side IN ('buy','sell')),
                qty_btc REAL NOT NULL,
                notional_usd REAL NOT NULL,
                price REAL NOT NULL,
                fee REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'filled',
                demo_mode INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                qty_btc REAL NOT NULL,
                price REAL NOT NULL,
                fee REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(order_id) REFERENCES orders(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                qty_btc REAL NOT NULL DEFAULT 0,
                avg_cost REAL NOT NULL DEFAULT 0,
                realized_pnl REAL NOT NULL DEFAULT 0,
                fees REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS equity_curve (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_locks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lock_key TEXT NOT NULL UNIQUE,
                acquired_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                process_id TEXT,
                metadata TEXT
            )
            """
        )
    
    # Initialize default settings
    init_default_settings()


def _upsert_position(conn: sqlite3.Connection, symbol: str) -> None:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM positions WHERE symbol=?", (symbol,))
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO positions(symbol, qty_btc, avg_cost, realized_pnl, fees, updated_at) VALUES(?,?,?,?,?,?)",
            (symbol, 0.0, 0.0, 0.0, 0.0, datetime.utcnow().isoformat()),
        )


def record_order(symbol: str, side: str, qty_btc: float, notional_usd: float, price: float, demo_mode: bool, exchange_order_id: Optional[str] = None, fee: float = 0.0, metadata: Optional[Dict] = None) -> int:
    """Persist an order, update position book, and return the order row id.

    The position book uses average-cost accounting:
    - Buy: new_avg_cost = (old_qty*old_avg_cost + qty*price + fee) / (old_qty + qty)
    - Sell: realized = (price - avg_cost)*qty - fee; qty decreases; avg_cost unchanged unless qty becomes 0
    """
    created_at = datetime.utcnow().isoformat()
    with _connect() as conn:
        _upsert_position(conn, symbol)

        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO orders(exchange_order_id, symbol, side, qty_btc, notional_usd, price, fee, status, demo_mode, created_at, metadata)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                exchange_order_id,
                symbol,
                side,
                float(qty_btc),
                float(notional_usd),
                float(price),
                float(fee),
                "filled",
                1 if demo_mode else 0,
                created_at,
                json.dumps(metadata or {}),
            ),
        )
        order_id = cur.lastrowid

        # Update position
        cur.execute("SELECT qty_btc, avg_cost, realized_pnl, fees FROM positions WHERE symbol=?", (symbol,))
        row = cur.fetchone()
        qty = float(row[0])
        avg_cost = float(row[1])
        realized = float(row[2])
        fees_accum = float(row[3])

        if side == "buy":
            new_qty = qty + qty_btc
            if new_qty <= 0:
                new_avg_cost = 0.0
            else:
                new_avg_cost = (qty * avg_cost + qty_btc * price + fee) / new_qty
            realized_delta = 0.0
            qty_after = new_qty
            avg_cost_after = new_avg_cost
        else:  # sell
            qty_to_close = min(qty_btc, qty)
            realized_delta = (price - avg_cost) * qty_to_close - fee
            qty_after = max(0.0, qty - qty_btc)
            avg_cost_after = 0.0 if qty_after == 0 else avg_cost

        cur.execute(
            """
            UPDATE positions
            SET qty_btc=?, avg_cost=?, realized_pnl=realized_pnl+?, fees=fees+?, updated_at=?
            WHERE symbol=?
            """,
            (
                qty_after,
                avg_cost_after,
                realized_delta,
                fee,
                datetime.utcnow().isoformat(),
                symbol,
            ),
        )

        return order_id


def snapshot_equity(equity_value: float) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO equity_curve(timestamp, equity) VALUES(?,?)",
            (datetime.utcnow().isoformat(), float(equity_value)),
        )


def get_equity_curve(limit: int = 1000) -> List[Dict]:
    """Return the most recent equity snapshots (timestamp, equity)."""
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT timestamp, equity FROM equity_curve ORDER BY id DESC LIMIT ?",
            (int(limit),),
        )
        rows = cur.fetchall()
        # Return in chronological order
        rows = list(reversed(rows))
        return [{"timestamp": r[0], "equity": float(r[1])} for r in rows]


def get_hodl_curve(limit: int = 1000) -> List[Dict]:
    """Return HODL benchmark series aligned to equity snapshots.

    Assumes buying the cumulative net invested USD at the first trade's price and holding.
    For each equity snapshot timestamp, compute net invested up to that timestamp.
    """
    with _connect() as conn:
        cur = conn.cursor()
        # Get first price
        cur.execute("SELECT created_at, price FROM orders ORDER BY created_at ASC LIMIT 1")
        first = cur.fetchone()
        if not first:
            return []
        first_price = float(first[1]) or 0.0
        if first_price <= 0:
            return []
        # Get equity snapshots (chronological)
        cur.execute("SELECT timestamp FROM equity_curve ORDER BY id DESC LIMIT ?", (int(limit),))
        snaps = [r[0] for r in reversed(cur.fetchall())]
        if not snaps:
            return []
        res = []
        net_invested = 0.0
        order_idx = 0
        # Load all orders ordered by created_at
        cur.execute("SELECT created_at, side, notional_usd FROM orders ORDER BY created_at ASC")
        orders = cur.fetchall()
        for ts in snaps:
            # accumulate orders up to timestamp ts
            while order_idx < len(orders) and orders[order_idx][0] <= ts:
                side = orders[order_idx][1]
                notional = float(orders[order_idx][2])
                net_invested += notional if side == "buy" else -notional
                order_idx += 1
            btc_hodl = (net_invested / first_price) if first_price > 0 else 0.0
            value = btc_hodl * get_last_price(conn, default=first_price)
            res.append({"timestamp": ts, "hodl_value": float(value)})
        return res


def get_last_price(conn: sqlite3.Connection, default: float = 0.0) -> float:
    cur = conn.cursor()
    cur.execute("SELECT price FROM orders ORDER BY id DESC LIMIT 1")
    r = cur.fetchone()
    return float(r[0]) if r else float(default)


def get_positions() -> Dict:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT symbol, qty_btc, avg_cost, realized_pnl, fees FROM positions")
        rows = cur.fetchall()
        return {
            r[0]: {
                "qty_btc": float(r[1]),
                "avg_cost": float(r[2]),
                "realized_pnl": float(r[3]),
                "fees": float(r[4]),
            }
            for r in rows
        }


def compute_pnl(current_price: float) -> Dict:
    """Return realized/unrealized pnl, fees, and open exposure for BTC."""
    pos = get_positions().get("BTC", {"qty_btc": 0.0, "avg_cost": 0.0, "realized_pnl": 0.0, "fees": 0.0})
    qty = pos["qty_btc"]
    avg_cost = pos["avg_cost"]
    realized = pos["realized_pnl"]
    fees = pos["fees"]
    unrealized = (current_price - avg_cost) * qty if qty > 0 else 0.0
    exposure_usd = qty * current_price
    return {
        "symbol": "BTC",
        "qty_btc": qty,
        "avg_cost": avg_cost,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "fees": fees,
        "exposure_usd": exposure_usd,
    }


def get_hodl_benchmark(current_price: float) -> Tuple[float, float]:
    """Compute a simple HODL benchmark based on net invested capital.

    Logic: If you had taken the same net USD cash flow (buys minus sells) at the time of
    the first trade and bought BTC then, held to now, what is it worth?
    Returns (hodl_value_usd, hodl_pnl_usd).
    """
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT created_at, side, notional_usd FROM orders ORDER BY created_at ASC")
        rows = cur.fetchall()
        if not rows:
            return 0.0, 0.0

        first_ts = rows[0][0]
        # Find an approximate first price from the first order row
        cur.execute("SELECT price FROM orders WHERE created_at=? LIMIT 1", (first_ts,))
        r = cur.fetchone()
        if not r:
            return 0.0, 0.0
        first_price = float(r[0])

        # Net invested USD (buys positive, sells negative)
        net_invested = 0.0
        for row in rows:
            side = row[1]
            notional = float(row[2])
            net_invested += notional if side == "buy" else -notional

        btc_if_hodl = (net_invested / first_price) if first_price > 0 else 0.0
        hodl_value = btc_if_hodl * get_last_price(conn, default=first_price)
        hodl_pnl = hodl_value - net_invested
        return hodl_value, hodl_pnl


def get_realized_pnl_total() -> float:
    """Get total realized PnL from all positions."""
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT SUM(realized_pnl) FROM positions 
            WHERE realized_pnl IS NOT NULL
        """)
        result = cur.fetchone()
        return float(result[0]) if result and result[0] is not None else 0.0


def get_fees_total() -> float:
    """Get total fees from positions (more reliable than fills table)."""
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT SUM(fees) FROM positions WHERE fees IS NOT NULL")
        result = cur.fetchone()
        return float(result[0]) if result and result[0] is not None else 0.0


def get_avg_cost_from_positions() -> Optional[float]:
    """Get average cost from current position."""
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT avg_cost FROM positions 
            WHERE avg_cost IS NOT NULL AND avg_cost > 0
            ORDER BY updated_at DESC LIMIT 1
        """)
        result = cur.fetchone()
        return float(result[0]) if result and result[0] is not None else None


def read_settings() -> Dict:
    """Read all settings from the database."""
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM settings")
        rows = cur.fetchall()
        settings = {}
        for row in rows:
            key = row[0]
            value = row[1]
            # Try to convert to appropriate type
            try:
                if value.lower() in ('true', 'false'):
                    settings[key] = value.lower() == 'true'
                elif '.' in value:
                    settings[key] = float(value)
                else:
                    settings[key] = int(value)
            except (ValueError, AttributeError):
                settings[key] = value
        return settings


def write_setting(key: str, value) -> None:
    """Write a setting to the database."""
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO settings(key, value, updated_at)
            VALUES(?, ?, ?)
            """,
            (key, str(value), datetime.utcnow().isoformat())
        )


def get_setting(key: str, default=None):
    """Get a single setting value from the database."""
    settings = read_settings()
    return settings.get(key, default)


def init_default_settings() -> None:
    """Initialize default settings if they don't exist."""
    import os
    defaults = {
        "min_confidence": float(os.getenv("MIN_CONFIDENCE", "0.7")),
        "max_exposure": float(os.getenv("MAX_EXPOSURE", "0.8")),
        "trade_cooldown_hours": float(os.getenv("TRADE_COOLDOWN_HOURS", "3")),
        "min_trade_delta": float(os.getenv("MIN_TRADE_DELTA", "0.05")),
        "min_trade_delta_usd": float(os.getenv("MIN_TRADE_DELTA_USD", "30.0")),
        "auto_trade_enabled": True,
        "safety_skip_degraded": True,
        "safety_max_price_staleness_sec": 120.0,
        "safety_min_expected_move_pct": 0.1,
        "safety_daily_pnl_limit_usd": -5.0,
        "safety_daily_equity_drop_pct": 3.0,
    }
    
    current_settings = read_settings()
    for key, default_value in defaults.items():
        if key not in current_settings:
            write_setting(key, default_value)


def acquire_trade_lock(lock_key: str = "auto_trade", ttl_sec: int = 600, process_id: str = None, metadata: Dict = None) -> bool:
    """Acquire a trade lock with TTL.
    
    Args:
        lock_key: Unique identifier for the lock
        ttl_sec: Time-to-live in seconds (default 10 minutes)
        process_id: Optional process identifier
        metadata: Optional metadata for the lock
        
    Returns:
        True if lock was acquired, False if already held and not expired
    """
    now = datetime.utcnow()
    expires_at = now.replace(microsecond=0) + timedelta(seconds=ttl_sec)
    
    with _connect() as conn:
        cur = conn.cursor()
        
        # First, clean up expired locks
        cur.execute(
            "DELETE FROM trade_locks WHERE expires_at < ?",
            (now.isoformat(),)
        )
        
        # Try to acquire the lock
        try:
            cur.execute(
                """
                INSERT INTO trade_locks(lock_key, acquired_at, expires_at, process_id, metadata)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    lock_key,
                    now.isoformat(),
                    expires_at.isoformat(),
                    process_id or str(os.getpid()),
                    json.dumps(metadata or {})
                )
            )
            return True
        except sqlite3.IntegrityError:
            # Lock already exists, check if it's expired
            cur.execute(
                "SELECT expires_at FROM trade_locks WHERE lock_key = ?",
                (lock_key,)
            )
            row = cur.fetchone()
            if row:
                existing_expires = datetime.fromisoformat(row[0])
                if existing_expires < now:
                    # Lock is expired, update it
                    cur.execute(
                        """
                        UPDATE trade_locks 
                        SET acquired_at = ?, expires_at = ?, process_id = ?, metadata = ?
                        WHERE lock_key = ?
                        """,
                        (
                            now.isoformat(),
                            expires_at.isoformat(),
                            process_id or str(os.getpid()),
                            json.dumps(metadata or {}),
                            lock_key
                        )
                    )
                    return True
            return False


def release_trade_lock(lock_key: str = "auto_trade") -> bool:
    """Release a trade lock.
    
    Args:
        lock_key: Unique identifier for the lock
        
    Returns:
        True if lock was released, False if it didn't exist
    """
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM trade_locks WHERE lock_key = ?", (lock_key,))
        return cur.rowcount > 0


def get_trade_lock_info(lock_key: str = "auto_trade") -> Optional[Dict]:
    """Get information about a trade lock.
    
    Args:
        lock_key: Unique identifier for the lock
        
    Returns:
        Dictionary with lock info or None if not found
    """
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT lock_key, acquired_at, expires_at, process_id, metadata FROM trade_locks WHERE lock_key = ?",
            (lock_key,)
        )
        row = cur.fetchone()
        if row:
            return {
                "lock_key": row[0],
                "acquired_at": row[1],
                "expires_at": row[2],
                "process_id": row[3],
                "metadata": json.loads(row[4]) if row[4] else {}
            }
        return None


def cleanup_expired_locks() -> int:
    """Clean up expired trade locks.
    
    Returns:
        Number of locks cleaned up
    """
    now = datetime.utcnow()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM trade_locks WHERE expires_at < ?", (now.isoformat(),))
        return cur.rowcount


