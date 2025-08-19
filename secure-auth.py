#!/usr/bin/env python3
"""
Enhanced Security for Bitcoin Trading System
"""

import os
import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

# Security configuration
MAX_LOGIN_ATTEMPTS = 5
LOGIN_TIMEOUT_MINUTES = 15
SESSION_TIMEOUT_HOURS = 8
RATE_LIMIT_REQUESTS = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

# Store for rate limiting and failed login attempts
login_attempts = {}
rate_limit_store = {}

class SecurityManager:
    """Enhanced security manager for trading system"""
    
    def __init__(self):
        self.secret_key = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        
    def create_access_token(self, data: dict):
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str):
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def check_rate_limit(self, client_ip: str):
        """Check rate limiting for client IP"""
        now = time.time()
        if client_ip not in rate_limit_store:
            rate_limit_store[client_ip] = []
        
        # Remove old requests outside window
        rate_limit_store[client_ip] = [
            req_time for req_time in rate_limit_store[client_ip] 
            if now - req_time < RATE_LIMIT_WINDOW
        ]
        
        # Check if limit exceeded
        if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Add current request
        rate_limit_store[client_ip].append(now)
    
    def check_login_attempts(self, username: str):
        """Check failed login attempts"""
        now = time.time()
        if username in login_attempts:
            attempts = login_attempts[username]
            # Remove old attempts outside timeout window
            attempts = [attempt_time for attempt_time in attempts 
                       if now - attempt_time < LOGIN_TIMEOUT_MINUTES * 60]
            
            if len(attempts) >= MAX_LOGIN_ATTEMPTS:
                raise HTTPException(status_code=429, detail="Too many failed login attempts")
            
            login_attempts[username] = attempts
    
    def record_failed_login(self, username: str):
        """Record failed login attempt"""
        now = time.time()
        if username not in login_attempts:
            login_attempts[username] = []
        login_attempts[username].append(now)

# Security middleware
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    security_manager = SecurityManager()
    payload = security_manager.verify_token(credentials.credentials)
    return payload

def require_admin(user = Depends(get_current_user)):
    """Require admin privileges"""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

def rate_limit_middleware(request: Request):
    """Rate limiting middleware"""
    client_ip = request.client.host
    security_manager = SecurityManager()
    security_manager.check_rate_limit(client_ip) 