#!/usr/bin/env python3
"""
Technical Analysis Module for Bitcoin Trading
Comprehensive technical indicators and analysis for AI-driven trading decisions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import ta
import yfinance as yf
import json

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Advanced technical analysis for Bitcoin trading"""
    
    def __init__(self):
        """Initialize technical analyzer"""
        self.indicators = {}
        self.data_cache = {}
        self.cache_duration = timedelta(minutes=5)
        
    def get_btc_data(self, period: str = "1y") -> pd.DataFrame:
        """Get Bitcoin historical data"""
        try:
            # Try to get data from cache first
            cache_key = f"btc_data_{period}"
            if cache_key in self.data_cache:
                cached_data, timestamp = self.data_cache[cache_key]
                if datetime.now() - timestamp < self.cache_duration:
                    return cached_data
            
            # Fetch fresh data
            btc = yf.Ticker("BTC-USD")
            data = btc.history(period=period)
            
            if data.empty:
                logger.warning("No data received from yfinance, using fallback")
                return self._generate_fallback_data()
            
            # Cache the data
            self.data_cache[cache_key] = (data, datetime.now())
            return data
            
        except Exception as e:
            logger.error(f"Error fetching BTC data: {e}")
            return self._generate_fallback_data()
    
    def _generate_fallback_data(self) -> pd.DataFrame:
        """Generate fallback data for testing"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
        prices = np.random.normal(45000, 5000, len(dates))
        volumes = np.random.normal(1000000, 200000, len(dates))
        
        data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        return data
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            rsi = ta.momentum.RSIIndicator(data['Close'], window=period)
            return rsi.rsi()
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(data))
    
    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            macd_indicator = ta.trend.MACD(data['Close'], window_fast=fast, window_slow=slow, window_sign=signal)
            return {
                'macd': macd_indicator.macd(),
                'macd_signal': macd_indicator.macd_signal(),
                'macd_histogram': macd_indicator.macd_diff()
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {
                'macd': pd.Series([0] * len(data)),
                'macd_signal': pd.Series([0] * len(data)),
                'macd_histogram': pd.Series([0] * len(data))
            }
    
    def calculate_moving_averages(self, data: pd.DataFrame) -> Dict:
        """Calculate various moving averages"""
        try:
            return {
                'sma_20': ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator(),
                'sma_50': ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator(),
                'sma_200': ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator(),
                'ema_12': ta.trend.EMAIndicator(data['Close'], window=12).ema_indicator(),
                'ema_26': ta.trend.EMAIndicator(data['Close'], window=26).ema_indicator()
            }
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return {}
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        try:
            bb = ta.volatility.BollingerBands(data['Close'], window=period, window_dev=std)
            return {
                'bb_upper': bb.bollinger_hband(),
                'bb_middle': bb.bollinger_mavg(),
                'bb_lower': bb.bollinger_lband(),
                'bb_width': bb.bollinger_wband(),
                'bb_percent': bb.bollinger_pband()
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {}
    
    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict:
        """Calculate Stochastic Oscillator"""
        try:
            stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], 
                                                    window=k_period, smooth_window=d_period)
            return {
                'stoch_k': stoch.stoch(),
                'stoch_d': stoch.stoch_signal()
            }
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return {}
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            atr = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=period)
            return atr.average_true_range()
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series([0] * len(data))
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate volume-based indicators"""
        try:
            return {
                'volume_sma': ta.volume.volume_sma(data['Close'], data['Volume'], window=20),
                'volume_ema': ta.volume.volume_ema(data['Close'], data['Volume'], window=20),
                'on_balance_volume': ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume(),
                'volume_price_trend': ta.volume.VolumePriceTrendIndicator(data['Close'], data['Volume']).volume_price_trend()
            }
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return {}
    
    def get_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict:
        """Calculate support and resistance levels"""
        try:
            highs = data['High'].rolling(window=window, center=True).max()
            lows = data['Low'].rolling(window=window, center=True).min()
            
            current_price = data['Close'].iloc[-1]
            
            # Find nearest support and resistance
            resistance_levels = highs.dropna().unique()
            support_levels = lows.dropna().unique()
            
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.05)
            nearest_support = max([s for s in support_levels if s < current_price], default=current_price * 0.95)
            
            return {
                'current_price': current_price,
                'nearest_resistance': nearest_resistance,
                'nearest_support': nearest_support,
                'resistance_distance': ((nearest_resistance - current_price) / current_price) * 100,
                'support_distance': ((current_price - nearest_support) / current_price) * 100
            }
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {}
    
    def analyze_trend(self, data: pd.DataFrame) -> Dict:
        """Comprehensive trend analysis"""
        try:
            current_price = data['Close'].iloc[-1]
            price_20_ago = data['Close'].iloc[-20] if len(data) >= 20 else current_price
            price_50_ago = data['Close'].iloc[-50] if len(data) >= 50 else current_price
            
            # Calculate trend strength
            short_trend = ((current_price - price_20_ago) / price_20_ago) * 100
            long_trend = ((current_price - price_50_ago) / price_50_ago) * 100
            
            # Determine trend direction
            if short_trend > 2 and long_trend > 5:
                trend_direction = "strong_uptrend"
            elif short_trend > 1 and long_trend > 2:
                trend_direction = "uptrend"
            elif short_trend < -2 and long_trend < -5:
                trend_direction = "strong_downtrend"
            elif short_trend < -1 and long_trend < -2:
                trend_direction = "downtrend"
            else:
                trend_direction = "sideways"
            
            return {
                'trend_direction': trend_direction,
                'short_trend': short_trend,
                'long_trend': long_trend,
                'trend_strength': abs(short_trend) + abs(long_trend)
            }
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {}
    
    def get_comprehensive_analysis(self) -> Dict:
        """Get comprehensive technical analysis"""
        try:
            # Get data
            data = self.get_btc_data()
            
            if data.empty:
                return {"error": "No data available"}
            
            # Calculate all indicators
            rsi = self.calculate_rsi(data)
            macd_data = self.calculate_macd(data)
            ma_data = self.calculate_moving_averages(data)
            bb_data = self.calculate_bollinger_bands(data)
            stoch_data = self.calculate_stochastic(data)
            atr = self.calculate_atr(data)
            volume_data = self.calculate_volume_indicators(data)
            support_resistance = self.get_support_resistance(data)
            trend_analysis = self.analyze_trend(data)
            
            # Get latest values
            current_price = data['Close'].iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'current_volume': current_volume,
                'price_change_24h': ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0,
                'indicators': {
                    'rsi': float(rsi.iloc[-1]) if not rsi.empty else 50,
                    'macd': {
                        'macd': float(macd_data['macd'].iloc[-1]) if not macd_data['macd'].empty else 0,
                        'signal': float(macd_data['macd_signal'].iloc[-1]) if not macd_data['macd_signal'].empty else 0,
                        'histogram': float(macd_data['macd_histogram'].iloc[-1]) if not macd_data['macd_histogram'].empty else 0
                    },
                    'moving_averages': {
                        'sma_20': float(ma_data.get('sma_20', pd.Series()).iloc[-1]) if ma_data.get('sma_20') is not None and not ma_data['sma_20'].empty else current_price,
                        'sma_50': float(ma_data.get('sma_50', pd.Series()).iloc[-1]) if ma_data.get('sma_50') is not None and not ma_data['sma_50'].empty else current_price,
                        'sma_200': float(ma_data.get('sma_200', pd.Series()).iloc[-1]) if ma_data.get('sma_200') is not None and not ma_data['sma_200'].empty else current_price
                    },
                    'bollinger_bands': {
                        'upper': float(bb_data.get('bb_upper', pd.Series()).iloc[-1]) if bb_data.get('bb_upper') is not None and not bb_data['bb_upper'].empty else current_price * 1.02,
                        'middle': float(bb_data.get('bb_middle', pd.Series()).iloc[-1]) if bb_data.get('bb_middle') is not None and not bb_data['bb_middle'].empty else current_price,
                        'lower': float(bb_data.get('bb_lower', pd.Series()).iloc[-1]) if bb_data.get('bb_lower') is not None and not bb_data['bb_lower'].empty else current_price * 0.98,
                        'width': float(bb_data.get('bb_width', pd.Series()).iloc[-1]) if bb_data.get('bb_width') is not None and not bb_data['bb_width'].empty else 0.04
                    },
                    'stochastic': {
                        'k': float(stoch_data.get('stoch_k', pd.Series()).iloc[-1]) if stoch_data.get('stoch_k') is not None and not stoch_data['stoch_k'].empty else 50,
                        'd': float(stoch_data.get('stoch_d', pd.Series()).iloc[-1]) if stoch_data.get('stoch_d') is not None and not stoch_data['stoch_d'].empty else 50
                    },
                    'atr': float(atr.iloc[-1]) if not atr.empty else 0
                },
                'support_resistance': support_resistance,
                'trend_analysis': trend_analysis,
                'signals': self._generate_signals(data, rsi, macd_data, ma_data, bb_data, stoch_data)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {"error": str(e)}
    
    def _generate_signals(self, data: pd.DataFrame, rsi: pd.Series, macd_data: Dict, 
                         ma_data: Dict, bb_data: Dict, stoch_data: Dict) -> Dict:
        """Generate trading signals based on technical indicators"""
        try:
            signals = {}
            
            # RSI signals
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            if current_rsi > 70:
                signals['rsi'] = 'sell'
            elif current_rsi < 30:
                signals['rsi'] = 'buy'
            else:
                signals['rsi'] = 'neutral'
            
            # MACD signals
            current_macd = macd_data['macd'].iloc[-1] if not macd_data['macd'].empty else 0
            current_signal = macd_data['macd_signal'].iloc[-1] if not macd_data['macd_signal'].empty else 0
            if current_macd > current_signal:
                signals['macd'] = 'buy'
            elif current_macd < current_signal:
                signals['macd'] = 'sell'
            else:
                signals['macd'] = 'neutral'
            
            # Moving average signals
            current_price = data['Close'].iloc[-1]
            sma_20 = ma_data.get('sma_20', pd.Series()).iloc[-1] if ma_data.get('sma_20') is not None and not ma_data['sma_20'].empty else current_price
            sma_50 = ma_data.get('sma_50', pd.Series()).iloc[-1] if ma_data.get('sma_50') is not None and not ma_data['sma_50'].empty else current_price
            
            if current_price > sma_20 > sma_50:
                signals['moving_averages'] = 'buy'
            elif current_price < sma_20 < sma_50:
                signals['moving_averages'] = 'sell'
            else:
                signals['moving_averages'] = 'neutral'
            
            # Bollinger Bands signals
            bb_upper = bb_data.get('bb_upper', pd.Series()).iloc[-1] if bb_data.get('bb_upper') is not None and not bb_data['bb_upper'].empty else current_price * 1.02
            bb_lower = bb_data.get('bb_lower', pd.Series()).iloc[-1] if bb_data.get('bb_lower') is not None and not bb_data['bb_lower'].empty else current_price * 0.98
            
            if current_price > bb_upper:
                signals['bollinger_bands'] = 'sell'
            elif current_price < bb_lower:
                signals['bollinger_bands'] = 'buy'
            else:
                signals['bollinger_bands'] = 'neutral'
            
            # Overall signal
            buy_signals = sum(1 for signal in signals.values() if signal == 'buy')
            sell_signals = sum(1 for signal in signals.values() if signal == 'sell')
            
            if buy_signals > sell_signals:
                signals['overall'] = 'buy'
            elif sell_signals > buy_signals:
                signals['overall'] = 'sell'
            else:
                signals['overall'] = 'neutral'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {'overall': 'neutral'}

if __name__ == "__main__":
    # Test the technical analyzer
    analyzer = TechnicalAnalyzer()
    analysis = analyzer.get_comprehensive_analysis()
    print("Technical Analysis Results:")
    print(json.dumps(analysis, indent=2))
