#!/usr/bin/env python3
"""
Risk Management Module for Bitcoin Trading
Comprehensive risk assessment and management for AI-driven trading.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class RiskAssessment:
    """Risk assessment result"""
    overall_risk: str  # "low", "medium", "high", "extreme"
    risk_score: float  # 0.0 to 1.0
    max_position_size: float
    recommended_stop_loss: float
    recommended_take_profit: float
    risk_factors: Dict[str, float]
    risk_recommendations: List[str]

class RiskManager:
    """Advanced risk management for Bitcoin trading"""
    
    def __init__(self):
        """Initialize risk manager"""
        # Risk parameters
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.max_position_risk = 0.02  # 2% max risk per position
        self.max_portfolio_risk = 0.10  # 10% max portfolio risk
        self.volatility_threshold = 0.03  # 3% volatility threshold
        
        # Position sizing parameters
        self.base_position_size = 1000.0
        self.min_position_size = 10.0
        self.max_position_size = 10000.0
        
        # Stop loss and take profit parameters
        self.default_stop_loss_pct = 0.05  # 5% default stop loss
        self.default_take_profit_pct = 0.10  # 10% default take profit
        self.trailing_stop_pct = 0.03  # 3% trailing stop
        
        # Risk scoring weights
        self.risk_weights = {
            'volatility': 0.25,
            'market_conditions': 0.20,
            'position_concentration': 0.15,
            'liquidity': 0.15,
            'news_sentiment': 0.10,
            'technical_indicators': 0.10,
            'time_of_day': 0.05
        }
        
        logger.info("Risk manager initialized successfully")
    
    def assess_market_risk(self, 
                          current_price: float,
                          price_change_24h: float,
                          volume_24h: float,
                          volatility: float,
                          news_sentiment: str,
                          technical_indicators: Dict = None) -> RiskAssessment:
        """Comprehensive market risk assessment"""
        
        try:
            risk_factors = {}
            risk_recommendations = []
            
            # 1. Volatility Risk
            volatility_risk = self._calculate_volatility_risk(volatility, price_change_24h)
            risk_factors['volatility'] = volatility_risk
            
            # 2. Market Conditions Risk
            market_risk = self._calculate_market_conditions_risk(current_price, price_change_24h, volume_24h)
            risk_factors['market_conditions'] = market_risk
            
            # 3. Position Concentration Risk
            concentration_risk = self._calculate_concentration_risk()
            risk_factors['position_concentration'] = concentration_risk
            
            # 4. Liquidity Risk
            liquidity_risk = self._calculate_liquidity_risk(volume_24h, current_price)
            risk_factors['liquidity'] = liquidity_risk
            
            # 5. News Sentiment Risk
            sentiment_risk = self._calculate_sentiment_risk(news_sentiment)
            risk_factors['news_sentiment'] = sentiment_risk
            
            # 6. Technical Indicators Risk
            technical_risk = self._calculate_technical_risk(technical_indicators)
            risk_factors['technical_indicators'] = technical_risk
            
            # 7. Time of Day Risk
            time_risk = self._calculate_time_risk()
            risk_factors['time_of_day'] = time_risk
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(risk_factors)
            
            # Determine risk level
            risk_level = self._determine_risk_level(overall_risk_score)
            
            # Calculate position sizing
            max_position_size = self._calculate_max_position_size(overall_risk_score, current_price)
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_risk_levels(current_price, overall_risk_score)
            
            # Generate recommendations
            risk_recommendations = self._generate_risk_recommendations(risk_factors, overall_risk_score)
            
            return RiskAssessment(
                overall_risk=risk_level,
                risk_score=overall_risk_score,
                max_position_size=max_position_size,
                recommended_stop_loss=stop_loss,
                recommended_take_profit=take_profit,
                risk_factors=risk_factors,
                risk_recommendations=risk_recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in market risk assessment: {e}")
            return RiskAssessment(
                overall_risk="high",
                risk_score=0.8,
                max_position_size=self.min_position_size,
                recommended_stop_loss=current_price * 0.95,
                recommended_take_profit=current_price * 1.05,
                risk_factors={},
                risk_recommendations=["Risk assessment failed - use conservative approach"]
            )
    
    def _calculate_volatility_risk(self, volatility: float, price_change_24h: float) -> float:
        """Calculate volatility risk score"""
        try:
            # Normalize volatility (0-1 scale)
            normalized_volatility = min(volatility / self.volatility_threshold, 1.0)
            
            # Consider price change magnitude
            price_change_risk = min(abs(price_change_24h) / 10.0, 1.0)  # 10% change = max risk
            
            # Combine volatility and price change
            volatility_risk = (normalized_volatility * 0.7) + (price_change_risk * 0.3)
            
            return min(volatility_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating volatility risk: {e}")
            return 0.5
    
    def _calculate_market_conditions_risk(self, current_price: float, price_change_24h: float, volume_24h: float) -> float:
        """Calculate market conditions risk"""
        try:
            risk_score = 0.0
            
            # Price trend risk
            if abs(price_change_24h) > 5.0:
                risk_score += 0.4
            elif abs(price_change_24h) > 2.0:
                risk_score += 0.2
            
            # Volume risk (low volume = higher risk)
            normalized_volume = volume_24h / 1000000.0  # Normalize to millions
            if normalized_volume < 0.5:
                risk_score += 0.3
            elif normalized_volume < 1.0:
                risk_score += 0.1
            
            # Price level risk (extreme prices = higher risk)
            if current_price > 100000 or current_price < 10000:
                risk_score += 0.2
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating market conditions risk: {e}")
            return 0.5
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate position concentration risk"""
        try:
            # This would typically check current positions
            # For now, return a default value
            return 0.3
            
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {e}")
            return 0.5
    
    def _calculate_liquidity_risk(self, volume_24h: float, current_price: float) -> float:
        """Calculate liquidity risk"""
        try:
            # Normalize volume to price
            volume_price_ratio = volume_24h / current_price
            
            # Low liquidity = higher risk
            if volume_price_ratio < 1000:
                return 0.8
            elif volume_price_ratio < 5000:
                return 0.5
            else:
                return 0.2
                
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {e}")
            return 0.5
    
    def _calculate_sentiment_risk(self, news_sentiment: str) -> float:
        """Calculate sentiment risk"""
        try:
            sentiment_scores = {
                'positive': 0.2,
                'neutral': 0.5,
                'negative': 0.8
            }
            
            return sentiment_scores.get(news_sentiment.lower(), 0.5)
            
        except Exception as e:
            logger.error(f"Error calculating sentiment risk: {e}")
            return 0.5
    
    def _calculate_technical_risk(self, technical_indicators: Dict = None) -> float:
        """Calculate technical indicators risk"""
        try:
            if not technical_indicators:
                return 0.5
            
            risk_score = 0.0
            
            # RSI risk
            rsi = technical_indicators.get('rsi', 50)
            if rsi > 80 or rsi < 20:
                risk_score += 0.3
            elif rsi > 70 or rsi < 30:
                risk_score += 0.1
            
            # MACD risk
            macd = technical_indicators.get('macd', {})
            if isinstance(macd, dict):
                macd_value = macd.get('macd', 0)
                if abs(macd_value) > 100:
                    risk_score += 0.2
            
            # Bollinger Bands risk
            bb = technical_indicators.get('bollinger_bands', {})
            if isinstance(bb, dict):
                bb_width = bb.get('width', 0.04)
                if bb_width > 0.08:
                    risk_score += 0.2
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating technical risk: {e}")
            return 0.5
    
    def _calculate_time_risk(self) -> float:
        """Calculate time of day risk"""
        try:
            current_hour = datetime.now().hour
            
            # Higher risk during low liquidity hours (weekends, early morning)
            if current_hour < 6 or current_hour > 22:
                return 0.7
            elif current_hour < 9 or current_hour > 17:
                return 0.5
            else:
                return 0.3
                
        except Exception as e:
            logger.error(f"Error calculating time risk: {e}")
            return 0.5
    
    def _calculate_overall_risk_score(self, risk_factors: Dict[str, float]) -> float:
        """Calculate overall risk score from individual factors"""
        try:
            weighted_score = 0.0
            total_weight = 0.0
            
            for factor, score in risk_factors.items():
                weight = self.risk_weights.get(factor, 0.1)
                weighted_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_score / total_weight
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating overall risk score: {e}")
            return 0.5
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score"""
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.6:
            return "medium"
        elif risk_score < 0.8:
            return "high"
        else:
            return "extreme"
    
    def _calculate_max_position_size(self, risk_score: float, current_price: float) -> float:
        """Calculate maximum position size based on risk"""
        try:
            # Risk-adjusted position size
            risk_multiplier = 1.0 - risk_score
            
            # Base position size adjusted for risk
            adjusted_size = self.base_position_size * risk_multiplier
            
            # Ensure within bounds
            adjusted_size = max(adjusted_size, self.min_position_size)
            adjusted_size = min(adjusted_size, self.max_position_size)
            
            return round(adjusted_size, 2)
            
        except Exception as e:
            logger.error(f"Error calculating max position size: {e}")
            return self.min_position_size
    
    def _calculate_risk_levels(self, current_price: float, risk_score: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        try:
            # Adjust stop loss and take profit based on risk
            risk_multiplier = 1.0 + risk_score
            
            stop_loss_pct = self.default_stop_loss_pct * risk_multiplier
            take_profit_pct = self.default_take_profit_pct * risk_multiplier
            
            stop_loss = current_price * (1.0 - stop_loss_pct)
            take_profit = current_price * (1.0 + take_profit_pct)
            
            return round(stop_loss, 2), round(take_profit, 2)
            
        except Exception as e:
            logger.error(f"Error calculating risk levels: {e}")
            return current_price * 0.95, current_price * 1.05
    
    def _generate_risk_recommendations(self, risk_factors: Dict[str, float], overall_risk_score: float) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        try:
            # Overall risk recommendations
            if overall_risk_score > 0.8:
                recommendations.append("Extreme risk - consider reducing position sizes or avoiding trades")
            elif overall_risk_score > 0.6:
                recommendations.append("High risk - use conservative position sizing and tight stop losses")
            elif overall_risk_score > 0.4:
                recommendations.append("Moderate risk - standard position sizing recommended")
            else:
                recommendations.append("Low risk - normal trading conditions")
            
            # Specific factor recommendations
            if risk_factors.get('volatility', 0) > 0.7:
                recommendations.append("High volatility - consider wider stop losses")
            
            if risk_factors.get('liquidity', 0) > 0.7:
                recommendations.append("Low liquidity - consider smaller position sizes")
            
            if risk_factors.get('market_conditions', 0) > 0.7:
                recommendations.append("Unfavorable market conditions - exercise caution")
            
            if risk_factors.get('technical_indicators', 0) > 0.7:
                recommendations.append("Technical indicators suggest high risk - review analysis")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating risk recommendations: {e}")
            return ["Risk assessment error - use conservative approach"]
    
    def validate_trade(self, 
                      trade_amount: float,
                      current_price: float,
                      risk_assessment: RiskAssessment) -> Tuple[bool, str]:
        """Validate if a trade meets risk management criteria"""
        
        try:
            # Check position size limits
            if trade_amount > risk_assessment.max_position_size:
                return False, f"Trade amount ${trade_amount} exceeds maximum position size ${risk_assessment.max_position_size}"
            
            # Check minimum position size
            if trade_amount < self.min_position_size:
                return False, f"Trade amount ${trade_amount} below minimum position size ${self.min_position_size}"
            
            # Check risk level
            if risk_assessment.overall_risk == "extreme":
                return False, "Market risk is extreme - trading not recommended"
            
            # Check daily loss limits (would need to track daily P&L)
            # This is a placeholder for actual implementation
            
            return True, "Trade validated successfully"
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False, f"Trade validation error: {str(e)}"
    
    def get_risk_report(self, risk_assessment: RiskAssessment) -> Dict:
        """Generate comprehensive risk report"""
        
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_risk": risk_assessment.overall_risk,
                "risk_score": risk_assessment.risk_score,
                "max_position_size": risk_assessment.max_position_size,
                "recommended_stop_loss": risk_assessment.recommended_stop_loss,
                "recommended_take_profit": risk_assessment.recommended_take_profit,
                "risk_factors": risk_assessment.risk_factors,
                "risk_recommendations": risk_assessment.risk_recommendations,
                "risk_weights": self.risk_weights
            }
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Test the risk manager
    risk_manager = RiskManager()
    
    # Sample risk assessment
    assessment = risk_manager.assess_market_risk(
        current_price=45000.0,
        price_change_24h=2.5,
        volume_24h=1000000.0,
        volatility=0.04,
        news_sentiment="positive",
        technical_indicators={
            'rsi': 65,
            'macd': {'macd': 50, 'signal': 45},
            'bollinger_bands': {'width': 0.05}
        }
    )
    
    print("Risk Assessment Results:")
    print(json.dumps(risk_manager.get_risk_report(assessment), indent=2))
