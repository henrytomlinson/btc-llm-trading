#!/usr/bin/env python3
"""
LLM Trading Strategy for Bitcoin
Uses Cohere API to analyze market conditions and make automated trading decisions.
"""

import os
import logging
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketAnalysis:
    """Market analysis result from LLM"""
    sentiment: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    recommended_action: str  # "buy", "sell", "hold"
    risk_level: str  # "low", "medium", "high"
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None

class LLMTradingStrategy:
    """LLM-powered trading strategy for Bitcoin"""
    
    def __init__(self):
        """Initialize the LLM trading strategy"""
        self.cohere_key = os.getenv('COHERE_KEY')
        if not self.cohere_key:
            raise ValueError("Cohere API key not found")
        
        self.base_url = "https://api.cohere.ai/v1"
        self.model = "command"  # Cohere's latest model
        
        # Trading parameters
        self.min_confidence = 0.7  # Minimum confidence for trades
        self.max_risk_level = "medium"  # Maximum risk level to accept
        self.position_size_multiplier = {
            "low": 1.0,
            "medium": 0.7,
            "high": 0.5
        }
        
        logger.info("LLM trading strategy initialized successfully")
    
    def analyze_market_conditions(self, 
                                current_price: float,
                                price_change_24h: float,
                                volume_24h: float,
                                news_sentiment: str,
                                technical_indicators: Dict = None) -> MarketAnalysis:
        """Analyze market conditions using LLM"""
        
        try:
            # Prepare market data for LLM analysis
            market_context = self._prepare_market_context(
                current_price, price_change_24h, volume_24h, 
                news_sentiment, technical_indicators
            )
            
            # Create prompt for LLM analysis
            prompt = self._create_analysis_prompt(market_context)
            
            # Get LLM analysis
            response = self._query_llm(prompt)
            
            # Parse LLM response
            analysis = self._parse_llm_response(response, current_price)
            
            logger.info(f"LLM Analysis: {analysis.sentiment} sentiment, {analysis.confidence:.2f} confidence")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in LLM market analysis: {e}")
            # Return neutral analysis as fallback
            return MarketAnalysis(
                sentiment="neutral",
                confidence=0.5,
                reasoning="LLM analysis failed, using neutral stance",
                recommended_action="hold",
                risk_level="medium"
            )
    
    def _prepare_market_context(self, 
                               current_price: float,
                               price_change_24h: float,
                               volume_24h: float,
                               news_sentiment: str,
                               technical_indicators: Dict = None) -> str:
        """Prepare market context for LLM analysis"""
        
        context = f"""
Current Bitcoin Market Data:
- Price: ${current_price:,.2f}
- 24h Change: {price_change_24h:+.2f}%
- 24h Volume: ${volume_24h:,.0f}
- News Sentiment: {news_sentiment}
"""
        
        if technical_indicators:
            context += f"""
Technical Indicators:
- RSI: {technical_indicators.get('rsi', 'N/A')}
- MACD: {technical_indicators.get('macd', 'N/A')}
- Moving Averages: {technical_indicators.get('ma', 'N/A')}
"""
        
        return context
    
    def _create_analysis_prompt(self, market_context: str) -> str:
        """Create prompt for LLM market analysis"""
        
        return f"""
You are a professional Bitcoin trading analyst. Analyze the following market data and provide trading recommendations.

{market_context}

Please provide your analysis in the following JSON format:
{{
    "sentiment": "bullish|bearish|neutral",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your analysis",
    "recommended_action": "buy|sell|hold",
    "risk_level": "low|medium|high",
    "price_target": "Optional target price",
    "stop_loss": "Optional stop loss price"
}}

Consider:
1. Price momentum and trend
2. Volume analysis
3. News sentiment impact
4. Technical indicators
5. Risk management
6. Market volatility

Focus on Bitcoin's unique characteristics as a digital asset and cryptocurrency market dynamics.
"""
    
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
            "temperature": 0.3,  # Lower temperature for more consistent analysis
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods": "NONE"
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            headers=headers,
            json=data,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result['generations'][0]['text'].strip()
    
    def _parse_llm_response(self, response: str, current_price: float) -> MarketAnalysis:
        """Parse LLM response into MarketAnalysis object"""
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # Parse values with defaults
            sentiment = data.get('sentiment', 'neutral')
            confidence = float(data.get('confidence', 0.5))
            reasoning = data.get('reasoning', 'No reasoning provided')
            recommended_action = data.get('recommended_action', 'hold')
            risk_level = data.get('risk_level', 'medium')
            
            # Parse optional numeric values
            price_target = None
            stop_loss = None
            
            if 'price_target' in data and data['price_target']:
                try:
                    price_target = float(data['price_target'])
                except (ValueError, TypeError):
                    pass
            
            if 'stop_loss' in data and data['stop_loss']:
                try:
                    stop_loss = float(data['stop_loss'])
                except (ValueError, TypeError):
                    pass
            
            return MarketAnalysis(
                sentiment=sentiment,
                confidence=confidence,
                reasoning=reasoning,
                recommended_action=recommended_action,
                risk_level=risk_level,
                price_target=price_target,
                stop_loss=stop_loss
            )
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Raw response: {response}")
            
            # Return neutral analysis as fallback
            return MarketAnalysis(
                sentiment="neutral",
                confidence=0.5,
                reasoning=f"Failed to parse LLM response: {str(e)}",
                recommended_action="hold",
                risk_level="medium"
            )
    
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
        
        # Check if action is valid
        if analysis.recommended_action not in ["buy", "sell", "hold"]:
            return False, f"Invalid action: {analysis.recommended_action}"
        
        # Don't execute if recommendation is "hold"
        if analysis.recommended_action == "hold":
            return False, "LLM recommends holding"
        
        return True, f"Execute {analysis.recommended_action.upper()}: {analysis.reasoning}"
    
    def get_position_size(self, analysis: MarketAnalysis, base_amount: float) -> float:
        """Calculate position size based on LLM analysis"""
        
        # Base multiplier from risk level
        risk_multiplier = self.position_size_multiplier.get(analysis.risk_level, 0.7)
        
        # Confidence multiplier
        confidence_multiplier = analysis.confidence
        
        # Calculate final position size
        position_size = base_amount * risk_multiplier * confidence_multiplier
        
        # Ensure minimum position size
        min_position = 10.0  # Minimum $10 position
        position_size = max(position_size, min_position)
        
        return round(position_size, 2)
    
    def generate_trading_signal(self, 
                               current_price: float,
                               price_change_24h: float,
                               volume_24h: float,
                               news_sentiment: str,
                               technical_indicators: Dict = None) -> Dict:
        """Generate trading signal using LLM analysis"""
        
        try:
            # Get LLM analysis
            analysis = self.analyze_market_conditions(
                current_price, price_change_24h, volume_24h, 
                news_sentiment, technical_indicators
            )
            
            # Determine if trade should be executed
            should_trade, reason = self.should_execute_trade(analysis)
            
            # Calculate position size if trade should be executed
            position_size = None
            if should_trade and analysis.recommended_action in ["buy", "sell"]:
                base_amount = 1000.0  # Base $1000 position
                position_size = self.get_position_size(analysis, base_amount)
            
            return {
                "should_trade": should_trade,
                "action": analysis.recommended_action,
                "reason": reason,
                "sentiment": analysis.sentiment,
                "confidence": analysis.confidence,
                "risk_level": analysis.risk_level,
                "position_size": position_size,
                "price_target": analysis.price_target,
                "stop_loss": analysis.stop_loss,
                "analysis": analysis.reasoning,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return {
                "should_trade": False,
                "action": "hold",
                "reason": f"Error in LLM analysis: {str(e)}",
                "sentiment": "neutral",
                "confidence": 0.0,
                "risk_level": "high",
                "position_size": None,
                "timestamp": datetime.now().isoformat()
            } 