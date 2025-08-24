#!/usr/bin/env python3
"""
LLM Trading Strategy for Bitcoin
Uses Cohere API to analyze market conditions and make automated trading decisions.
"""

import os
import logging
import requests
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
from pydantic import model_validator
from strategy_core import decide_target_allocation

# Add database import for bias persistence
try:
    from db import write_setting, read_setting
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

logger = logging.getLogger(__name__)

class LLMResponseSchema(BaseModel):
    """Strict Pydantic schema for LLM response validation"""
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(..., description="Market sentiment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level 0.0-1.0")
    reasoning: str = Field(..., min_length=10, max_length=1000, description="Detailed reasoning")
    recommended_action: Literal["buy", "sell", "hold"] = Field(..., description="Trading recommendation")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="Risk assessment")
    price_target: Optional[float] = Field(None, ge=0.0, description="Optional target price")
    stop_loss: Optional[float] = Field(None, ge=0.0, description="Optional stop loss price")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return round(v, 3)  # Round to 3 decimal places
    
    @validator('price_target', 'stop_loss')
    def validate_prices(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Price values must be positive')
        return v
    
    @model_validator(mode="after")
    def validate_action_consistency(self):
        # If previous validations failed, skip
        if not getattr(self, 'sentiment', None) or not getattr(self, 'recommended_action', None):
            return self
        if self.sentiment == "bullish" and self.recommended_action == "sell":
            raise ValueError("Bullish sentiment inconsistent with sell action")
        if self.sentiment == "bearish" and self.recommended_action == "buy":
            raise ValueError("Bearish sentiment inconsistent with buy action")
        return self

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
        self.min_confidence = float(os.getenv('MIN_CONFIDENCE', '0.7'))  # Minimum confidence for trades
        self.max_risk_level = os.getenv('MAX_RISK_LEVEL', 'medium')  # Maximum risk level to accept
        self.max_exposure = float(os.getenv('MAX_EXPOSURE', '0.8'))  # cap allocation at 80% of equity
        self.position_size_multiplier = {
            "low": 1.0,
            "medium": 0.7,
            "high": 0.5
        }
        
        # Logging configuration
        self.log_prompts = os.getenv('LOG_LLM_PROMPTS', 'true').lower() == 'true'
        self.log_responses = os.getenv('LOG_LLM_RESPONSES', 'true').lower() == 'true'
        
        logger.info("LLM trading strategy initialized successfully")
        logger.info(f"Min confidence: {self.min_confidence}, Max risk level: {self.max_risk_level}")
    
    def clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp a value between min and max"""
        return max(min_val, min(value, max_val))
    
    def persist_bias(self, symbol: str, bias: float, ts: datetime = None) -> bool:
        """Persist bias allocation to database"""
        try:
            if not DB_AVAILABLE:
                logger.warning("Database not available for bias persistence")
                return False
            
            if ts is None:
                ts = datetime.now()
            
            bias_data = {
                "bias": bias,
                "timestamp": ts.isoformat(),
                "symbol": symbol
            }
            
            write_setting(f"llm_bias_{symbol}", json.dumps(bias_data))
            logger.info(f"💾 Persisted LLM bias for {symbol}: {bias:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to persist bias: {e}")
            return False
    
    def get_persisted_bias(self, symbol: str) -> Optional[float]:
        """Get persisted bias allocation from database"""
        try:
            if not DB_AVAILABLE:
                return None
            
            bias_data = read_setting(f"llm_bias_{symbol}")
            if bias_data:
                data = json.loads(bias_data)
                return data.get("bias")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get persisted bias: {e}")
            return None
    
    def _sanitize_for_logging(self, text: str) -> str:
        """Sanitize text for logging by removing sensitive information"""
        if not text:
            return text
        
        # Remove API keys and other sensitive patterns
        sanitized = text
        
        # Remove API keys (common patterns)
        sanitized = re.sub(r'Bearer\s+[a-zA-Z0-9]{20,}', 'Bearer [REDACTED]', sanitized)
        sanitized = re.sub(r'api[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}["\']?', 'api_key=[REDACTED]', sanitized)
        sanitized = re.sub(r'cohere[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}["\']?', 'cohere_key=[REDACTED]', sanitized)
        
        # Remove other potential secrets
        sanitized = re.sub(r'password["\']?\s*[:=]\s*["\']?[^"\']+["\']?', 'password=[REDACTED]', sanitized)
        sanitized = re.sub(r'secret["\']?\s*[:=]\s*["\']?[^"\']+["\']?', 'secret=[REDACTED]', sanitized)
        
        return sanitized
    
    def _log_prompt(self, prompt: str, context: Dict = None):
        """Log the prompt being sent to LLM"""
        if not self.log_prompts:
            return
        
        sanitized_prompt = self._sanitize_for_logging(prompt)
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "prompt_length": len(prompt),
            "prompt_preview": sanitized_prompt[:500] + "..." if len(sanitized_prompt) > 500 else sanitized_prompt,
            "context": context
        }
        
        logger.info(f"LLM Prompt sent: {json.dumps(log_data, indent=2)}")
    
    def _log_response(self, response: str, validation_result: Dict = None):
        """Log the response received from LLM"""
        if not self.log_responses:
            return
        
        sanitized_response = self._sanitize_for_logging(response)
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "response_length": len(response),
            "response_preview": sanitized_response[:500] + "..." if len(sanitized_response) > 500 else sanitized_response,
            "validation_result": validation_result
        }
        
        logger.info(f"LLM Response received: {json.dumps(log_data, indent=2)}")
    
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

IMPORTANT: Respond ONLY with valid JSON in the exact format below. Do not include any additional text, explanations, or formatting outside the JSON object.

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

Remember: Return ONLY the JSON object, nothing else.
"""
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with the given prompt"""
        
        # Log the prompt being sent
        self._log_prompt(prompt, {"model": self.model, "temperature": 0.3})
        
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
        
        llm_response = result['generations'][0]['text'].strip()
        
        # Log the response received
        self._log_response(llm_response)
        
        return llm_response
    
    def _parse_llm_response(self, response: str, current_price: float) -> MarketAnalysis:
        """Parse LLM response into MarketAnalysis object with strict validation"""
        
        validation_result = {"status": "unknown", "errors": []}
        
        try:
            # Clean the response by removing control characters and extra whitespace
            cleaned_response = response.strip()
            
            # Remove common control characters that can break JSON parsing
            cleaned_response = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_response)
            
            # Extract JSON from response
            json_start = cleaned_response.find('{')
            json_end = cleaned_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = cleaned_response[json_start:json_end]
            
            # Try to fix common JSON formatting issues
            json_str = json_str.replace('\n', ' ').replace('\r', ' ')
            json_str = re.sub(r'\s+', ' ', json_str)  # Normalize whitespace
            
            # Attempt to parse the cleaned JSON
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as json_error:
                logger.warning(f"Initial JSON parsing failed: {json_error}")
                validation_result["errors"].append(f"JSON parse error: {json_error}")
                
                # Try to extract key-value pairs manually as fallback
                data = self._extract_key_value_pairs(json_str)
            
            # Validate with Pydantic schema
            try:
                validated_data = LLMResponseSchema(**data)
                validation_result["status"] = "valid"
                
                # Check confidence threshold
                if validated_data.confidence < self.min_confidence:
                    validation_result["status"] = "rejected"
                    validation_result["errors"].append(f"Confidence {validated_data.confidence} below minimum {self.min_confidence}")
                    logger.warning(f"LLM response rejected: confidence {validated_data.confidence} < {self.min_confidence}")
                
                # Check risk level
                risk_levels = {"low": 1, "medium": 2, "high": 3}
                max_risk = risk_levels.get(self.max_risk_level, 2)
                current_risk = risk_levels.get(validated_data.risk_level, 2)
                
                if current_risk > max_risk:
                    validation_result["status"] = "rejected"
                    validation_result["errors"].append(f"Risk level {validated_data.risk_level} exceeds maximum {self.max_risk_level}")
                    logger.warning(f"LLM response rejected: risk level {validated_data.risk_level} > {self.max_risk_level}")
                
                # Log validation result
                self._log_response(response, validation_result)
                
                # Return validated analysis
                return MarketAnalysis(
                    sentiment=validated_data.sentiment,
                    confidence=validated_data.confidence,
                    reasoning=validated_data.reasoning,
                    recommended_action=validated_data.recommended_action,
                    risk_level=validated_data.risk_level,
                    price_target=validated_data.price_target,
                    stop_loss=validated_data.stop_loss
                )
                
            except Exception as validation_error:
                validation_result["status"] = "invalid"
                validation_result["errors"].append(f"Schema validation failed: {validation_error}")
                logger.error(f"LLM response validation failed: {validation_error}")
                
                # Log the failed validation
                self._log_response(response, validation_result)
                
                # Fall back to manual parsing for backward compatibility
                return self._fallback_parse_response(data, current_price)
            
        except Exception as e:
            validation_result["status"] = "error"
            validation_result["errors"].append(f"Parse error: {e}")
            logger.error(f"Error parsing LLM response: {e}")
            
            # Log the error
            self._log_response(response, validation_result)
            
            # Return neutral analysis as fallback
            return MarketAnalysis(
                sentiment="neutral",
                confidence=0.5,
                reasoning=f"Failed to parse LLM response: {str(e)}",
                recommended_action="hold",
                risk_level="medium"
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
    
    def _fallback_parse_response(self, data: Dict, current_price: float) -> MarketAnalysis:
        """Fallback parsing when Pydantic validation fails"""
        try:
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
            logger.error(f"Fallback parsing failed: {e}")
            return MarketAnalysis(
                sentiment="neutral",
                confidence=0.5,
                reasoning=f"Fallback parsing failed: {str(e)}",
                recommended_action="hold",
                risk_level="medium"
            )

    def _extract_key_value_pairs(self, text: str) -> Dict:
        """Extract key-value pairs from malformed JSON text as fallback"""
        try:
            # Pattern to match key-value pairs
            pattern = r'"([^"]+)"\s*:\s*"([^"]*)"'
            matches = re.findall(pattern, text)
            
            # Also try to match numeric values
            numeric_pattern = r'"([^"]+)"\s*:\s*([0-9.]+)'
            numeric_matches = re.findall(numeric_pattern, text)
            
            data = {}
            
            # Add string matches
            for key, value in matches:
                data[key] = value
            
            # Add numeric matches
            for key, value in numeric_matches:
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = value
            
            # Set defaults for missing required fields
            if 'sentiment' not in data:
                data['sentiment'] = 'neutral'
            if 'confidence' not in data:
                data['confidence'] = 0.5
            if 'reasoning' not in data:
                data['reasoning'] = 'Extracted from malformed response'
            if 'recommended_action' not in data:
                data['recommended_action'] = 'hold'
            if 'risk_level' not in data:
                data['risk_level'] = 'medium'
            
            logger.info(f"Extracted key-value pairs: {data}")
            return data
            
        except Exception as e:
            logger.error(f"Error extracting key-value pairs: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'reasoning': 'Failed to extract data',
                'recommended_action': 'hold',
                'risk_level': 'medium'
            }
    
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
            
            # Convert LLM decision to bias_allocation only (e.g., +0.4 long, -0.3 short/flat)
            bias = self.clamp(analysis.confidence * (+1 if analysis.recommended_action == "buy" else -1), -1, +1) * self.max_exposure
            
            # Persist bias for grid execution
            self.persist_bias("BTC", bias, ts=datetime.now())
            
            # Determine if trade should be executed
            should_trade, reason = self.should_execute_trade(analysis)

            # Compute target exposure via pure decision function
            target_exposure = 0.0
            if analysis.recommended_action in ["buy", "sell"]:
                target_exposure = decide_target_allocation(
                    analysis.recommended_action,
                    analysis.confidence,
                    self.max_exposure,
                )

            return {
                "should_trade": should_trade,
                "action": analysis.recommended_action,
                "reason": reason,
                "sentiment": analysis.sentiment,
                "confidence": analysis.confidence,
                "risk_level": analysis.risk_level,
                # For legacy callers; not used by target allocation path
                "position_size": None,
                # New: target exposure [-max_exposure, +max_exposure]
                "target_exposure": target_exposure,
                # New: bias allocation for grid execution
                "bias_allocation": bias,
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