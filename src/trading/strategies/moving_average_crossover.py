"""
Moving Average Crossover Strategy - SOLID Implementation
Quantum Trading Revolution - Phase 1 QuantConnect Implementation

This strategy implements a classic moving average crossover algorithm:
- Single Responsibility: Focus only on MA crossover logic
- Open/Closed: Extensible via parameter modification
- Liskov Substitution: Implements ITradingStrategy interface
- Interface Segregation: Uses only required interfaces
- Dependency Inversion: Depends on abstractions

Reference Performance Benchmarks:
- Target Sharpe Ratio: > 1.5 (good performance)
- Max Drawdown Target: < 20%
- Win Rate Expected: 45-55% (trend following typical)
- Risk-Adjusted Returns: Beat buy-and-hold benchmark
"""

from typing import Dict, Any, Deque
from collections import deque
from datetime import datetime
import pandas as pd
import numpy as np

from ...core.interfaces import ITradingStrategy, TradingSignal, MarketData, SignalType


class MovingAverageCrossoverStrategy(ITradingStrategy):
    """
    Moving Average Crossover Strategy Implementation
    
    Strategy Logic:
    - BUY signal when fast MA crosses above slow MA
    - SELL signal when fast MA crosses below slow MA
    - HOLD signal when no clear crossover pattern
    
    Key Features:
    - Configurable fast/slow periods
    - Confidence scoring based on MA separation
    - Built-in trend strength assessment
    - Noise filtering via minimum separation threshold
    """
    
    def __init__(
        self, 
        fast_period: int = 10, 
        slow_period: int = 30,
        min_separation_threshold: float = 0.02,  # 2% minimum separation
        confidence_multiplier: float = 1.0
    ):
        """
        Initialize Moving Average Crossover Strategy
        
        Args:
            fast_period: Fast moving average period (default: 10)
            slow_period: Slow moving average period (default: 30)  
            min_separation_threshold: Minimum MA separation for signal (default: 2%)
            confidence_multiplier: Multiplier for confidence calculation (default: 1.0)
        """
        # Parameter validation - defensive programming
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        if fast_period < 2 or slow_period < 2:
            raise ValueError("Both periods must be >= 2")
        if min_separation_threshold < 0:
            raise ValueError("Separation threshold must be >= 0")
            
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.min_separation_threshold = min_separation_threshold
        self.confidence_multiplier = confidence_multiplier
        
        # Price history storage - efficient deque for O(1) operations
        self.price_history: Deque[float] = deque(maxlen=slow_period)
        
        # Moving averages cache
        self._fast_ma_history: Deque[float] = deque(maxlen=3)  # Store last 3 for crossover detection
        self._slow_ma_history: Deque[float] = deque(maxlen=3)
        
        # Performance tracking
        self._signal_count = 0
        self._last_signal_time = None
        self._strategy_initialized = False
        
    def generate_signal(self, market_data: MarketData) -> TradingSignal:
        """
        Generate trading signal based on moving average crossover
        
        Args:
            market_data: Current market data point
            
        Returns:
            TradingSignal: Signal with confidence score and metadata
        """
        # Add new price to history
        self.price_history.append(market_data.close_price)
        
        # Need enough data points for slow MA
        if len(self.price_history) < self.slow_period:
            return self._create_hold_signal(
                market_data, 
                "Insufficient data for MA calculation", 
                confidence=0.0
            )
        
        # Calculate moving averages
        fast_ma = self._calculate_moving_average(self.fast_period)
        slow_ma = self._calculate_moving_average(self.slow_period)
        
        # Store MA values for crossover detection
        self._fast_ma_history.append(fast_ma)
        self._slow_ma_history.append(slow_ma)
        
        # Need at least 2 MA points for crossover detection
        if len(self._fast_ma_history) < 2:
            return self._create_hold_signal(
                market_data,
                "Insufficient MA history for crossover detection",
                confidence=0.0
            )
        
        # Detect crossover patterns
        signal_type = self._detect_crossover()
        
        # Calculate signal confidence
        confidence = self._calculate_confidence(fast_ma, slow_ma, market_data.close_price)
        
        # Create signal with comprehensive metadata
        metadata = {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'ma_separation': abs(fast_ma - slow_ma) / slow_ma,
            'trend_strength': self._assess_trend_strength(),
            'signal_number': self._signal_count,
            'price_above_slow_ma': market_data.close_price > slow_ma,
            'strategy_params': self.get_parameters()
        }
        
        # Update internal state
        if signal_type != SignalType.HOLD:
            self._signal_count += 1
            self._last_signal_time = market_data.timestamp
        
        return TradingSignal(
            symbol=market_data.symbol,
            signal_type=signal_type,
            confidence=confidence,
            timestamp=market_data.timestamp,
            price=market_data.close_price,
            metadata=metadata
        )
    
    def _calculate_moving_average(self, period: int) -> float:
        """Calculate simple moving average for given period"""
        if len(self.price_history) < period:
            return 0.0
        
        return sum(list(self.price_history)[-period:]) / period
    
    def _detect_crossover(self) -> SignalType:
        """
        Detect MA crossover patterns
        
        Returns:
            SignalType: BUY for bullish crossover, SELL for bearish, HOLD otherwise
        """
        if len(self._fast_ma_history) < 2 or len(self._slow_ma_history) < 2:
            return SignalType.HOLD
        
        # Current and previous MA values
        fast_current = self._fast_ma_history[-1]
        fast_previous = self._fast_ma_history[-2]
        slow_current = self._slow_ma_history[-1]
        slow_previous = self._slow_ma_history[-2]
        
        # Bullish crossover: fast MA crosses above slow MA
        if fast_previous <= slow_previous and fast_current > slow_current:
            # Validate minimum separation threshold
            if abs(fast_current - slow_current) / slow_current >= self.min_separation_threshold:
                return SignalType.BUY
        
        # Bearish crossover: fast MA crosses below slow MA
        elif fast_previous >= slow_previous and fast_current < slow_current:
            # Validate minimum separation threshold  
            if abs(fast_current - slow_current) / slow_current >= self.min_separation_threshold:
                return SignalType.SELL
        
        return SignalType.HOLD
    
    def _calculate_confidence(self, fast_ma: float, slow_ma: float, current_price: float) -> float:
        """
        Calculate signal confidence based on multiple factors
        
        Confidence factors:
        1. MA separation (wider = higher confidence)
        2. Price position relative to MAs
        3. Trend consistency
        4. Recent volatility
        
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        if fast_ma == 0 or slow_ma == 0:
            return 0.0
        
        # Factor 1: MA separation (normalized)
        ma_separation = abs(fast_ma - slow_ma) / slow_ma
        separation_confidence = min(ma_separation / 0.05, 1.0)  # Cap at 5% separation
        
        # Factor 2: Price position consistency
        price_consistency = 0.0
        if fast_ma > slow_ma and current_price > fast_ma:  # Bullish alignment
            price_consistency = 0.8
        elif fast_ma < slow_ma and current_price < fast_ma:  # Bearish alignment
            price_consistency = 0.8
        else:
            price_consistency = 0.2  # Mixed signals
        
        # Factor 3: Trend strength assessment
        trend_strength = self._assess_trend_strength()
        
        # Combine factors with weights
        base_confidence = (
            0.4 * separation_confidence +
            0.3 * price_consistency + 
            0.3 * trend_strength
        )
        
        # Apply confidence multiplier and cap
        final_confidence = min(base_confidence * self.confidence_multiplier, 1.0)
        
        return final_confidence
    
    def _assess_trend_strength(self) -> float:
        """
        Assess trend strength based on recent price action
        
        Returns:
            float: Trend strength score between 0.0 and 1.0
        """
        if len(self.price_history) < 5:
            return 0.0
        
        # Calculate recent price momentum
        recent_prices = list(self.price_history)[-5:]
        price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
        
        # Trend consistency - same direction moves
        positive_moves = sum(1 for change in price_changes if change > 0)
        negative_moves = sum(1 for change in price_changes if change < 0)
        
        trend_consistency = max(positive_moves, negative_moves) / len(price_changes)
        
        # Price volatility - lower volatility = stronger trend
        if len(recent_prices) > 1:
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            volatility_score = max(0.0, 1.0 - volatility * 10)  # Normalize volatility
        else:
            volatility_score = 0.0
        
        return (0.6 * trend_consistency + 0.4 * volatility_score)
    
    def _create_hold_signal(self, market_data: MarketData, reason: str, confidence: float) -> TradingSignal:
        """Helper method to create HOLD signals with metadata"""
        metadata = {
            'hold_reason': reason,
            'strategy_params': self.get_parameters(),
            'data_points_available': len(self.price_history)
        }
        
        return TradingSignal(
            symbol=market_data.symbol,
            signal_type=SignalType.HOLD,
            confidence=confidence,
            timestamp=market_data.timestamp,
            price=market_data.close_price,
            metadata=metadata
        )
    
    # ITradingStrategy interface implementation
    
    def get_strategy_name(self) -> str:
        """Return strategy identification name"""
        return f"MovingAverageCrossover_{self.fast_period}_{self.slow_period}"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return current strategy parameters"""
        return {
            'strategy_type': 'MovingAverageCrossover',
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'min_separation_threshold': self.min_separation_threshold,
            'confidence_multiplier': self.confidence_multiplier,
            'signal_count': self._signal_count,
            'last_signal_time': self._last_signal_time,
            'data_points_collected': len(self.price_history)
        }
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update strategy parameters - Open/Closed Principle
        
        Args:
            parameters: Dictionary of parameters to update
        """
        if 'fast_period' in parameters:
            new_fast = parameters['fast_period']
            if new_fast >= self.slow_period or new_fast < 2:
                raise ValueError("Invalid fast_period value")
            self.fast_period = new_fast
            
        if 'slow_period' in parameters:
            new_slow = parameters['slow_period']
            if new_slow <= self.fast_period or new_slow < 2:
                raise ValueError("Invalid slow_period value")
            self.slow_period = new_slow
            
        if 'min_separation_threshold' in parameters:
            threshold = parameters['min_separation_threshold']
            if threshold < 0:
                raise ValueError("Separation threshold must be >= 0")
            self.min_separation_threshold = threshold
            
        if 'confidence_multiplier' in parameters:
            multiplier = parameters['confidence_multiplier']
            if multiplier <= 0:
                raise ValueError("Confidence multiplier must be > 0")
            self.confidence_multiplier = multiplier
        
        # Reset history buffers if periods changed
        if 'fast_period' in parameters or 'slow_period' in parameters:
            max_period = max(self.fast_period, self.slow_period)
            self.price_history = deque(list(self.price_history), maxlen=max_period)
            self._fast_ma_history.clear()
            self._slow_ma_history.clear()
    
    # Performance and debugging methods
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get detailed debug information for strategy analysis"""
        return {
            'price_history_length': len(self.price_history),
            'recent_prices': list(self.price_history)[-5:] if self.price_history else [],
            'fast_ma_history': list(self._fast_ma_history),
            'slow_ma_history': list(self._slow_ma_history),
            'signals_generated': self._signal_count,
            'last_signal_timestamp': self._last_signal_time,
            'current_parameters': self.get_parameters()
        }
    
    def reset_state(self) -> None:
        """Reset strategy internal state - useful for backtesting"""
        self.price_history.clear()
        self._fast_ma_history.clear()
        self._slow_ma_history.clear()
        self._signal_count = 0
        self._last_signal_time = None
        self._strategy_initialized = False
