"""
Core Interfaces - SOLID Principle Implementation
Quantum Trading Revolution - Algorithmic Trading System

This module defines the core interfaces following SOLID principles:
- Single Responsibility: Each interface has one clear purpose
- Open/Closed: Extensible without modification
- Liskov Substitution: Implementations must be substitutable
- Interface Segregation: Specific, focused interfaces
- Dependency Inversion: Depend on abstractions, not concretions
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


# ========================================
# CORE DATA STRUCTURES
# ========================================

class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(Enum):
    """Order types for execution"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class TradingSignal:
    """Immutable trading signal data structure"""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    price: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class MarketData:
    """Market data container"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    additional_data: Dict[str, Any] = None


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: int
    average_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float


# ========================================
# TRADING STRATEGY INTERFACES
# ========================================

class ITradingStrategy(ABC):
    """
    Core trading strategy interface - Single Responsibility
    Each strategy implementation focuses on one specific approach
    """
    
    @abstractmethod
    def generate_signal(self, market_data: MarketData) -> TradingSignal:
        """
        Generate trading signal based on market data
        
        Args:
            market_data: Current market data
            
        Returns:
            TradingSignal: Signal with confidence score
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy identification name"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return current strategy parameters"""
        pass
    
    @abstractmethod
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update strategy parameters"""
        pass


class IDataProvider(ABC):
    """
    Market data provider interface - Single Responsibility
    Handles all data acquisition and preprocessing
    """
    
    @abstractmethod
    def get_latest_data(self, symbol: str) -> MarketData:
        """Get latest market data for symbol"""
        pass
    
    @abstractmethod
    def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical market data"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check data provider connection status"""
        pass


class IOrderExecutor(ABC):
    """
    Order execution interface - Single Responsibility
    Handles all order management and execution
    """
    
    @abstractmethod
    def place_order(
        self, 
        symbol: str, 
        quantity: int, 
        order_type: OrderType, 
        price: Optional[float] = None
    ) -> str:
        """
        Place trading order
        
        Returns:
            order_id: Unique order identifier
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel existing order"""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get current order status"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get all current positions"""
        pass


class IRiskManager(ABC):
    """
    Risk management interface - Single Responsibility
    Handles all risk assessment and position sizing
    """
    
    @abstractmethod
    def validate_order(
        self, 
        symbol: str, 
        quantity: int, 
        price: float
    ) -> bool:
        """Validate order against risk parameters"""
        pass
    
    @abstractmethod
    def calculate_position_size(
        self, 
        signal: TradingSignal, 
        available_capital: float
    ) -> int:
        """Calculate optimal position size"""
        pass
    
    @abstractmethod
    def check_risk_limits(self, positions: List[Position]) -> bool:
        """Check if current positions exceed risk limits"""
        pass


class IPerformanceAnalyzer(ABC):
    """
    Performance analysis interface - Single Responsibility
    Handles all performance metrics and reporting
    """
    
    @abstractmethod
    def calculate_returns(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate various return metrics"""
        pass
    
    @abstractmethod
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        pass
    
    @abstractmethod
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        pass
    
    @abstractmethod
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        pass


# ========================================
# ADVANCED INTERFACES (Future Extensions)
# ========================================

class ISentimentAnalyzer(ABC):
    """Interface for LLM-based sentiment analysis - Future Phase 1B"""
    
    @abstractmethod
    def analyze_sentiment(self, text_data: List[str]) -> Dict[str, float]:
        """Analyze sentiment from text sources"""
        pass


class ITensorNetworkProcessor(ABC):
    """Interface for tensor network applications - Future Phase 2"""
    
    @abstractmethod
    def optimize_portfolio(
        self, 
        returns_data: pd.DataFrame, 
        constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """Tensor network-based portfolio optimization"""
        pass


class IDeFiArbitrageScanner(ABC):
    """Interface for DeFi arbitrage opportunities - Future Phase 1C"""
    
    @abstractmethod
    def scan_opportunities(self) -> List[Dict[str, Any]]:
        """Scan for DeFi arbitrage opportunities"""
        pass


# ========================================
# COMPOSITE INTERFACES
# ========================================

class ITradingEngine(ABC):
    """
    Main trading engine interface - Dependency Inversion
    Orchestrates all components through abstractions
    """
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize trading engine"""
        pass
    
    @abstractmethod
    def run_trading_loop(self) -> None:
        """Execute main trading loop"""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Graceful shutdown"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        pass


# ========================================
# FACTORY INTERFACES  
# ========================================

class IStrategyFactory(ABC):
    """Factory for creating trading strategies - Open/Closed Principle"""
    
    @abstractmethod
    def create_strategy(self, strategy_type: str, parameters: Dict[str, Any]) -> ITradingStrategy:
        """Create strategy instance"""
        pass
    
    @abstractmethod
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy types"""
        pass


# Module exports
__all__ = [
    'SignalType', 'OrderType', 'TradingSignal', 'MarketData', 'Position',
    'ITradingStrategy', 'IDataProvider', 'IOrderExecutor', 'IRiskManager', 
    'IPerformanceAnalyzer', 'ISentimentAnalyzer', 'ITensorNetworkProcessor',
    'IDeFiArbitrageScanner', 'ITradingEngine', 'IStrategyFactory'
]
