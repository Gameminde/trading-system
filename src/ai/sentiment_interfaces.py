"""
Sentiment Analysis Interfaces - SOLID Implementation
Quantum Trading Revolution - Mission 1B Track 2

This module defines interfaces for sentiment analysis components:
- Single Responsibility: Each interface focuses on specific sentiment tasks
- Open/Closed: Extensible for different models (FinGPT, FinBERT, etc.)
- Liskov Substitution: All implementations must be substitutable
- Interface Segregation: Specific, focused interfaces
- Dependency Inversion: Depend on abstractions, not implementations

Target Performance: 87% forecast accuracy (from research benchmarks)
Integration: Enhance MA Crossover strategy confidence scoring
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


# ========================================
# SENTIMENT DATA STRUCTURES
# ========================================

class SentimentPolarity(Enum):
    """Sentiment polarity classification"""
    VERY_BEARISH = "VERY_BEARISH"    # -1.0 to -0.6
    BEARISH = "BEARISH"              # -0.6 to -0.2
    NEUTRAL = "NEUTRAL"              # -0.2 to +0.2
    BULLISH = "BULLISH"              # +0.2 to +0.6
    VERY_BULLISH = "VERY_BULLISH"    # +0.6 to +1.0


class SentimentSource(Enum):
    """Data source for sentiment analysis"""
    TWITTER = "TWITTER"
    REDDIT = "REDDIT"
    NEWS = "NEWS"
    EARNINGS_CALLS = "EARNINGS_CALLS"
    SEC_FILINGS = "SEC_FILINGS"
    ANALYST_REPORTS = "ANALYST_REPORTS"


@dataclass
class SentimentScore:
    """Immutable sentiment score data structure"""
    symbol: str
    score: float  # -1.0 (very bearish) to +1.0 (very bullish)
    confidence: float  # 0.0 to 1.0 confidence in the score
    polarity: SentimentPolarity
    source: SentimentSource
    timestamp: datetime
    text_sample: Optional[str] = None  # Sample of analyzed text
    metadata: Dict[str, Any] = None


@dataclass
class TextData:
    """Text data container for sentiment analysis"""
    text: str
    source: SentimentSource
    symbol: str
    timestamp: datetime
    url: Optional[str] = None
    author: Optional[str] = None
    engagement_metrics: Dict[str, Any] = None  # likes, shares, comments, etc.


@dataclass
class SentimentSignal:
    """Enhanced sentiment signal for trading integration"""
    symbol: str
    sentiment_score: float  # Aggregated sentiment (-1.0 to +1.0)
    confidence: float  # Overall confidence (0.0 to 1.0)
    source_breakdown: Dict[SentimentSource, float]  # Per-source sentiment
    signal_strength: float  # 0.0 to 1.0 strength for trading decisions
    timestamp: datetime
    validity_period: int  # Minutes this signal remains valid
    supporting_data_count: int  # Number of text samples used
    metadata: Dict[str, Any] = None


# ========================================
# CORE SENTIMENT ANALYSIS INTERFACES
# ========================================

class ISentimentModel(ABC):
    """
    Core sentiment analysis model interface - Single Responsibility
    Each model implementation focuses on one specific approach
    """
    
    @abstractmethod
    def analyze_text(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple[sentiment_score, confidence]
            sentiment_score: -1.0 (bearish) to +1.0 (bullish)
            confidence: 0.0 to 1.0
        """
        pass
    
    @abstractmethod
    def analyze_batch(self, texts: List[str]) -> List[Tuple[float, float]]:
        """Analyze multiple texts efficiently"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and capabilities"""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference"""
        pass


class IDataSource(ABC):
    """
    Data source interface - Single Responsibility
    Handles data collection from specific sources
    """
    
    @abstractmethod
    def collect_recent_data(self, symbol: str, hours_back: int = 24) -> List[TextData]:
        """Collect recent text data for symbol"""
        pass
    
    @abstractmethod
    def collect_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[TextData]:
        """Collect historical text data"""
        pass
    
    @abstractmethod
    def get_source_type(self) -> SentimentSource:
        """Return the source type this collector handles"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if data source is accessible"""
        pass
    
    @abstractmethod
    def get_rate_limits(self) -> Dict[str, Any]:
        """Return API rate limits and usage info"""
        pass


class ISentimentAggregator(ABC):
    """
    Sentiment aggregation interface - Single Responsibility
    Combines multiple sentiment scores into unified signals
    """
    
    @abstractmethod
    def aggregate_scores(self, scores: List[SentimentScore]) -> SentimentSignal:
        """Aggregate multiple sentiment scores into single signal"""
        pass
    
    @abstractmethod
    def weight_by_source(self, scores: List[SentimentScore]) -> Dict[SentimentSource, float]:
        """Calculate source-specific weights"""
        pass
    
    @abstractmethod
    def calculate_confidence(self, scores: List[SentimentScore]) -> float:
        """Calculate overall confidence in aggregated signal"""
        pass
    
    @abstractmethod
    def filter_by_quality(self, scores: List[SentimentScore]) -> List[SentimentScore]:
        """Filter out low-quality or unreliable scores"""
        pass


class ISentimentPreprocessor(ABC):
    """
    Text preprocessing interface - Single Responsibility
    Handles text cleaning and preparation for analysis
    """
    
    @abstractmethod
    def clean_financial_text(self, text: str) -> str:
        """Clean and normalize financial text"""
        pass
    
    @abstractmethod
    def extract_ticker_mentions(self, text: str) -> List[str]:
        """Extract ticker symbols mentioned in text"""
        pass
    
    @abstractmethod
    def remove_noise(self, text: str) -> str:
        """Remove URLs, mentions, hashtags, etc."""
        pass
    
    @abstractmethod
    def normalize_text(self, text: str) -> str:
        """Normalize text for model input"""
        pass


class ISentimentValidator(ABC):
    """
    Sentiment validation interface - Single Responsibility  
    Validates sentiment signals against market data
    """
    
    @abstractmethod
    def validate_against_price_action(
        self, 
        sentiment_signal: SentimentSignal,
        price_data: pd.Series
    ) -> float:
        """Validate sentiment against actual price movements"""
        pass
    
    @abstractmethod
    def calculate_accuracy_metrics(
        self,
        predictions: List[SentimentSignal],
        actual_returns: List[float]
    ) -> Dict[str, float]:
        """Calculate accuracy, precision, recall metrics"""
        pass
    
    @abstractmethod
    def detect_anomalies(self, signals: List[SentimentSignal]) -> List[int]:
        """Detect potentially unreliable signals"""
        pass


# ========================================
# COMPOSITE INTERFACES
# ========================================

class ISentimentPipeline(ABC):
    """
    Main sentiment analysis pipeline - Dependency Inversion
    Orchestrates all sentiment components through abstractions
    """
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize all pipeline components"""
        pass
    
    @abstractmethod
    def generate_sentiment_signal(self, symbol: str) -> SentimentSignal:
        """Generate complete sentiment signal for symbol"""
        pass
    
    @abstractmethod
    def generate_multi_asset_signals(self, symbols: List[str]) -> Dict[str, SentimentSignal]:
        """Generate signals for multiple assets"""
        pass
    
    @abstractmethod
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get overall pipeline health and status"""
        pass
    
    @abstractmethod
    def update_models(self) -> bool:
        """Update/retrain models with recent data"""
        pass


class ITradingSentimentIntegrator(ABC):
    """
    Trading integration interface - Interface Segregation
    Specifically for integrating sentiment with trading strategies
    """
    
    @abstractmethod
    def enhance_trading_signal(
        self,
        technical_confidence: float,
        sentiment_signal: SentimentSignal
    ) -> float:
        """Enhance technical trading signal with sentiment"""
        pass
    
    @abstractmethod
    def calculate_position_adjustment(
        self,
        base_position_size: float,
        sentiment_signal: SentimentSignal
    ) -> float:
        """Adjust position size based on sentiment"""
        pass
    
    @abstractmethod
    def generate_combined_confidence(
        self,
        technical_signals: Dict[str, float],
        sentiment_signals: Dict[str, SentimentSignal]
    ) -> Dict[str, float]:
        """Generate combined confidence scores"""
        pass


# ========================================
# FACTORY INTERFACES
# ========================================

class ISentimentModelFactory(ABC):
    """Factory for creating sentiment models - Open/Closed Principle"""
    
    @abstractmethod
    def create_fingpt_model(self, model_path: Optional[str] = None) -> ISentimentModel:
        """Create FinGPT model instance"""
        pass
    
    @abstractmethod
    def create_finbert_model(self, model_path: Optional[str] = None) -> ISentimentModel:
        """Create FinBERT model instance"""
        pass
    
    @abstractmethod
    def create_ensemble_model(self, models: List[ISentimentModel]) -> ISentimentModel:
        """Create ensemble of multiple models"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available model types"""
        pass


class IDataSourceFactory(ABC):
    """Factory for creating data source collectors"""
    
    @abstractmethod
    def create_twitter_source(self, api_credentials: Dict[str, str]) -> IDataSource:
        """Create Twitter data source"""
        pass
    
    @abstractmethod
    def create_reddit_source(self, api_credentials: Dict[str, str]) -> IDataSource:
        """Create Reddit data source"""
        pass
    
    @abstractmethod
    def create_news_source(self, api_credentials: Dict[str, str]) -> IDataSource:
        """Create news data source"""
        pass
    
    @abstractmethod
    def get_available_sources(self) -> List[SentimentSource]:
        """Get available data sources"""
        pass


# ========================================
# PERFORMANCE MONITORING INTERFACES
# ========================================

class ISentimentMetrics(ABC):
    """
    Performance metrics interface - Single Responsibility
    Tracks sentiment analysis performance and accuracy
    """
    
    @abstractmethod
    def track_prediction_accuracy(
        self,
        symbol: str,
        predicted_sentiment: float,
        actual_return: float,
        time_horizon: int
    ) -> None:
        """Track prediction vs actual performance"""
        pass
    
    @abstractmethod
    def get_accuracy_report(self, symbol: str, days_back: int = 30) -> Dict[str, float]:
        """Get accuracy metrics for symbol"""
        pass
    
    @abstractmethod
    def get_model_performance_comparison(self) -> Dict[str, Dict[str, float]]:
        """Compare performance of different models"""
        pass
    
    @abstractmethod
    def calculate_sentiment_alpha(
        self,
        sentiment_enhanced_returns: List[float],
        benchmark_returns: List[float]
    ) -> float:
        """Calculate alpha generated by sentiment signals"""
        pass


# Module exports
__all__ = [
    # Enums and Data Classes
    'SentimentPolarity', 'SentimentSource', 'SentimentScore', 'TextData', 'SentimentSignal',
    
    # Core Interfaces
    'ISentimentModel', 'IDataSource', 'ISentimentAggregator', 'ISentimentPreprocessor',
    'ISentimentValidator', 'ISentimentPipeline', 'ITradingSentimentIntegrator',
    
    # Factory Interfaces
    'ISentimentModelFactory', 'IDataSourceFactory',
    
    # Performance Monitoring
    'ISentimentMetrics'
]
