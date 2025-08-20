"""
Dual Track Monitoring Dashboard
Quantum Trading Revolution - Week 2 Execution

Real-time monitoring for:
- TRACK 1: QuantConnect trading performance
- TRACK 2: FinGPT sentiment pipeline performance
- Combined: Integration metrics and enhancement tracking

Target: Track progress toward Sharpe ratio improvement 0.801 → 1.5+
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradingMetrics:
    """Track 1: Trading performance metrics"""
    timestamp: datetime
    portfolio_value: float
    daily_return: float
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    trades_executed: int
    win_rate: float
    current_position: str
    last_signal: str
    signal_confidence: float


@dataclass
class SentimentMetrics:
    """Track 2: Sentiment pipeline metrics"""
    timestamp: datetime
    total_sentiment_signals: int
    sentiment_accuracy: float
    data_source_uptime: Dict[str, float]
    processing_latency: float
    signal_quality_score: float
    api_usage: Dict[str, int]
    model_confidence: float
    active_assets: List[str]


@dataclass 
class CombinedMetrics:
    """Integration tracking metrics"""
    timestamp: datetime
    sentiment_enhanced_signals: int
    sharpe_improvement: float
    signal_quality_enhancement: float
    risk_adjustment_impact: float
    integration_uptime: float


class DualTrackMonitoringDashboard:
    """
    Comprehensive monitoring dashboard for dual track execution
    Tracks both trading performance and sentiment pipeline metrics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Storage for metrics history
        self.trading_history: List[TradingMetrics] = []
        self.sentiment_history: List[SentimentMetrics] = []
        self.combined_history: List[CombinedMetrics] = []
        
        # Performance targets
        self.targets = {
            'sharpe_target': 1.5,
            'sharpe_baseline': 0.801,
            'sentiment_accuracy_target': 0.87,
            'max_drawdown_limit': 0.15,
            'min_uptime': 0.99
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'drawdown_warning': 0.12,  # 12% drawdown warning
            'sharpe_degradation': 0.6,  # Below 0.6 Sharpe
            'sentiment_accuracy_low': 0.75,  # Below 75% accuracy
            'system_downtime': 0.95  # Below 95% uptime
        }
    
    def update_trading_metrics(self, metrics: TradingMetrics) -> None:
        """Update trading performance metrics (Track 1)"""
        self.trading_history.append(metrics)
        self.logger.info(f"Trading metrics updated: Portfolio ${metrics.portfolio_value:,.2f}, Sharpe {metrics.sharpe_ratio:.3f}")
        
        # Check for alerts
        self._check_trading_alerts(metrics)
    
    def update_sentiment_metrics(self, metrics: SentimentMetrics) -> None:
        """Update sentiment pipeline metrics (Track 2)"""
        self.sentiment_history.append(metrics)
        self.logger.info(f"Sentiment metrics updated: {metrics.total_sentiment_signals} signals, {metrics.sentiment_accuracy:.1%} accuracy")
        
        # Check for alerts
        self._check_sentiment_alerts(metrics)
    
    def update_combined_metrics(self, metrics: CombinedMetrics) -> None:
        """Update integration metrics (Combined tracking)"""
        self.combined_history.append(metrics)
        self.logger.info(f"Integration metrics updated: Sharpe improvement {metrics.sharpe_improvement:.3f}")
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate comprehensive daily status report"""
        
        if not self.trading_history or not self.sentiment_history:
            return {'error': 'Insufficient data for report'}
        
        # Latest metrics
        latest_trading = self.trading_history[-1]
        latest_sentiment = self.sentiment_history[-1]
        latest_combined = self.combined_history[-1] if self.combined_history else None
        
        # Calculate progress metrics
        sharpe_progress = (latest_trading.sharpe_ratio - self.targets['sharpe_baseline']) / (self.targets['sharpe_target'] - self.targets['sharpe_baseline'])
        sentiment_progress = latest_sentiment.sentiment_accuracy / self.targets['sentiment_accuracy_target']
        
        report = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'summary': {
                'overall_status': self._calculate_overall_status(),
                'week_2_progress': f"{max(0, min(100, sharpe_progress * 100)):.1f}%"
            },
            
            # Track 1 - Trading Performance
            'track_1_trading': {
                'status': '🟢 OPERATIONAL' if latest_trading.portfolio_value > 99000 else '🔴 ISSUES',
                'portfolio_value': f"${latest_trading.portfolio_value:,.2f}",
                'daily_return': f"{latest_trading.daily_return:.2%}",
                'cumulative_return': f"{latest_trading.cumulative_return:.2%}",
                'sharpe_ratio': f"{latest_trading.sharpe_ratio:.3f}",
                'sharpe_vs_target': f"{latest_trading.sharpe_ratio:.3f} / {self.targets['sharpe_target']:.3f}",
                'max_drawdown': f"{latest_trading.max_drawdown:.2%}",
                'trades_executed': latest_trading.trades_executed,
                'win_rate': f"{latest_trading.win_rate:.1%}",
                'current_position': latest_trading.current_position,
                'last_signal': latest_trading.last_signal
            },
            
            # Track 2 - Sentiment Pipeline
            'track_2_sentiment': {
                'status': '🟢 OPERATIONAL' if latest_sentiment.sentiment_accuracy > 0.75 else '🔴 ISSUES',
                'total_signals': latest_sentiment.total_sentiment_signals,
                'accuracy': f"{latest_sentiment.sentiment_accuracy:.1%}",
                'accuracy_vs_target': f"{latest_sentiment.sentiment_accuracy:.1%} / {self.targets['sentiment_accuracy_target']:.1%}",
                'processing_latency': f"{latest_sentiment.processing_latency:.1f}s",
                'signal_quality': f"{latest_sentiment.signal_quality_score:.2f}",
                'model_confidence': f"{latest_sentiment.model_confidence:.2f}",
                'active_assets': latest_sentiment.active_assets,
                'data_sources': latest_sentiment.data_source_uptime
            },
            
            # Combined Integration
            'integration': {
                'status': '🟢 READY' if latest_combined and latest_combined.integration_uptime > 0.95 else '🔄 DEVELOPING',
                'enhanced_signals': latest_combined.sentiment_enhanced_signals if latest_combined else 0,
                'sharpe_improvement': f"{latest_combined.sharpe_improvement:.3f}" if latest_combined else 'N/A',
                'signal_enhancement': f"{latest_combined.signal_quality_enhancement:.2%}" if latest_combined else 'N/A',
                'integration_uptime': f"{latest_combined.integration_uptime:.1%}" if latest_combined else 'N/A'
            },
            
            # Performance vs Targets
            'progress_tracking': {
                'sharpe_baseline': f"{self.targets['sharpe_baseline']:.3f}",
                'sharpe_current': f"{latest_trading.sharpe_ratio:.3f}",
                'sharpe_target': f"{self.targets['sharpe_target']:.3f}",
                'sharpe_progress': f"{sharpe_progress:.1%}",
                'sentiment_progress': f"{sentiment_progress:.1%}",
                'overall_week2_progress': f"{(sharpe_progress + sentiment_progress) / 2:.1%}"
            },
            
            # Alerts and Issues
            'alerts': self._get_active_alerts(),
            
            # Recommendations
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall system status"""
        if not self.trading_history or not self.sentiment_history:
            return '🔄 INITIALIZING'
        
        latest_trading = self.trading_history[-1]
        latest_sentiment = self.sentiment_history[-1]
        
        # Check critical metrics
        trading_ok = (latest_trading.max_drawdown < self.alert_thresholds['drawdown_warning'] and
                     latest_trading.sharpe_ratio > self.alert_thresholds['sharpe_degradation'])
        
        sentiment_ok = (latest_sentiment.sentiment_accuracy > self.alert_thresholds['sentiment_accuracy_low'] and
                       latest_sentiment.signal_quality_score > 0.6)
        
        if trading_ok and sentiment_ok:
            return '🟢 EXCELLENT'
        elif trading_ok or sentiment_ok:
            return '🟡 GOOD'
        else:
            return '🔴 NEEDS ATTENTION'
    
    def _check_trading_alerts(self, metrics: TradingMetrics) -> None:
        """Check for trading-related alerts"""
        alerts = []
        
        if metrics.max_drawdown > self.alert_thresholds['drawdown_warning']:
            alerts.append(f"🚨 High drawdown: {metrics.max_drawdown:.2%}")
        
        if metrics.sharpe_ratio < self.alert_thresholds['sharpe_degradation']:
            alerts.append(f"⚠️ Low Sharpe ratio: {metrics.sharpe_ratio:.3f}")
        
        if metrics.portfolio_value < 95000:  # 5% loss alert
            alerts.append(f"🚨 Significant portfolio loss: ${metrics.portfolio_value:,.2f}")
        
        for alert in alerts:
            self.logger.warning(f"TRADING ALERT: {alert}")
    
    def _check_sentiment_alerts(self, metrics: SentimentMetrics) -> None:
        """Check for sentiment pipeline alerts"""
        alerts = []
        
        if metrics.sentiment_accuracy < self.alert_thresholds['sentiment_accuracy_low']:
            alerts.append(f"🚨 Low sentiment accuracy: {metrics.sentiment_accuracy:.1%}")
        
        if metrics.processing_latency > 300:  # 5 minutes
            alerts.append(f"⚠️ High processing latency: {metrics.processing_latency:.1f}s")
        
        # Check data source uptime
        for source, uptime in metrics.data_source_uptime.items():
            if uptime < self.alert_thresholds['system_downtime']:
                alerts.append(f"🚨 {source} downtime: {uptime:.1%}")
        
        for alert in alerts:
            self.logger.warning(f"SENTIMENT ALERT: {alert}")
    
    def _get_active_alerts(self) -> List[str]:
        """Get list of currently active alerts"""
        alerts = []
        
        if self.trading_history:
            latest_trading = self.trading_history[-1]
            if latest_trading.max_drawdown > self.alert_thresholds['drawdown_warning']:
                alerts.append(f"High drawdown: {latest_trading.max_drawdown:.2%}")
            if latest_trading.sharpe_ratio < self.alert_thresholds['sharpe_degradation']:
                alerts.append(f"Low Sharpe ratio: {latest_trading.sharpe_ratio:.3f}")
        
        if self.sentiment_history:
            latest_sentiment = self.sentiment_history[-1]
            if latest_sentiment.sentiment_accuracy < self.alert_thresholds['sentiment_accuracy_low']:
                alerts.append(f"Low sentiment accuracy: {latest_sentiment.sentiment_accuracy:.1%}")
        
        return alerts
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if self.trading_history:
            latest_trading = self.trading_history[-1]
            
            if latest_trading.sharpe_ratio < self.targets['sharpe_baseline']:
                recommendations.append("Priority: Focus on sentiment integration to boost Sharpe ratio")
            
            if latest_trading.win_rate < 0.5:
                recommendations.append("Consider tightening signal confidence thresholds")
            
            if latest_trading.trades_executed == 0:
                recommendations.append("Monitor for signal generation issues or low volatility period")
        
        if self.sentiment_history:
            latest_sentiment = self.sentiment_history[-1]
            
            if latest_sentiment.sentiment_accuracy < 0.85:
                recommendations.append("Optimize sentiment model weights or add data sources")
            
            if latest_sentiment.processing_latency > 60:
                recommendations.append("Consider cloud infrastructure upgrade for faster processing")
        
        if not self.combined_history:
            recommendations.append("Priority: Begin sentiment-trading integration development")
        
        return recommendations
    
    def create_performance_visualization(self, days_back: int = 7) -> None:
        """Create visualization dashboard"""
        
        if len(self.trading_history) < 2:
            print("Insufficient data for visualization")
            return
        
        # Prepare data
        trading_df = pd.DataFrame([asdict(m) for m in self.trading_history[-days_back:]])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Quantum Trading Revolution - Dual Track Performance Dashboard', fontsize=16)
        
        # Plot 1: Portfolio Value
        axes[0,0].plot(trading_df['timestamp'], trading_df['portfolio_value'], 'b-', linewidth=2)
        axes[0,0].axhline(y=100000, color='g', linestyle='--', alpha=0.5, label='Baseline')
        axes[0,0].set_title('Track 1: Portfolio Value')
        axes[0,0].set_ylabel('Value ($)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Sharpe Ratio Progress
        axes[0,1].plot(trading_df['timestamp'], trading_df['sharpe_ratio'], 'r-', linewidth=2, label='Current')
        axes[0,1].axhline(y=self.targets['sharpe_baseline'], color='orange', linestyle='--', label='Baseline (0.801)')
        axes[0,1].axhline(y=self.targets['sharpe_target'], color='g', linestyle='--', label='Target (1.5)')
        axes[0,1].set_title('Sharpe Ratio Progress')
        axes[0,1].set_ylabel('Sharpe Ratio')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        axes[1,0].fill_between(trading_df['timestamp'], 0, -trading_df['max_drawdown'], 
                              alpha=0.7, color='red', label='Drawdown')
        axes[1,0].axhline(y=-self.targets['max_drawdown_limit'], color='r', 
                         linestyle='--', label='Max Limit (-15%)')
        axes[1,0].set_title('Maximum Drawdown')
        axes[1,0].set_ylabel('Drawdown (%)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Success Metrics
        if self.sentiment_history:
            sentiment_df = pd.DataFrame([asdict(m) for m in self.sentiment_history[-days_back:]])
            
            # Dual y-axis for sentiment metrics
            ax4 = axes[1,1]
            ax4_twin = ax4.twinx()
            
            ax4.plot(sentiment_df['timestamp'], sentiment_df['sentiment_accuracy'], 'g-', linewidth=2, label='Accuracy')
            ax4.axhline(y=self.targets['sentiment_accuracy_target'], color='g', linestyle='--', alpha=0.5)
            ax4.set_ylabel('Sentiment Accuracy', color='g')
            ax4.set_title('Track 2: Sentiment Pipeline')
            
            ax4_twin.plot(sentiment_df['timestamp'], sentiment_df['total_sentiment_signals'], 'b-', linewidth=2, label='Signals')
            ax4_twin.set_ylabel('Total Signals', color='b')
            
            ax4.grid(True, alpha=0.3)
        else:
            axes[1,1].text(0.5, 0.5, 'Track 2: Sentiment Data\nComing Soon', 
                          ha='center', va='center', transform=axes[1,1].transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.savefig('dual_track_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Dashboard visualization saved as 'dual_track_dashboard.png'")


# Example usage and testing
def simulate_sample_data():
    """Generate sample data for testing dashboard"""
    
    dashboard = DualTrackMonitoringDashboard()
    
    # Simulate 7 days of data
    for day in range(7):
        timestamp = datetime.now() - timedelta(days=6-day)
        
        # Sample trading metrics (showing improvement over time)
        portfolio_value = 100000 + day * 1000 + np.random.normal(0, 500)
        daily_return = np.random.normal(0.001, 0.02)
        cumulative_return = portfolio_value / 100000 - 1
        sharpe_ratio = 0.801 + day * 0.05 + np.random.normal(0, 0.05)
        max_drawdown = max(0.02, 0.15 - day * 0.01 + np.random.normal(0, 0.01))
        
        trading_metrics = TradingMetrics(
            timestamp=timestamp,
            portfolio_value=portfolio_value,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            trades_executed=day,
            win_rate=0.6 + np.random.normal(0, 0.1),
            current_position='LONG' if day % 3 == 0 else 'CASH',
            last_signal='BUY' if day % 3 == 0 else 'HOLD',
            signal_confidence=np.random.uniform(0.5, 0.9)
        )
        
        # Sample sentiment metrics (showing progress toward 87% accuracy)
        sentiment_accuracy = 0.75 + day * 0.02 + np.random.normal(0, 0.02)
        
        sentiment_metrics = SentimentMetrics(
            timestamp=timestamp,
            total_sentiment_signals=day * 10 + np.random.randint(5, 15),
            sentiment_accuracy=min(0.9, sentiment_accuracy),
            data_source_uptime={'Twitter': 0.98, 'Reddit': 0.96, 'FinBERT': 0.99},
            processing_latency=np.random.uniform(30, 120),
            signal_quality_score=np.random.uniform(0.6, 0.9),
            api_usage={'Twitter': day * 100, 'Reddit': day * 50},
            model_confidence=np.random.uniform(0.7, 0.9),
            active_assets=['SPY', 'AAPL', 'TSLA', 'BTC', 'ETH']
        )
        
        # Sample combined metrics (showing integration progress)
        if day >= 3:  # Start integration metrics from day 3
            combined_metrics = CombinedMetrics(
                timestamp=timestamp,
                sentiment_enhanced_signals=day - 2,
                sharpe_improvement=max(0, sharpe_ratio - 0.801),
                signal_quality_enhancement=0.05 + day * 0.02,
                risk_adjustment_impact=0.02 + day * 0.005,
                integration_uptime=0.95 + np.random.normal(0, 0.02)
            )
            dashboard.update_combined_metrics(combined_metrics)
        
        dashboard.update_trading_metrics(trading_metrics)
        dashboard.update_sentiment_metrics(sentiment_metrics)
    
    # Generate reports
    daily_report = dashboard.generate_daily_report()
    
    print("📊 DUAL TRACK MONITORING DASHBOARD")
    print("=" * 50)
    print(f"Date: {daily_report['date']}")
    print(f"Overall Status: {daily_report['summary']['overall_status']}")
    print(f"Week 2 Progress: {daily_report['summary']['week_2_progress']}")
    print()
    
    print("🏦 TRACK 1 - TRADING PERFORMANCE")
    for key, value in daily_report['track_1_trading'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    print()
    
    print("🤖 TRACK 2 - SENTIMENT PIPELINE") 
    for key, value in daily_report['track_2_sentiment'].items():
        if key != 'data_sources':
            print(f"  {key.replace('_', ' ').title()}: {value}")
    print()
    
    print("🔗 INTEGRATION STATUS")
    for key, value in daily_report['integration'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    print()
    
    print("📈 PROGRESS TRACKING")
    for key, value in daily_report['progress_tracking'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    print()
    
    if daily_report['alerts']:
        print("🚨 ACTIVE ALERTS")
        for alert in daily_report['alerts']:
            print(f"  - {alert}")
        print()
    
    print("💡 RECOMMENDATIONS")
    for rec in daily_report['recommendations']:
        print(f"  - {rec}")
    
    # Create visualization
    dashboard.create_performance_visualization()
    
    return dashboard


if __name__ == "__main__":
    print("🚀 QUANTUM TRADING REVOLUTION - DUAL TRACK MONITORING")
    print("Simulating sample data for dashboard testing...")
    
    dashboard = simulate_sample_data()
    
    print("\n✅ Dashboard system ready for production deployment")
    print("Ready to monitor both QuantConnect trading and sentiment pipeline performance")
