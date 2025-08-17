"""
Local Backtesting Tool - Moving Average Strategy Validation
Quantum Trading Revolution - Phase 1 Mission 1A

This tool provides local backtesting capabilities before QuantConnect deployment:
- Validates strategy logic with real historical data
- Calculates performance metrics matching QuantConnect standards  
- Generates comprehensive performance reports
- Enables rapid strategy iteration and debugging

Performance Targets (Mission 1 Objectives):
- Sharpe Ratio > 1.5 (target 2.0+ excellent)
- Max Drawdown < 15% (risk control)
- Annual Return > 15% (beat market baseline)
- Win Rate 45-55% (realistic trend following)

Usage:
    python backtest_ma_strategy.py --symbol SPY --start-date 2023-07-01 --end-date 2024-01-01
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our SOLID strategy
from trading.strategies.moving_average_crossover import MovingAverageCrossoverStrategy
from core.interfaces import MarketData, SignalType


class LocalBacktester:
    """
    Local backtesting engine with QuantConnect-compatible metrics
    
    Features:
    - Real historical data via yfinance
    - Transaction cost modeling
    - Slippage simulation
    - Risk management integration
    - Comprehensive performance analytics
    - QuantConnect metric compatibility
    """
    
    def __init__(
        self, 
        initial_capital: float = 100000,
        transaction_cost_percent: float = 0.001,  # 0.1% transaction cost
        slippage_percent: float = 0.0005,        # 0.05% slippage
        max_position_percent: float = 0.95       # 95% max position
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_percent = transaction_cost_percent
        self.slippage_percent = slippage_percent
        self.max_position_percent = max_position_percent
        
        # Portfolio state
        self.cash = initial_capital
        self.shares = 0
        self.portfolio_values = []
        self.trades = []
        self.equity_curve = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0
        self.peak_value = initial_capital
        
        print(f"✅ Backtester initialized with ${initial_capital:,.2f} capital")
        print(f"   Transaction costs: {transaction_cost_percent*100:.3f}%")
        print(f"   Slippage: {slippage_percent*100:.3f}%")
    
    def run_backtest(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        fast_period: int = 10,
        slow_period: int = 30
    ) -> Dict[str, Any]:
        """
        Run complete backtesting simulation
        
        Args:
            symbol: Asset symbol to trade
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)  
            fast_period: Fast MA period
            slow_period: Slow MA period
            
        Returns:
            Dict containing complete performance results
        """
        print(f"🚀 Starting backtest for {symbol}")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Strategy: MA Crossover ({fast_period}/{slow_period})")
        print("="*60)
        
        # Download historical data
        data = self._download_data(symbol, start_date, end_date)
        if data is None or len(data) == 0:
            raise ValueError(f"Failed to download data for {symbol}")
        
        print(f"✅ Downloaded {len(data)} days of historical data")
        
        # Initialize strategy
        strategy = MovingAverageCrossoverStrategy(
            fast_period=fast_period,
            slow_period=slow_period,
            min_separation_threshold=0.01,  # 1% minimum separation
            confidence_multiplier=1.0
        )
        
        # Run simulation
        position = None  # None, 'LONG', 'SHORT'
        entry_price = None
        entry_date = None
        
        for i, (date, row) in enumerate(data.iterrows()):
            # Create market data object
            market_data = MarketData(
                symbol=symbol,
                timestamp=date,
                open_price=float(row['Open']),
                high_price=float(row['High']),
                low_price=float(row['Low']),
                close_price=float(row['Close']),
                volume=int(row['Volume'])
            )
            
            # Generate trading signal
            signal = strategy.generate_signal(market_data)
            
            # Process signal
            current_price = market_data.close_price
            
            # Entry logic
            if signal.signal_type == SignalType.BUY and position is None:
                if signal.confidence >= 0.3:  # Minimum confidence threshold
                    shares_to_buy = self._calculate_position_size(current_price)
                    if shares_to_buy > 0:
                        execution_price = self._apply_slippage(current_price, 'BUY')
                        total_cost = shares_to_buy * execution_price
                        transaction_cost = total_cost * self.transaction_cost_percent
                        
                        if self.cash >= (total_cost + transaction_cost):
                            self.shares = shares_to_buy
                            self.cash -= (total_cost + transaction_cost)
                            position = 'LONG'
                            entry_price = execution_price
                            entry_date = date
                            
                            # Log trade
                            self.trades.append({
                                'type': 'BUY',
                                'date': date,
                                'price': execution_price,
                                'shares': shares_to_buy,
                                'cost': total_cost + transaction_cost,
                                'confidence': signal.confidence,
                                'signal_metadata': signal.metadata
                            })
                            
                            print(f"📈 BUY: {shares_to_buy} shares at ${execution_price:.2f} on {date.strftime('%Y-%m-%d')} (Confidence: {signal.confidence:.2f})")
            
            # Exit logic
            elif signal.signal_type == SignalType.SELL and position == 'LONG':
                if signal.confidence >= 0.3:
                    execution_price = self._apply_slippage(current_price, 'SELL')
                    total_proceeds = self.shares * execution_price
                    transaction_cost = total_proceeds * self.transaction_cost_percent
                    net_proceeds = total_proceeds - transaction_cost
                    
                    # Calculate trade PnL
                    trade_pnl = net_proceeds - (self.shares * entry_price)
                    trade_return = trade_pnl / (self.shares * entry_price)
                    hold_days = (date - entry_date).days
                    
                    self.cash += net_proceeds
                    
                    # Update trade statistics
                    self.total_trades += 1
                    if trade_pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                    
                    # Log trade
                    self.trades.append({
                        'type': 'SELL',
                        'date': date,
                        'price': execution_price,
                        'shares': -self.shares,
                        'proceeds': net_proceeds,
                        'pnl': trade_pnl,
                        'return': trade_return,
                        'hold_days': hold_days,
                        'confidence': signal.confidence,
                        'signal_metadata': signal.metadata
                    })
                    
                    print(f"📉 SELL: {self.shares} shares at ${execution_price:.2f} on {date.strftime('%Y-%m-%d')} | PnL: ${trade_pnl:.2f} ({trade_return*100:.1f}%) | Hold: {hold_days} days")
                    
                    # Reset position
                    self.shares = 0
                    position = None
                    entry_price = None
                    entry_date = None
            
            # Calculate current portfolio value
            if self.shares > 0:
                portfolio_value = self.cash + (self.shares * current_price)
            else:
                portfolio_value = self.cash
            
            self.portfolio_values.append(portfolio_value)
            self.equity_curve.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'position_value': self.shares * current_price if self.shares > 0 else 0,
                'shares': self.shares,
                'price': current_price
            })
            
            # Update drawdown tracking
            if portfolio_value > self.peak_value:
                self.peak_value = portfolio_value
            
            current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
        
        # Close any open position at end
        if self.shares > 0:
            final_price = data.iloc[-1]['Close']
            final_value = self.cash + (self.shares * final_price)
            print(f"🔄 Final position closed at ${final_price:.2f}")
        else:
            final_value = self.cash
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(data, symbol)
        
        print("="*60)
        print("🎯 BACKTEST COMPLETED")
        print("="*60)
        
        return performance
    
    def _download_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download historical data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            return data
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            return None
    
    def _calculate_position_size(self, price: float) -> int:
        """Calculate position size based on available cash and max position limit"""
        max_investment = self.cash * self.max_position_percent
        shares = int(max_investment / price)
        return max(0, shares)
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply realistic slippage to execution price"""
        if side == 'BUY':
            return price * (1 + self.slippage_percent)
        else:  # SELL
            return price * (1 - self.slippage_percent)
    
    def _calculate_performance_metrics(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        # Basic returns
        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Annualized metrics
        days = len(data)
        years = days / 365.25
        annual_return = (final_value / self.initial_capital) ** (1/years) - 1
        
        # Returns series for Sharpe calculation
        portfolio_values_series = pd.Series(self.portfolio_values, index=data.index)
        daily_returns = portfolio_values_series.pct_change().dropna()
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = daily_returns - (risk_free_rate / 252)  # Daily risk-free rate
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Benchmark comparison (buy and hold)
        benchmark_return = (data.iloc[-1]['Close'] - data.iloc[0]['Open']) / data.iloc[0]['Open']
        benchmark_annual = (1 + benchmark_return) ** (1/years) - 1
        
        # Win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Trade statistics
        if len(self.trades) > 0:
            trade_returns = [t.get('return', 0) for t in self.trades if 'return' in t]
            avg_win = np.mean([r for r in trade_returns if r > 0]) if trade_returns else 0
            avg_loss = np.mean([r for r in trade_returns if r < 0]) if trade_returns else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            avg_win = avg_loss = profit_factor = 0
        
        # Performance summary
        performance = {
            # Core metrics
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            
            # Trading metrics
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            
            # Benchmark comparison
            'benchmark_return': benchmark_return,
            'benchmark_annual': benchmark_annual,
            'alpha': annual_return - benchmark_annual,
            
            # Additional data
            'days_traded': days,
            'years': years,
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'daily_returns': daily_returns.tolist()
        }
        
        # Print performance summary
        self._print_performance_summary(performance, symbol)
        
        return performance
    
    def _print_performance_summary(self, perf: Dict[str, Any], symbol: str) -> None:
        """Print formatted performance summary"""
        print(f"📊 PERFORMANCE SUMMARY - {symbol}")
        print("-" * 40)
        print(f"Initial Capital:      ${perf['initial_capital']:,.2f}")
        print(f"Final Value:          ${perf['final_value']:,.2f}")
        print(f"Total Return:         {perf['total_return']*100:.2f}%")
        print(f"Annual Return:        {perf['annual_return']*100:.2f}%")
        print(f"Benchmark Return:     {perf['benchmark_annual']*100:.2f}%")
        print(f"Alpha:                {perf['alpha']*100:.2f}%")
        print()
        print("📈 RISK METRICS")
        print("-" * 40)
        print(f"Sharpe Ratio:         {perf['sharpe_ratio']:.3f}")
        print(f"Max Drawdown:         {perf['max_drawdown']*100:.2f}%")
        print()
        print("🎯 TRADING METRICS")
        print("-" * 40)
        print(f"Total Trades:         {perf['total_trades']}")
        print(f"Winning Trades:       {perf['winning_trades']}")
        print(f"Losing Trades:        {perf['losing_trades']}")
        print(f"Win Rate:             {perf['win_rate']*100:.1f}%")
        if perf['avg_win'] > 0:
            print(f"Average Win:          {perf['avg_win']*100:.2f}%")
        if perf['avg_loss'] < 0:
            print(f"Average Loss:         {perf['avg_loss']*100:.2f}%")
        if perf['profit_factor'] > 0:
            print(f"Profit Factor:        {perf['profit_factor']:.2f}")
        print()
        
        # Performance assessment
        print("🏆 PERFORMANCE ASSESSMENT")
        print("-" * 40)
        
        # Sharpe ratio assessment
        if perf['sharpe_ratio'] >= 2.0:
            sharpe_grade = "🟢 EXCELLENT"
        elif perf['sharpe_ratio'] >= 1.5:
            sharpe_grade = "🟡 GOOD"
        elif perf['sharpe_ratio'] >= 1.0:
            sharpe_grade = "🟠 ACCEPTABLE"
        else:
            sharpe_grade = "🔴 POOR"
        print(f"Sharpe Grade:         {sharpe_grade}")
        
        # Drawdown assessment
        if perf['max_drawdown'] <= 0.15:
            dd_grade = "🟢 EXCELLENT"
        elif perf['max_drawdown'] <= 0.20:
            dd_grade = "🟡 GOOD"
        elif perf['max_drawdown'] <= 0.30:
            dd_grade = "🟠 ACCEPTABLE"
        else:
            dd_grade = "🔴 POOR"
        print(f"Drawdown Grade:       {dd_grade}")
        
        # Alpha assessment
        if perf['alpha'] >= 0.05:
            alpha_grade = "🟢 OUTPERFORMING"
        elif perf['alpha'] >= 0:
            alpha_grade = "🟡 MATCHING BENCHMARK"
        else:
            alpha_grade = "🔴 UNDERPERFORMING"
        print(f"Alpha Grade:          {alpha_grade}")


def main():
    """Main execution function with CLI interface"""
    parser = argparse.ArgumentParser(description='Backtest Moving Average Strategy')
    parser.add_argument('--symbol', default='SPY', help='Symbol to trade (default: SPY)')
    parser.add_argument('--start-date', default='2023-07-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--fast-period', type=int, default=10, help='Fast MA period (default: 10)')
    parser.add_argument('--slow-period', type=int, default=30, help='Slow MA period (default: 30)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    
    args = parser.parse_args()
    
    print("🚀 QUANTUM TRADING REVOLUTION - BACKTESTING ENGINE")
    print("="*60)
    
    # Initialize backtester
    backtester = LocalBacktester(initial_capital=args.capital)
    
    # Run backtest
    try:
        results = backtester.run_backtest(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            fast_period=args.fast_period,
            slow_period=args.slow_period
        )
        
        print("✅ Backtest completed successfully!")
        print("\n🎯 READY FOR QUANTCONNECT DEPLOYMENT")
        print("Next steps:")
        print("1. Deploy algorithm to QuantConnect platform")
        print("2. Run full backtesting with institutional data")
        print("3. Transition to paper trading")
        print("4. Scale to multiple assets")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
