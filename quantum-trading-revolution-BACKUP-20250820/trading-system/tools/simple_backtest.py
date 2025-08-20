"""
Simple Backtesting Tool - Quick Validation
Quantum Trading Revolution - Mission 1A Validation

Simplified standalone backtesting to validate MA crossover logic without complex imports.
Focus on rapid validation before QuantConnect deployment.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SimpleMAStrategy:
    """Simplified MA Crossover Strategy for rapid testing"""
    
    def __init__(self, fast_period=10, slow_period=30):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.price_history = []
        self.signals_generated = 0
    
    def add_price(self, price):
        """Add new price to history"""
        self.price_history.append(price)
        if len(self.price_history) > self.slow_period + 1:
            self.price_history = self.price_history[-(self.slow_period + 1):]
    
    def get_signal(self):
        """Generate trading signal"""
        if len(self.price_history) < self.slow_period:
            return 'HOLD', 0.0, {}
        
        # Current MAs
        fast_ma = sum(self.price_history[-self.fast_period:]) / self.fast_period
        slow_ma = sum(self.price_history[-self.slow_period:]) / self.slow_period
        
        # Previous MAs (for crossover detection)
        if len(self.price_history) >= self.slow_period + 1:
            prev_fast = sum(self.price_history[-(self.fast_period+1):-1]) / self.fast_period
            prev_slow = sum(self.price_history[-(self.slow_period+1):-1]) / self.slow_period
            
            # Calculate separation
            separation = abs(fast_ma - slow_ma) / slow_ma if slow_ma > 0 else 0
            
            # Bullish crossover
            if prev_fast <= prev_slow and fast_ma > slow_ma and separation >= 0.01:
                self.signals_generated += 1
                return 'BUY', min(separation * 10, 1.0), {
                    'fast_ma': fast_ma, 'slow_ma': slow_ma, 'separation': separation
                }
            
            # Bearish crossover
            elif prev_fast >= prev_slow and fast_ma < slow_ma and separation >= 0.01:
                self.signals_generated += 1
                return 'SELL', min(separation * 10, 1.0), {
                    'fast_ma': fast_ma, 'slow_ma': slow_ma, 'separation': separation
                }
        
        return 'HOLD', 0.5, {'fast_ma': fast_ma, 'slow_ma': slow_ma}


def download_data(symbol, start_date, end_date):
    """Download data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None


def run_backtest(symbol='SPY', start_date='2023-07-01', end_date='2024-01-01'):
    """Run simplified backtesting"""
    print("🚀 QUANTUM TRADING REVOLUTION - SIMPLE BACKTESTING")
    print("="*60)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Strategy: Moving Average Crossover (10/30)")
    print("="*60)
    
    # Download data
    data = download_data(symbol, start_date, end_date)
    if data is None or len(data) == 0:
        print("❌ Failed to download data")
        return
    
    print(f"✅ Downloaded {len(data)} days of data")
    
    # Initialize
    strategy = SimpleMAStrategy(fast_period=10, slow_period=30)
    
    # Portfolio tracking
    initial_capital = 100000
    cash = initial_capital
    shares = 0
    portfolio_values = []
    trades = []
    
    position = None
    entry_price = None
    entry_date = None
    
    total_trades = 0
    winning_trades = 0
    
    transaction_cost = 0.001  # 0.1%
    
    # Run simulation
    for i, (date, row) in enumerate(data.iterrows()):
        current_price = float(row['Close'])
        strategy.add_price(current_price)
        
        signal, confidence, metadata = strategy.get_signal()
        
        # Process signals
        if signal == 'BUY' and position is None and confidence >= 0.3:
            # Calculate position size
            available_cash = cash * 0.95  # Use 95% of available cash
            shares_to_buy = int(available_cash / current_price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                transaction_fee = cost * transaction_cost
                total_cost = cost + transaction_fee
                
                if cash >= total_cost:
                    shares = shares_to_buy
                    cash -= total_cost
                    position = 'LONG'
                    entry_price = current_price
                    entry_date = date
                    
                    trades.append({
                        'type': 'BUY',
                        'date': date,
                        'price': current_price,
                        'shares': shares_to_buy,
                        'confidence': confidence
                    })
                    
                    print(f"📈 BUY: {shares_to_buy} shares at ${current_price:.2f} on {date.strftime('%Y-%m-%d')} (Confidence: {confidence:.2f})")
        
        elif signal == 'SELL' and position == 'LONG' and confidence >= 0.3:
            # Sell position
            proceeds = shares * current_price
            transaction_fee = proceeds * transaction_cost
            net_proceeds = proceeds - transaction_fee
            
            # Calculate trade performance
            trade_pnl = net_proceeds - (shares * entry_price)
            trade_return = trade_pnl / (shares * entry_price)
            hold_days = (date - entry_date).days
            
            cash += net_proceeds
            
            trades.append({
                'type': 'SELL',
                'date': date,
                'price': current_price,
                'shares': -shares,
                'pnl': trade_pnl,
                'return': trade_return,
                'hold_days': hold_days,
                'confidence': confidence
            })
            
            total_trades += 1
            if trade_pnl > 0:
                winning_trades += 1
            
            print(f"📉 SELL: {shares} shares at ${current_price:.2f} on {date.strftime('%Y-%m-%d')} | PnL: ${trade_pnl:.2f} ({trade_return*100:.1f}%) | Hold: {hold_days} days")
            
            shares = 0
            position = None
            entry_price = None
            entry_date = None
        
        # Calculate portfolio value
        current_portfolio_value = cash + (shares * current_price if shares > 0 else 0)
        portfolio_values.append(current_portfolio_value)
    
    # Final position handling
    final_price = data.iloc[-1]['Close']
    if shares > 0:
        final_portfolio_value = cash + (shares * final_price)
        print(f"🔄 Final position: {shares} shares at ${final_price:.2f}")
    else:
        final_portfolio_value = cash
    
    # Performance calculation
    total_return = (final_portfolio_value - initial_capital) / initial_capital
    
    # Calculate daily returns for Sharpe
    portfolio_series = pd.Series(portfolio_values, index=data.index)
    daily_returns = portfolio_series.pct_change().dropna()
    
    # Sharpe ratio (assuming 2% risk-free rate)
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        risk_free_daily = 0.02 / 252
        excess_returns = daily_returns - risk_free_daily
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Max drawdown
    peak = portfolio_series.expanding().max()
    drawdown = (portfolio_series - peak) / peak
    max_drawdown = drawdown.min()
    
    # Benchmark (buy and hold)
    benchmark_return = (data.iloc[-1]['Close'] - data.iloc[0]['Close']) / data.iloc[0]['Close']
    
    # Print results
    print("="*60)
    print("🎯 BACKTEST RESULTS")
    print("="*60)
    print(f"📊 PERFORMANCE SUMMARY")
    print("-" * 40)
    print(f"Initial Capital:      ${initial_capital:,.2f}")
    print(f"Final Value:          ${final_portfolio_value:,.2f}")
    print(f"Total Return:         {total_return*100:.2f}%")
    print(f"Benchmark (B&H):      {benchmark_return*100:.2f}%")
    print(f"Alpha:                {(total_return - benchmark_return)*100:.2f}%")
    print()
    print(f"📈 RISK METRICS")
    print("-" * 40)
    print(f"Sharpe Ratio:         {sharpe_ratio:.3f}")
    print(f"Max Drawdown:         {abs(max_drawdown)*100:.2f}%")
    print()
    print(f"🎯 TRADING METRICS")
    print("-" * 40)
    print(f"Signals Generated:    {strategy.signals_generated}")
    print(f"Total Trades:         {total_trades}")
    print(f"Winning Trades:       {winning_trades}")
    print(f"Win Rate:             {(winning_trades/total_trades)*100:.1f}%" if total_trades > 0 else "0.0%")
    
    # Performance assessment
    print()
    print("🏆 PERFORMANCE ASSESSMENT")
    print("-" * 40)
    
    # Sharpe assessment
    if sharpe_ratio >= 2.0:
        sharpe_grade = "🟢 EXCELLENT"
    elif sharpe_ratio >= 1.5:
        sharpe_grade = "🟡 GOOD"
    elif sharpe_ratio >= 1.0:
        sharpe_grade = "🟠 ACCEPTABLE"
    else:
        sharpe_grade = "🔴 POOR"
    print(f"Sharpe Grade:         {sharpe_grade}")
    
    # Drawdown assessment
    dd_pct = abs(max_drawdown)
    if dd_pct <= 0.15:
        dd_grade = "🟢 EXCELLENT"
    elif dd_pct <= 0.20:
        dd_grade = "🟡 GOOD"  
    elif dd_pct <= 0.30:
        dd_grade = "🟠 ACCEPTABLE"
    else:
        dd_grade = "🔴 POOR"
    print(f"Drawdown Grade:       {dd_grade}")
    
    # Alpha assessment
    alpha = total_return - benchmark_return
    if alpha >= 0.05:
        alpha_grade = "🟢 OUTPERFORMING"
    elif alpha >= 0:
        alpha_grade = "🟡 MATCHING BENCHMARK"
    else:
        alpha_grade = "🔴 UNDERPERFORMING"
    print(f"Alpha Grade:          {alpha_grade}")
    
    print("="*60)
    
    # Mission 1A validation
    print("✅ MISSION 1A VALIDATION")
    print("-" * 40)
    
    success_criteria = 0
    total_criteria = 4
    
    if sharpe_ratio >= 1.5:
        print("✅ Sharpe Ratio ≥ 1.5: PASSED")
        success_criteria += 1
    else:
        print("❌ Sharpe Ratio ≥ 1.5: FAILED")
    
    if abs(max_drawdown) <= 0.15:
        print("✅ Max Drawdown ≤ 15%: PASSED")
        success_criteria += 1
    else:
        print("❌ Max Drawdown ≤ 15%: FAILED")
    
    if total_return >= 0.15:  # 15% annual return target
        print("✅ Annual Return ≥ 15%: PASSED")
        success_criteria += 1
    else:
        print("❌ Annual Return ≥ 15%: FAILED")
    
    if alpha >= 0:
        print("✅ Beats Benchmark: PASSED")
        success_criteria += 1
    else:
        print("❌ Beats Benchmark: FAILED")
    
    success_rate = success_criteria / total_criteria
    print(f"\n📊 Overall Success: {success_criteria}/{total_criteria} ({success_rate*100:.1f}%)")
    
    if success_rate >= 0.75:
        print("🚀 READY FOR QUANTCONNECT DEPLOYMENT")
    elif success_rate >= 0.5:
        print("⚠️ CONDITIONAL APPROVAL - Consider parameter tuning")
    else:
        print("❌ REQUIRES IMPROVEMENT before deployment")
    
    return {
        'success_rate': success_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': abs(max_drawdown),
        'total_return': total_return,
        'alpha': alpha,
        'trades': total_trades,
        'win_rate': winning_trades/total_trades if total_trades > 0 else 0
    }


if __name__ == "__main__":
    import sys
    
    # Parse simple command line arguments
    symbol = 'SPY'
    start_date = '2023-07-01'
    end_date = '2024-01-01'
    
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg == '--symbol' and i+1 < len(sys.argv):
                symbol = sys.argv[i+1]
            elif arg == '--start-date' and i+1 < len(sys.argv):
                start_date = sys.argv[i+1]
            elif arg == '--end-date' and i+1 < len(sys.argv):
                end_date = sys.argv[i+1]
    
    results = run_backtest(symbol, start_date, end_date)
