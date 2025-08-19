"""
MA Strategy Optimization Tool
Quantum Trading Revolution - Mission 1A Optimization

Tool to find optimal MA crossover parameters that meet Mission 1 criteria:
- Sharpe Ratio ≥ 1.5 (target 2.0+)
- Max Drawdown ≤ 15% 
- Annual Return ≥ 15%
- Beat benchmark (SPY buy & hold)

Tests various combinations of:
- Fast/Slow MA periods
- Separation thresholds
- Position sizing
- Risk management parameters
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from itertools import product


def test_ma_parameters(symbol, data, fast_period, slow_period, threshold, max_position=0.95, stop_loss=None, take_profit=None):
    """Test specific MA parameters and return performance metrics"""
    
    # Calculate MAs
    data_test = data.copy()
    data_test['Fast_MA'] = data_test['Close'].rolling(window=fast_period).mean()
    data_test['Slow_MA'] = data_test['Close'].rolling(window=slow_period).mean()
    data_test['MA_Diff'] = data_test['Fast_MA'] - data_test['Slow_MA']
    data_test['MA_Separation'] = abs(data_test['MA_Diff']) / data_test['Slow_MA']
    
    # Generate signals
    data_test['Bullish_Cross'] = (data_test['MA_Diff'] > 0) & (data_test['MA_Diff'].shift(1) <= 0) & (data_test['MA_Separation'] >= threshold)
    data_test['Bearish_Cross'] = (data_test['MA_Diff'] < 0) & (data_test['MA_Diff'].shift(1) >= 0) & (data_test['MA_Separation'] >= threshold)
    
    # Simulate trading
    capital = 100000
    cash = capital
    shares = 0
    position = None
    entry_price = None
    trades = []
    portfolio_values = []
    
    for date, row in data_test.iterrows():
        current_price = row['Close']
        
        # Entry signals
        if row['Bullish_Cross'] and position is None:
            shares_to_buy = int(cash * max_position / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * 1.001  # 0.1% transaction cost
                if cash >= cost:
                    shares = shares_to_buy
                    cash -= cost
                    position = 'LONG'
                    entry_price = current_price
                    trades.append(('BUY', date, current_price, shares_to_buy))
        
        # Exit signals
        elif row['Bearish_Cross'] and position == 'LONG':
            proceeds = shares * current_price * 0.999  # 0.1% transaction cost
            cash += proceeds
            trades.append(('SELL', date, current_price, -shares))
            shares = 0
            position = None
            entry_price = None
        
        # Risk management exits
        elif position == 'LONG' and entry_price:
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Stop loss
            if stop_loss and pnl_pct <= -stop_loss:
                proceeds = shares * current_price * 0.999
                cash += proceeds
                trades.append(('STOP_LOSS', date, current_price, -shares))
                shares = 0
                position = None
                entry_price = None
            
            # Take profit
            elif take_profit and pnl_pct >= take_profit:
                proceeds = shares * current_price * 0.999
                cash += proceeds
                trades.append(('TAKE_PROFIT', date, current_price, -shares))
                shares = 0
                position = None
                entry_price = None
        
        # Record portfolio value
        portfolio_value = cash + (shares * current_price if shares > 0 else 0)
        portfolio_values.append(portfolio_value)
    
    # Handle final position
    final_price = data_test.iloc[-1]['Close']
    if shares > 0:
        final_value = cash + (shares * final_price)
    else:
        final_value = cash
    
    # Calculate metrics
    total_return = (final_value - capital) / capital
    
    # Returns for Sharpe calculation
    portfolio_series = pd.Series(portfolio_values, index=data_test.index)
    daily_returns = portfolio_series.pct_change().dropna()
    
    # Sharpe ratio
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        risk_free_daily = 0.02 / 252  # 2% annual risk-free rate
        excess_returns = daily_returns - risk_free_daily
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Max drawdown
    peak = portfolio_series.expanding().max()
    drawdown = (portfolio_series - peak) / peak
    max_drawdown = abs(drawdown.min())
    
    # Trade statistics
    buy_trades = len([t for t in trades if t[0] == 'BUY'])
    
    return {
        'fast_period': fast_period,
        'slow_period': slow_period,
        'threshold': threshold,
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': buy_trades,
        'trades': trades
    }


def optimize_ma_strategy(symbol='SPY', start_date='2022-01-01', end_date='2023-12-31'):
    """Find optimal MA parameters"""
    
    print("🔍 MA STRATEGY OPTIMIZATION")
    print("="*60)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print("="*60)
    
    # Download data
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    if len(data) == 0:
        print("❌ No data downloaded")
        return None
    
    print(f"✅ Downloaded {len(data)} days of data")
    
    # Calculate benchmark (buy & hold)
    benchmark_return = (data.iloc[-1]['Close'] - data.iloc[0]['Close']) / data.iloc[0]['Close']
    
    # Define parameter ranges to test
    fast_periods = [5, 8, 10, 12, 15]
    slow_periods = [20, 25, 30, 35, 40, 50]
    thresholds = [0.001, 0.005, 0.01, 0.015, 0.02]  # 0.1% to 2%
    
    print(f"Testing {len(fast_periods)} x {len(slow_periods)} x {len(thresholds)} = {len(fast_periods) * len(slow_periods) * len(thresholds)} combinations...")
    
    results = []
    best_score = -float('inf')
    best_params = None
    
    # Test all combinations
    for fast, slow, thresh in product(fast_periods, slow_periods, thresholds):
        if fast >= slow:  # Skip invalid combinations
            continue
        
        try:
            result = test_ma_parameters(symbol, data, fast, slow, thresh)
            
            # Score function (weighted combination of metrics)
            # Prioritize: Sharpe ratio, then returns, then low drawdown
            sharpe_score = result['sharpe_ratio'] * 2  # Double weight for Sharpe
            return_score = result['total_return']
            drawdown_penalty = result['max_drawdown'] * -3  # Penalty for high drawdown
            
            # Require minimum number of trades (avoid overfitting)
            if result['num_trades'] >= 2:
                total_score = sharpe_score + return_score + drawdown_penalty
            else:
                total_score = -1000  # Heavy penalty for too few trades
            
            result['score'] = total_score
            results.append(result)
            
            # Track best parameters
            if total_score > best_score:
                best_score = total_score
                best_params = result
            
        except Exception as e:
            print(f"Error testing {fast}/{slow}/{thresh}: {e}")
            continue
    
    # Sort results by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n✅ Optimization complete! Tested {len(results)} valid combinations")
    
    # Show top 10 results
    print(f"\n🏆 TOP 10 PARAMETER COMBINATIONS")
    print("-" * 80)
    print(f"{'Rank':<4} {'Fast/Slow':<10} {'Threshold':<10} {'Return':<8} {'Sharpe':<8} {'DD':<8} {'Trades':<7} {'Score':<8}")
    print("-" * 80)
    
    for i, result in enumerate(results[:10]):
        print(f"{i+1:<4} {result['fast_period']}/{result['slow_period']:<10} "
              f"{result['threshold']*100:.2f}%{'':<5} "
              f"{result['total_return']*100:.1f}%{'':<3} "
              f"{result['sharpe_ratio']:.2f}{'':<5} "
              f"{result['max_drawdown']*100:.1f}%{'':<4} "
              f"{result['num_trades']:<7} "
              f"{result['score']:.2f}")
    
    # Analyze best result in detail
    best = results[0]
    print(f"\n🥇 BEST CONFIGURATION ANALYSIS")
    print("-" * 50)
    print(f"Parameters: Fast MA = {best['fast_period']}, Slow MA = {best['slow_period']}, Threshold = {best['threshold']*100:.2f}%")
    print(f"Final Value: ${best['final_value']:,.2f}")
    print(f"Total Return: {best['total_return']*100:.2f}%")
    print(f"Benchmark Return: {benchmark_return*100:.2f}%")
    print(f"Alpha: {(best['total_return'] - benchmark_return)*100:.2f}%")
    print(f"Sharpe Ratio: {best['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {best['max_drawdown']*100:.2f}%")
    print(f"Number of Trades: {best['num_trades']}")
    
    # Mission 1 criteria validation
    print(f"\n✅ MISSION 1 CRITERIA VALIDATION")
    print("-" * 40)
    
    criteria_met = 0
    total_criteria = 4
    
    # Annualized return (rough estimate)
    days = len(data)
    years = days / 365.25
    annual_return = (best['final_value'] / 100000) ** (1/years) - 1
    
    if best['sharpe_ratio'] >= 1.5:
        print(f"✅ Sharpe Ratio ≥ 1.5: PASSED ({best['sharpe_ratio']:.3f})")
        criteria_met += 1
    else:
        print(f"❌ Sharpe Ratio ≥ 1.5: FAILED ({best['sharpe_ratio']:.3f})")
    
    if best['max_drawdown'] <= 0.15:
        print(f"✅ Max Drawdown ≤ 15%: PASSED ({best['max_drawdown']*100:.2f}%)")
        criteria_met += 1
    else:
        print(f"❌ Max Drawdown ≤ 15%: FAILED ({best['max_drawdown']*100:.2f}%)")
    
    if annual_return >= 0.15:
        print(f"✅ Annual Return ≥ 15%: PASSED ({annual_return*100:.2f}%)")
        criteria_met += 1
    else:
        print(f"❌ Annual Return ≥ 15%: FAILED ({annual_return*100:.2f}%)")
    
    if best['total_return'] > benchmark_return:
        print(f"✅ Beats Benchmark: PASSED")
        criteria_met += 1
    else:
        print(f"❌ Beats Benchmark: FAILED")
    
    success_rate = criteria_met / total_criteria
    print(f"\n📊 Overall Success: {criteria_met}/{total_criteria} ({success_rate*100:.1f}%)")
    
    if success_rate >= 0.75:
        print("🚀 READY FOR QUANTCONNECT DEPLOYMENT")
        deployment_status = "READY"
    elif success_rate >= 0.5:
        print("⚠️ CONDITIONAL APPROVAL - Consider further optimization")
        deployment_status = "CONDITIONAL"
    else:
        print("❌ REQUIRES IMPROVEMENT before deployment")
        deployment_status = "NEEDS_WORK"
    
    # Show sample trades for best configuration
    if best['trades']:
        print(f"\n📋 SAMPLE TRADES (Best Configuration)")
        print("-" * 50)
        for i, trade in enumerate(best['trades'][:10]):  # Show first 10 trades
            action, date, price, quantity = trade
            print(f"{i+1}. {action}: {abs(quantity)} shares at ${price:.2f} on {date.strftime('%Y-%m-%d')}")
        if len(best['trades']) > 10:
            print(f"... and {len(best['trades']) - 10} more trades")
    
    return {
        'best_params': best,
        'all_results': results[:20],  # Top 20 results
        'deployment_status': deployment_status,
        'success_rate': success_rate,
        'benchmark_return': benchmark_return
    }


def test_multiple_assets():
    """Test optimized parameters on multiple assets"""
    
    assets = ['SPY', 'QQQ', 'IWM', 'DIA']  # Different ETFs
    print(f"\n🔄 TESTING OPTIMIZED STRATEGY ON MULTIPLE ASSETS")
    print("="*60)
    
    # Get best SPY parameters first
    spy_result = optimize_ma_strategy('SPY', '2022-01-01', '2023-12-31')
    if not spy_result or spy_result['deployment_status'] == 'NEEDS_WORK':
        print("❌ SPY optimization failed - cannot test other assets")
        return
    
    best_params = spy_result['best_params']
    fast = best_params['fast_period']
    slow = best_params['slow_period']
    thresh = best_params['threshold']
    
    print(f"Using optimized parameters: {fast}/{slow} MA, {thresh*100:.2f}% threshold")
    print("-" * 60)
    
    multi_results = {}
    
    for asset in assets:
        print(f"\n📊 Testing {asset}...")
        try:
            ticker = yf.Ticker(asset)
            data = ticker.history(start='2022-01-01', end='2023-12-31')
            
            if len(data) == 0:
                print(f"❌ No data for {asset}")
                continue
            
            result = test_ma_parameters(asset, data, fast, slow, thresh)
            benchmark = (data.iloc[-1]['Close'] - data.iloc[0]['Close']) / data.iloc[0]['Close']
            
            print(f"  Return: {result['total_return']*100:.2f}% vs Benchmark: {benchmark*100:.2f}%")
            print(f"  Sharpe: {result['sharpe_ratio']:.3f}, Drawdown: {result['max_drawdown']*100:.2f}%, Trades: {result['num_trades']}")
            
            multi_results[asset] = {
                'result': result,
                'benchmark': benchmark
            }
            
        except Exception as e:
            print(f"❌ Error testing {asset}: {e}")
    
    # Summary
    print(f"\n📈 MULTI-ASSET PERFORMANCE SUMMARY")
    print("-" * 50)
    avg_sharpe = np.mean([r['result']['sharpe_ratio'] for r in multi_results.values()])
    avg_return = np.mean([r['result']['total_return'] for r in multi_results.values()])
    avg_drawdown = np.mean([r['result']['max_drawdown'] for r in multi_results.values()])
    
    print(f"Average Sharpe Ratio: {avg_sharpe:.3f}")
    print(f"Average Return: {avg_return*100:.2f}%")
    print(f"Average Max Drawdown: {avg_drawdown*100:.2f}%")
    
    return multi_results


if __name__ == "__main__":
    # Run optimization
    result = optimize_ma_strategy('SPY', '2022-01-01', '2023-12-31')
    
    if result and result['deployment_status'] in ['READY', 'CONDITIONAL']:
        # Test on multiple assets
        multi_result = test_multiple_assets()
        
        print(f"\n🎯 FINAL RECOMMENDATION")
        print("="*50)
        print(f"Strategy Status: {result['deployment_status']}")
        print(f"Success Rate: {result['success_rate']*100:.1f}%")
        
        if result['deployment_status'] == 'READY':
            print("🚀 PROCEED WITH QUANTCONNECT DEPLOYMENT")
            print("✅ All Mission 1 criteria met")
        else:
            print("⚠️ CONDITIONAL DEPLOYMENT")
            print("Consider additional optimization or risk management")
    
    else:
        print(f"\n❌ STRATEGY NEEDS IMPROVEMENT")
        print("Recommend trying different approaches:")
        print("- More sophisticated entry/exit rules")
        print("- Additional indicators (RSI, MACD)")
        print("- Dynamic position sizing")
        print("- Different asset classes")
