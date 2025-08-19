"""
Debug Backtesting Tool - Strategy Analysis
Quantum Trading Revolution - Mission 1A Debug

Tool to debug why our MA crossover strategy isn't generating signals.
Analyzes MA values, crossovers, and threshold conditions.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def debug_ma_strategy(symbol='SPY', start_date='2023-07-01', end_date='2024-01-01', fast_period=10, slow_period=30):
    """Debug MA strategy to see what's happening"""
    
    print(f"🔍 DEBUG MODE - MA Crossover Analysis")
    print(f"Symbol: {symbol}, Period: {start_date} to {end_date}")
    print(f"Fast MA: {fast_period}, Slow MA: {slow_period}")
    print("="*60)
    
    # Download data
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    if len(data) == 0:
        print("❌ No data downloaded")
        return
    
    print(f"✅ Downloaded {len(data)} days of data")
    print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print()
    
    # Calculate MAs
    data['Fast_MA'] = data['Close'].rolling(window=fast_period).mean()
    data['Slow_MA'] = data['Close'].rolling(window=slow_period).mean()
    data['MA_Diff'] = data['Fast_MA'] - data['Slow_MA']
    data['MA_Separation'] = abs(data['MA_Diff']) / data['Slow_MA']
    
    # Identify crossovers
    data['Bullish_Cross'] = (data['MA_Diff'] > 0) & (data['MA_Diff'].shift(1) <= 0)
    data['Bearish_Cross'] = (data['MA_Diff'] < 0) & (data['MA_Diff'].shift(1) >= 0)
    
    # Find all crossovers
    bullish_crosses = data[data['Bullish_Cross']]
    bearish_crosses = data[data['Bearish_Cross']]
    
    print(f"📈 CROSSOVER ANALYSIS")
    print("-" * 40)
    print(f"Total Bullish Crossovers: {len(bullish_crosses)}")
    print(f"Total Bearish Crossovers: {len(bearish_crosses)}")
    
    # Analyze separation at crossovers
    if len(bullish_crosses) > 0:
        print(f"\n🔍 BULLISH CROSSOVERS:")
        for i, (date, row) in enumerate(bullish_crosses.iterrows()):
            separation = row['MA_Separation']
            print(f"  {i+1}. {date.strftime('%Y-%m-%d')}: Separation {separation*100:.3f}% (Threshold: 1.0%)")
            if separation >= 0.01:
                print(f"    ✅ Above threshold - Would generate BUY signal")
            else:
                print(f"    ❌ Below threshold - Signal filtered out")
    
    if len(bearish_crosses) > 0:
        print(f"\n🔍 BEARISH CROSSOVERS:")
        for i, (date, row) in enumerate(bearish_crosses.iterrows()):
            separation = row['MA_Separation']
            print(f"  {i+1}. {date.strftime('%Y-%m-%d')}: Separation {separation*100:.3f}% (Threshold: 1.0%)")
            if separation >= 0.01:
                print(f"    ✅ Above threshold - Would generate SELL signal")
            else:
                print(f"    ❌ Below threshold - Signal filtered out")
    
    # Statistics
    print(f"\n📊 MA STATISTICS")
    print("-" * 40)
    avg_separation = data['MA_Separation'].mean()
    max_separation = data['MA_Separation'].max()
    min_separation = data['MA_Separation'].min()
    
    print(f"Average MA Separation: {avg_separation*100:.3f}%")
    print(f"Maximum MA Separation: {max_separation*100:.3f}%")
    print(f"Minimum MA Separation: {min_separation*100:.3f}%")
    
    # Check threshold effectiveness
    above_threshold = (data['MA_Separation'] >= 0.01).sum()
    total_days = len(data)
    print(f"Days above 1% threshold: {above_threshold}/{total_days} ({above_threshold/total_days*100:.1f}%)")
    
    # Try different thresholds
    print(f"\n🔧 THRESHOLD ANALYSIS")
    print("-" * 40)
    thresholds = [0.005, 0.0075, 0.01, 0.015, 0.02]  # 0.5% to 2%
    
    for thresh in thresholds:
        # Count valid signals for each threshold
        valid_bullish = bullish_crosses[bullish_crosses['MA_Separation'] >= thresh]
        valid_bearish = bearish_crosses[bearish_crosses['MA_Separation'] >= thresh]
        total_signals = len(valid_bullish) + len(valid_bearish)
        
        print(f"Threshold {thresh*100:.2f}%: {total_signals} signals ({len(valid_bullish)} BUY, {len(valid_bearish)} SELL)")
    
    # Recommend optimal threshold
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 40)
    
    # Find threshold that gives reasonable number of signals (2-8 per year)
    target_signals_per_year = 6
    days_in_period = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    years_in_period = days_in_period / 365.25
    target_signals = int(target_signals_per_year * years_in_period)
    
    best_threshold = 0.01
    best_diff = float('inf')
    
    for thresh in np.arange(0.001, 0.05, 0.001):  # Test thresholds from 0.1% to 5%
        valid_bullish = bullish_crosses[bullish_crosses['MA_Separation'] >= thresh]
        valid_bearish = bearish_crosses[bearish_crosses['MA_Separation'] >= thresh]
        total_signals = len(valid_bullish) + len(valid_bearish)
        
        diff = abs(total_signals - target_signals)
        if diff < best_diff:
            best_diff = diff
            best_threshold = thresh
    
    print(f"Optimal threshold: {best_threshold*100:.2f}% (targets ~{target_signals} signals)")
    
    # Test optimal threshold
    valid_bullish_opt = bullish_crosses[bullish_crosses['MA_Separation'] >= best_threshold]
    valid_bearish_opt = bearish_crosses[bearish_crosses['MA_Separation'] >= best_threshold]
    
    print(f"With optimal threshold:")
    print(f"  - {len(valid_bullish_opt)} BUY signals")
    print(f"  - {len(valid_bearish_opt)} SELL signals")
    print(f"  - {len(valid_bullish_opt) + len(valid_bearish_opt)} total signals")
    
    # Show recent data
    print(f"\n📋 RECENT DATA SAMPLE (Last 10 days)")
    print("-" * 40)
    recent_data = data.tail(10)[['Close', 'Fast_MA', 'Slow_MA', 'MA_Diff', 'MA_Separation']]
    for date, row in recent_data.iterrows():
        if not pd.isna(row['Fast_MA']):
            print(f"{date.strftime('%Y-%m-%d')}: Close=${row['Close']:.2f}, Fast=${row['Fast_MA']:.2f}, Slow=${row['Slow_MA']:.2f}, Sep={row['MA_Separation']*100:.2f}%")
    
    return {
        'data': data,
        'bullish_crosses': bullish_crosses,
        'bearish_crosses': bearish_crosses,
        'optimal_threshold': best_threshold,
        'total_crossovers': len(bullish_crosses) + len(bearish_crosses)
    }


def quick_backtest_with_threshold(symbol='SPY', start_date='2023-07-01', end_date='2024-01-01', threshold=0.005):
    """Quick backtest with specified threshold"""
    
    print(f"\n🚀 QUICK BACKTEST WITH THRESHOLD {threshold*100:.2f}%")
    print("="*50)
    
    # Download data
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    # Calculate MAs
    data['Fast_MA'] = data['Close'].rolling(window=10).mean()
    data['Slow_MA'] = data['Close'].rolling(window=30).mean()
    data['MA_Diff'] = data['Fast_MA'] - data['Slow_MA']
    data['MA_Separation'] = abs(data['MA_Diff']) / data['Slow_MA']
    
    # Find valid crossovers
    data['Bullish_Cross'] = (data['MA_Diff'] > 0) & (data['MA_Diff'].shift(1) <= 0) & (data['MA_Separation'] >= threshold)
    data['Bearish_Cross'] = (data['MA_Diff'] < 0) & (data['MA_Diff'].shift(1) >= 0) & (data['MA_Separation'] >= threshold)
    
    # Simulate trades
    capital = 100000
    cash = capital
    shares = 0
    position = None
    trades = []
    
    for date, row in data.iterrows():
        if row['Bullish_Cross'] and position is None:
            # Buy
            shares_to_buy = int(cash * 0.95 / row['Close'])
            if shares_to_buy > 0:
                cost = shares_to_buy * row['Close'] * 1.001  # 0.1% transaction cost
                if cash >= cost:
                    shares = shares_to_buy
                    cash -= cost
                    position = 'LONG'
                    trades.append(('BUY', date, row['Close'], shares_to_buy))
                    print(f"📈 BUY: {shares_to_buy} shares at ${row['Close']:.2f} on {date.strftime('%Y-%m-%d')}")
        
        elif row['Bearish_Cross'] and position == 'LONG':
            # Sell
            proceeds = shares * row['Close'] * 0.999  # 0.1% transaction cost
            cash += proceeds
            trades.append(('SELL', date, row['Close'], -shares))
            print(f"📉 SELL: {shares} shares at ${row['Close']:.2f} on {date.strftime('%Y-%m-%d')}")
            shares = 0
            position = None
    
    # Final value
    final_price = data.iloc[-1]['Close']
    final_value = cash + (shares * final_price if shares > 0 else 0)
    total_return = (final_value - capital) / capital
    
    print(f"\nFinal Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Number of Trades: {len([t for t in trades if t[0] == 'BUY'])}")
    
    return final_value, total_return


if __name__ == "__main__":
    # Debug the current strategy
    result = debug_ma_strategy('SPY', '2023-07-01', '2024-01-01')
    
    # Test with optimal threshold
    if result and result['optimal_threshold']:
        print("\n" + "="*60)
        quick_backtest_with_threshold('SPY', '2023-07-01', '2024-01-01', result['optimal_threshold'])
    
    # Also test with a longer period
    print("\n" + "="*60)
    print("🔍 EXTENDED PERIOD ANALYSIS (2022-2024)")
    result_long = debug_ma_strategy('SPY', '2022-01-01', '2024-01-01')
    
    if result_long and result_long['optimal_threshold']:
        quick_backtest_with_threshold('SPY', '2022-01-01', '2024-01-01', result_long['optimal_threshold'])
