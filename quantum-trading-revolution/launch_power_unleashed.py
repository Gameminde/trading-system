#!/usr/bin/env python3
"""
Simple launcher for the POWER UNLEASHED agent
"""

import sys
import os
import logging
import json
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch the POWER UNLEASHED agent"""
    
    print(f"LAUNCHING POWER UNLEASHED AGENT at {datetime.now()}")
    
    try:
        # Import the POWER UNLEASHED agent
        from AGENT_POWER_UNLEASHED import PowerUnleashedConfig, PowerUnleashedTradingAgent
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
        )
        
        # Create configuration
        config = PowerUnleashedConfig()
        
        # Create agent
        agent = PowerUnleashedTradingAgent(config)
        
        print("POWER UNLEASHED AGENT created successfully")
        print(f"CONFIGURATION: Quantum boost {config.quantum_boost_factor:.1f}x, Smoothing {config.quantum_smoothing_factor:.0%}")
        print(f"QUALITY FILTERS: {config.quality_filter_minimum:.0%}+ minimum signal (POWER UNLEASHED)")
        
        # Define symbols
        symbols = ["BTC", "ETH", "AAPL", "TSLA", "MSFT", "GOOGL"]
        
        print(f"TRADING SYMBOLS: {', '.join(symbols)}")
        print("STARTING POWER UNLEASHED session (3 minutes)...")
        
        # Run the session
        start_time = time.time()
        results = agent.run_power_unleashed_session(symbols, duration_minutes=3)
        end_time = time.time()
        
        session_duration = end_time - start_time
        
        print(f"\nPOWER UNLEASHED SESSION COMPLETED in {session_duration:.1f} seconds")
        print("\nFINAL RESULTS:")
        print(f"   Cycles: {results['cycles']}")
        print(f"   Trades executed: {results['trades_executed']}")
        print(f"   Portfolio: ${results['final_portfolio']:.2f}")
        print(f"   Return: {results['return_percent']:.1f}%")
        print(f"   Success rate: {results['success_rate']:.1f}%")
        print(f"   Positions open: {results['positions_count']}")
        
        # Save results
        results['session_duration_seconds'] = session_duration
        results['completion_timestamp'] = datetime.now().isoformat()
        
        with open('../logs/power_unleashed_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("Results saved to power_unleashed_results.json")
        
        # Performance summary
        if results['trades_executed'] > 0:
            print(f"\nPERFORMANCE SUMMARY:")
            print(f"   Profitable trades: {results['profitable_trades']}")
            print(f"   Losing trades: {results['losing_trades']}")
            print(f"   Average profit: ${results['avg_profit']:.2f}")
            print(f"   Average loss: ${results['avg_loss']:.2f}")
        
        print("\nPOWER UNLEASHED SESSION COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"ERROR launching POWER UNLEASHED agent: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
