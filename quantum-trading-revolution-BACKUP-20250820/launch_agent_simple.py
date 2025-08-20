#!/usr/bin/env python3
"""
Simple launcher for the optimized trading agent
"""

import sys
import os
import logging
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch the optimized agent in non-interactive mode"""
    
    print(f"LAUNCHING OPTIMIZED TRADING AGENT at {datetime.now()}")
    
    try:
        # Import the agent
        from AGENT_OPTIMIZED_MAXIMUM_POWER import OptimizedConfig, OptimizedTradingAgent
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
        )
        
        # Create configuration
        config = OptimizedConfig()
        
        # Create agent
        agent = OptimizedTradingAgent(config)
        
        print("AGENT created successfully")
        print(f"CONFIGURATION: Quantum boost {config.quantum_boost_factor:.1f}x, Smoothing {config.quantum_smoothing_factor:.0%}")
        
        # Define symbols
        symbols = ["BTC", "ETH", "AAPL", "TSLA"]
        
        print(f"TRADING SYMBOLS: {', '.join(symbols)}")
        print("STARTING optimized session...")
        
        # Run the session
        results = agent.run_optimized_session(symbols, duration_minutes=1)
        
        print("\nSESSION RESULTS:")
        print(f"   Cycles: {results['cycles']}")
        print(f"   Trades executed: {results['trades_executed']}")
        print(f"   Portfolio: ${results['final_portfolio']:.2f}")
        print(f"   Return: {results['return_percent']:.1f}%")
        print(f"   Success rate: {results['success_rate']:.1f}%")
        
        # Save results to log
        with open('../logs/agent_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("SESSION completed and results saved")
        return True
        
    except Exception as e:
        print(f"ERROR launching agent: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
