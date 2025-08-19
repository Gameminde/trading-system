#!/usr/bin/env python3
"""
Extended launcher for the optimized trading agent with proper session completion
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
    """Launch the optimized agent with extended session"""
    
    print(f"LAUNCHING OPTIMIZED TRADING AGENT - EXTENDED SESSION at {datetime.now()}")
    
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
        print("STARTING extended session (5 minutes)...")
        
        # Run the extended session
        start_time = time.time()
        results = agent.run_optimized_session(symbols, duration_minutes=5)
        end_time = time.time()
        
        session_duration = end_time - start_time
        
        print(f"\nSESSION COMPLETED in {session_duration:.1f} seconds")
        print("\nFINAL RESULTS:")
        print(f"   Cycles completed: {results['cycles']}")
        print(f"   Trades executed: {results['trades_executed']}")
        print(f"   Portfolio final: ${results['final_portfolio']:.2f}")
        print(f"   Total return: ${results['total_return']:.2f}")
        print(f"   Return percent: {results['return_percent']:.1f}%")
        print(f"   Success rate: {results['success_rate']:.1f}%")
        print(f"   Positions open: {results['positions_count']}")
        
        # Save detailed results
        results['session_duration_seconds'] = session_duration
        results['completion_timestamp'] = datetime.now().isoformat()
        
        with open('../logs/agent_results_extended.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("RESULTS saved to agent_results_extended.json")
        
        # Performance summary
        if results['trades_executed'] > 0:
            print(f"\nPERFORMANCE SUMMARY:")
            print(f"   Profitable trades: {results['profitable_trades']}")
            print(f"   Losing trades: {results['losing_trades']}")
            print(f"   Average profit: ${results['avg_profit']:.2f}")
            print(f"   Average loss: ${results['avg_loss']:.2f}")
            print(f"   Average hold time: {results['avg_hold_time']:.1f} minutes")
        
        print("\nAGENT SESSION COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"ERROR launching agent: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error info
        error_info = {
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc()
        }
        
        try:
            with open('../logs/agent_error.json', 'w') as f:
                json.dump(error_info, f, indent=2)
            print("Error details saved to agent_error.json")
        except:
            pass
            
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
