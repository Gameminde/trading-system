#!/usr/bin/env python3
"""
Background launcher for the optimized trading agent
"""

import sys
import os
import logging
import json
import time
import threading
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_agent_in_thread(agent, symbols, duration_minutes, results_container):
    """Run agent in a separate thread to handle time.sleep properly"""
    try:
        results = agent.run_optimized_session(symbols, duration_minutes)
        results_container['results'] = results
        results_container['completed'] = True
    except Exception as e:
        results_container['error'] = str(e)
        results_container['completed'] = True

def main():
    """Launch the optimized agent in background mode"""
    
    print(f"LAUNCHING OPTIMIZED TRADING AGENT - BACKGROUND MODE at {datetime.now()}")
    
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
        print("STARTING background session (3 minutes)...")
        
        # Container for results
        results_container = {'completed': False, 'results': None, 'error': None}
        
        # Start agent in background thread
        agent_thread = threading.Thread(
            target=run_agent_in_thread,
            args=(agent, symbols, 3, results_container)
        )
        agent_thread.daemon = True
        agent_thread.start()
        
        # Monitor progress
        start_time = time.time()
        max_wait_time = 200  # 3 minutes + buffer
        
        print("MONITORING agent progress...")
        
        while not results_container['completed'] and (time.time() - start_time) < max_wait_time:
            time.sleep(5)  # Check every 5 seconds
            elapsed = time.time() - start_time
            print(f"   Elapsed: {elapsed:.0f}s - Waiting for completion...")
        
        if results_container['completed']:
            if 'error' in results_container and results_container['error']:
                print(f"AGENT ERROR: {results_container['error']}")
                return False
            
            results = results_container['results']
            session_duration = time.time() - start_time
            
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
            
            with open('../logs/agent_results_background.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print("RESULTS saved to agent_results_background.json")
            
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
        else:
            print("AGENT SESSION TIMEOUT - Agent did not complete in time")
            return False
        
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
            with open('../logs/agent_error_background.json', 'w') as f:
                json.dump(error_info, f, indent=2)
            print("Error details saved to agent_error_background.json")
        except:
            pass
            
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
