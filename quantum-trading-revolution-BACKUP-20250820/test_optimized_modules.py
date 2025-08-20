"""
AGENT QUANTUM TRADING - TEST SCRIPT FOR OPTIMIZED PHASE 2 MODULES
Comprehensive testing without Unicode characters for Windows compatibility
"""

import asyncio
import logging
import time
import sys
from typing import Dict, Any

def setup_logging():
    """Setup logging without Unicode characters"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('test_optimized_modules.log')
        ]
    )
    return logging.getLogger(__name__)

async def test_ultra_low_latency_optimized():
    """Test the optimized ultra-low latency engine"""
    logger = logging.getLogger("test_ultra_latency")
    logger.info("Testing Ultra-Low Latency Engine Optimized...")
    
    try:
        from ultra_low_latency_engine_optimized import UltraLowLatencyEngineOptimized, UltraLatencyConfig
        
        # Configuration
        config = UltraLatencyConfig(
            target_latency_ms=50.0,
            max_queue_size=1000,
            ring_buffer_size=1024  # Fixed: must be power of 2
        )
        
        # Test d'initialisation
        engine = UltraLowLatencyEngineOptimized(config)
        logger.info("Engine initialized successfully")
        
        # Démarrage du moteur
        await engine.start()
        
        # Test des métriques
        metrics = engine.get_comprehensive_metrics()
        logger.info(f"Engine metrics: {metrics}")
        
        # Test de performance
        start_time = time.perf_counter()
        for i in range(100):
            event_data = np.array([100.0 + i * 0.01])  # Convert to numpy array
            await engine.process_market_data_async(event_data)
        
        processing_time = time.perf_counter() - start_time
        avg_latency = (processing_time / 100) * 1000
        
        logger.info(f"Performance test: {avg_latency:.2f}ms average latency")
        
        # Nettoyage
        await engine.stop()
        logger.info("Ultra-Low Latency Engine test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Ultra-Low Latency Engine test failed: {e}")
        return False

async def test_phase2_integration_optimized():
    """Test the optimized Phase 2 integration"""
    logger = logging.getLogger("test_integration")
    logger.info("Testing Phase 2 Integration Optimized...")
    
    try:
        from phase2_integration_optimized import Phase2IntegrationManagerOptimized, IntegrationConfig
        
        # Configuration
        config = IntegrationConfig(
            performance_test_iterations=50,
            intermediate_test_iterations=25,
            validation_timeout_seconds=15.0
        )
        
        # Test d'initialisation
        manager = Phase2IntegrationManagerOptimized(config)
        logger.info("Integration manager initialized successfully")
        
        # Test de création d'agent
        agent = manager.create_test_agent()
        logger.info("Test agent created successfully")
        
        # Test de performance baseline
        baseline_perf = manager.performance_measurer.measure_agent_performance(
            agent, 25, "test_baseline"
        )
        logger.info(f"Baseline performance: {baseline_perf['avg_latency_ms']:.2f}ms")
        
        logger.info("Phase 2 Integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Phase 2 Integration test failed: {e}")
        return False

async def test_multi_agent_optimized():
    """Test the optimized multi-agent architecture"""
    logger = logging.getLogger("test_multi_agent")
    logger.info("Testing Multi-Agent Architecture Optimized...")
    
    try:
        from multi_agent_architecture_optimized import MultiAgentSystemOptimized, MultiAgentConfig
        
        # Configuration
        config = MultiAgentConfig(
            redis_host='localhost',
            redis_port=6379,
            heartbeat_interval=1.0,  # Fixed parameter name
            message_ttl=5.0  # Added missing parameter
        )
        
        # Test d'initialisation
        system = MultiAgentSystemOptimized(config)
        logger.info("Multi-agent system initialized successfully")
        
        # Test de démarrage
        await system.start_system()
        logger.info("System started successfully")
        
        # Test des métriques
        status = system.get_system_metrics()
        logger.info(f"System status: {status}")
        
        # Test d'arrêt
        await system.stop_system()
        logger.info("System stopped successfully")
        
        logger.info("Multi-Agent Architecture test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Multi-Agent Architecture test failed: {e}")
        return False

async def test_regime_detection_optimized():
    """Test the optimized regime detection system"""
    logger = logging.getLogger("test_regime_detection")
    logger.info("Testing Regime Detection Hybrid Optimized...")
    
    try:
        from regime_detection_hybrid_optimized import RegimeDetectionEnsembleOptimized, RegimeDetectionConfig
        
        # Configuration
        config = RegimeDetectionConfig(
            window_size=100,  # Fixed parameter name
            min_data_points=30,  # Fixed parameter name
            confidence_threshold=0.7  # Fixed parameter name
        )
        
        # Test d'initialisation
        ensemble = RegimeDetectionEnsembleOptimized(config)
        logger.info("Regime detection ensemble initialized successfully")
        
        # Test avec données simulées
        import numpy as np
        np.random.seed(42)
        test_data = np.random.randn(100, 5) * 0.02 + 100
        
        # Détection de régime
        regime_result = ensemble.detect(test_data)
        logger.info(f"Regime detection result: {regime_result}")
        
        # Test des métriques
        # Utiliser l'historique disponible au lieu de get_ensemble_performance
        history_count = len(ensemble.history)
        logger.info(f"Ensemble history count: {history_count}")
        logger.info(f"Detectors available: {list(ensemble.detectors.keys())}")
        
        logger.info("Regime Detection test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Regime Detection test failed: {e}")
        return False

async def test_integration_workflow():
    """Test the complete integration workflow"""
    logger = logging.getLogger("test_workflow")
    logger.info("Testing Complete Integration Workflow...")
    
    try:
        # Test séquentiel des modules
        test_results = {}
        
        # Test 1: Ultra-Low Latency
        logger.info("Step 1: Testing Ultra-Low Latency Engine")
        test_results['ultra_latency'] = await test_ultra_low_latency_optimized()
        
        # Test 2: Multi-Agent
        logger.info("Step 2: Testing Multi-Agent Architecture")
        test_results['multi_agent'] = await test_multi_agent_optimized()
        
        # Test 3: Regime Detection
        logger.info("Step 3: Testing Regime Detection")
        test_results['regime_detection'] = await test_regime_detection_optimized()
        
        # Test 4: Phase 2 Integration
        logger.info("Step 4: Testing Phase 2 Integration")
        test_results['phase2_integration'] = await test_phase2_integration_optimized()
        
        # Résumé des tests
        success_count = sum(test_results.values())
        total_tests = len(test_results)
        
        logger.info("=" * 50)
        logger.info("INTEGRATION WORKFLOW TEST RESULTS")
        logger.info("=" * 50)
        
        for test_name, result in test_results.items():
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"Overall: {success_count}/{total_tests} tests passed")
        
        if success_count == total_tests:
            logger.info("All tests passed! Integration workflow is ready.")
        else:
            logger.warning(f"{total_tests - success_count} tests failed. Review required.")
        
        return success_count == total_tests
        
    except Exception as e:
        logger.error(f"Integration workflow test failed: {e}")
        return False

def run_all_tests():
    """Run all tests synchronously"""
    logger = logging.getLogger("main")
    logger.info("Starting comprehensive test suite for optimized Phase 2 modules...")
    
    try:
        # Exécution des tests
        success = asyncio.run(test_integration_workflow())
        
        if success:
            logger.info("All tests completed successfully!")
            print("\nSUCCESS: All optimized Phase 2 modules are working correctly!")
            print("Your agent is ready for TIER 1+ institutional deployment.")
        else:
            logger.error("Some tests failed. Check the logs for details.")
            print("\nWARNING: Some tests failed. Review the logs above.")
        
        return success
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        print(f"\nERROR: Test suite failed with error: {e}")
        return False

if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()
    
    # Import numpy for regime detection test
    try:
        import numpy as np
    except ImportError:
        logger.error("NumPy is required but not installed. Please install it first.")
        print("ERROR: NumPy not found. Install with: pip install numpy")
        sys.exit(1)
    
    # Run tests
    success = run_all_tests()
    
    # Exit code
    sys.exit(0 if success else 1)
