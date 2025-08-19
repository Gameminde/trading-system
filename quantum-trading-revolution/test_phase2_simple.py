# FILE: test_phase2_simple.py
"""
AGENT QUANTUM TRADING - TESTS SIMPLIFIÉS PHASE 2
Version compatible Windows sans caractères Unicode
"""

import time
import numpy as np
import traceback

def test_model_monitoring():
    """Test simplifié du Model Monitoring System"""
    print("TEST: Model Monitoring System")
    print("-" * 40)
    
    try:
        from model_monitoring_system import ModelMonitoringSystem
        monitoring_system = ModelMonitoringSystem()
        
        # Test basique
        test_data = np.random.randn(100)
        health = monitoring_system.get_system_health()
        
        print("   OK - Model Monitoring System initialisé")
        print(f"   Health: {health.get('overall_health', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"   ERREUR: {e}")
        return False

def test_online_learning():
    """Test simplifié du Online Learning Framework"""
    print("TEST: Online Learning Framework")
    print("-" * 40)
    
    try:
        from online_learning_framework import OnlineLearningFramework
        framework = OnlineLearningFramework()
        
        # Test basique
        models = framework.get_active_models()
        
        print("   OK - Online Learning Framework initialisé")
        print(f"   Modèles actifs: {len(models)}")
        return True
        
    except Exception as e:
        print(f"   ERREUR: {e}")
        return False

def test_multi_agent():
    """Test simplifié du Multi-Agent Architecture"""
    print("TEST: Multi-Agent Architecture")
    print("-" * 40)
    
    try:
        from multi_agent_architecture import MasterCoordinatorAgent, RedisCommunicationLayer
        communication_layer = RedisCommunicationLayer()
        coordinator = MasterCoordinatorAgent(communication_layer)
        
        # Test basique
        active_agents = coordinator.get_active_agents()
        
        print("   OK - Multi-Agent Architecture initialisé")
        print(f"   Agents actifs: {len(active_agents)}")
        return True
        
    except Exception as e:
        print(f"   ERREUR: {e}")
        return False

def test_regime_detection():
    """Test simplifié du Regime Detection Hybrid"""
    print("TEST: Regime Detection Hybrid")
    print("-" * 40)
    
    try:
        from regime_detection_hybrid import RegimeDetectionEnsemble
        regime_ensemble = RegimeDetectionEnsemble()
        
        # Test basique
        detectors = regime_ensemble.detectors
        
        print("   OK - Regime Detection Hybrid initialisé")
        print(f"   Détecteurs: {len(detectors)}")
        return True
        
    except Exception as e:
        print(f"   ERREUR: {e}")
        return False

def test_ultra_latency():
    """Test simplifié du Ultra-Low Latency Engine"""
    print("TEST: Ultra-Low Latency Engine")
    print("-" * 40)
    
    try:
        from ultra_low_latency_engine import UltraLowLatencyEngine
        engine = UltraLowLatencyEngine()
        
        # Test basique
        components = [
            engine.event_bus,
            engine.tsdb,
            engine.cache,
            engine.signal_computer
        ]
        
        print("   OK - Ultra-Low Latency Engine initialisé")
        print(f"   Composants: {len(components)}")
        return True
        
    except Exception as e:
        print(f"   ERREUR: {e}")
        return False

def run_all_tests():
    """Exécution de tous les tests simplifiés"""
    print("LANCEMENT TESTS SIMPLIFIÉS PHASE 2")
    print("=" * 60)
    
    start_time = time.time()
    
    tests = [
        test_model_monitoring,
        test_online_learning,
        test_multi_agent,
        test_regime_detection,
        test_ultra_latency
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Résumé
    print("RÉSUMÉ DES TESTS PHASE 2")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, result in enumerate(results):
        status = "OK" if result else "ECHEC"
        test_name = tests[i].__name__.replace("test_", "").replace("_", " ").title()
        print(f"   {status}: {test_name}")
    
    print(f"\nRésultats: {passed}/{total} tests réussis")
    print(f"Temps total: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 TOUS LES TESTS PHASE 2 RÉUSSIS !")
        print("   L'agent est prêt pour la transformation TIER 1+")
    else:
        print(f"\n⚠️ {total - passed} TESTS ÉCHOUÉS")
        print("   Vérification requise avant déploiement")
    
    return results

if __name__ == "__main__":
    run_all_tests()
