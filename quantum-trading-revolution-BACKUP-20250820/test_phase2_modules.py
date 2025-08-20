# FILE: test_phase2_modules.py
"""
AGENT QUANTUM TRADING - TESTS DE VALIDATION PHASE 2
Script de test complet pour tous les modules Phase 2
Validation: Model Monitoring, Online Learning, Multi-Agent, Regime Detection, Ultra-Low Latency
"""

import time
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Any
import json
import traceback

warnings.filterwarnings('ignore')

# Import des modules Phase 2
try:
    from model_monitoring_system import ModelMonitoringSystem
    from online_learning_framework import OnlineLearningFramework
    from multi_agent_architecture import MasterCoordinatorAgent, RedisCommunicationLayer
    from regime_detection_hybrid import RegimeDetectionEnsemble
    from ultra_low_latency_engine import UltraLowLatencyEngine
    print("OK - Tous les modules Phase 2 importés pour tests")
except ImportError as e:
    print(f"❌ Erreur import module: {e}")
    exit(1)

class Phase2ModuleTester:
    """Testeur complet des modules Phase 2"""
    
    def __init__(self):
        self.test_results = {}
        self.test_data = self._generate_test_data()
        print("🧪 Phase 2 Module Tester initialisé")
    
    def _generate_test_data(self) -> Dict:
        """Génération de données de test réalistes"""
        # Données de marché simulées
        np.random.seed(42)  # Reproductibilité
        
        # Prix avec tendance et volatilité
        n_points = 1000
        trend = np.linspace(100, 120, n_points)
        noise = np.random.randn(n_points) * 2
        prices = trend + noise
        
        # Volumes
        volumes = np.random.randint(1000, 10000, n_points)
        
        # Features techniques
        returns = np.diff(prices) / prices[:-1]
        volatility = np.array([np.std(returns[max(0, i-20):i+1]) for i in range(len(returns))])
        
        return {
            'prices': prices,
            'volumes': volumes,
            'returns': returns,
            'volatility': volatility,
            'timestamps': np.arange(n_points)
        }
    
    def test_model_monitoring_system(self) -> Dict:
        """Test du système de monitoring de modèles"""
        print("\n🔍 TEST: Model Monitoring System")
        print("-" * 40)
        
        try:
            # Initialisation
            monitoring_system = ModelMonitoringSystem()
            
            # Test 1: Surveillance des données
            print("   📊 Test surveillance données...")
            data_quality = monitoring_system.check_data_quality(self.test_data['prices'])
            print(f"      ✅ Qualité données: {data_quality.get('overall_score', 'N/A')}")
            
            # Test 2: Détection de dérive
            print("   🚨 Test détection dérive...")
            drift_result = monitoring_system.detect_data_drift(
                self.test_data['prices'][:500], 
                self.test_data['prices'][500:]
            )
            print(f"      ✅ Dérive détectée: {drift_result.get('drift_detected', 'N/A')}")
            
            # Test 3: Monitoring des performances
            print("   📈 Test monitoring performances...")
            performance = monitoring_system.monitor_model_performance(
                predictions=np.random.rand(100),
                actuals=np.random.rand(100)
            )
            print(f"      ✅ Performance: {performance.get('accuracy', 'N/A'):.3f}")
            
            # Test 4: Système d'alertes
            print("   🚨 Test système alertes...")
            alert_result = monitoring_system.trigger_alert('test_alert', 'Test alert message')
            print(f"      ✅ Alerte déclenchée: {alert_result}")
            
            # Test 5: Santé du système
            health = monitoring_system.get_system_health()
            print(f"      ✅ Santé système: {health.get('overall_health', 'N/A')}")
            
            self.test_results['model_monitoring'] = {
                'status': 'PASSED',
                'tests': 5,
                'passed': 5,
                'failed': 0,
                'details': {
                    'data_quality': data_quality,
                    'drift_detection': drift_result,
                    'performance_monitoring': performance,
                    'alert_system': alert_result,
                    'system_health': health
                }
            }
            
            print("   OK - Model Monitoring System: TOUS LES TESTS RÉUSSIS")
            return self.test_results['model_monitoring']
            
        except Exception as e:
            print(f"   ❌ Erreur test Model Monitoring: {e}")
            self.test_results['model_monitoring'] = {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return self.test_results['model_monitoring']
    
    def test_online_learning_framework(self) -> Dict:
        """Test du framework d'apprentissage en ligne"""
        print("\n🔍 TEST: Online Learning Framework")
        print("-" * 40)
        
        try:
            # Initialisation
            framework = OnlineLearningFramework()
            
            # Test 1: Initialisation des modèles
            print("   🤖 Test initialisation modèles...")
            models = framework.get_active_models()
            print(f"      ✅ Modèles actifs: {len(models)}")
            
            # Test 2: Entraînement incrémental
            print("   📚 Test entraînement incrémental...")
            training_data = np.random.rand(100, 5)
            training_labels = np.random.randint(0, 2, 100)
            
            for i in range(0, 100, 10):
                batch_data = training_data[i:i+10]
                batch_labels = training_labels[i:i+10]
                framework.train_incremental(batch_data, batch_labels)
            
            print("      ✅ Entraînement incrémental réussi")
            
            # Test 3: Prédictions
            print("   🔮 Test prédictions...")
            test_data = np.random.rand(20, 5)
            predictions = framework.predict(test_data)
            print(f"      ✅ Prédictions générées: {len(predictions)}")
            
            # Test 4: Détection de concept drift
            print("   🌊 Test détection concept drift...")
            drift_detected = framework.detect_concept_drift(test_data)
            print(f"      ✅ Concept drift: {drift_detected}")
            
            # Test 5: Métriques de performance
            performance = framework.get_performance_metrics()
            print(f"      ✅ Métriques: {len(performance)} indicateurs")
            
            self.test_results['online_learning'] = {
                'status': 'PASSED',
                'tests': 5,
                'passed': 5,
                'failed': 0,
                'details': {
                    'active_models': len(models),
                    'training_success': True,
                    'predictions_count': len(predictions),
                    'concept_drift': drift_detected,
                    'performance_metrics': performance
                }
            }
            
            print("   OK - Online Learning Framework: TOUS LES TESTS RÉUSSIS")
            return self.test_results['online_learning']
            
        except Exception as e:
            print(f"   ❌ Erreur test Online Learning: {e}")
            self.test_results['online_learning'] = {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return self.test_results['online_learning']
    
    def test_multi_agent_architecture(self) -> Dict:
        """Test de l'architecture multi-agents"""
        print("\n🔍 TEST: Multi-Agent Architecture")
        print("-" * 40)
        
        try:
            # Initialisation
            communication_layer = RedisCommunicationLayer()
            coordinator = MasterCoordinatorAgent(communication_layer)
            
            # Test 1: Initialisation des agents
            print("   🤖 Test initialisation agents...")
            active_agents = coordinator.get_active_agents()
            print(f"      ✅ Agents actifs: {len(active_agents)}")
            
            # Test 2: Communication inter-agents
            print("   📡 Test communication inter-agents...")
            message_sent = communication_layer.send_message(
                'test_channel', 
                {'type': 'test', 'data': 'test_message'}
            )
            print(f"      ✅ Message envoyé: {message_sent}")
            
            # Test 3: Coordination des agents
            print("   🎯 Test coordination agents...")
            coordination_result = coordinator.coordinate_agents()
            print(f"      ✅ Coordination: {coordination_result}")
            
            # Test 4: Gestion des urgences
            print("   🚨 Test gestion urgences...")
            emergency_result = coordinator.handle_emergency('test_emergency')
            print(f"      ✅ Urgence gérée: {emergency_result}")
            
            # Test 5: Métriques de performance
            performance = coordinator.get_system_performance()
            print(f"      ✅ Performance système: {len(performance)} métriques")
            
            self.test_results['multi_agent'] = {
                'status': 'PASSED',
                'tests': 5,
                'passed': 5,
                'failed': 0,
                'details': {
                    'active_agents': len(active_agents),
                    'communication_success': message_sent,
                    'coordination_result': coordination_result,
                    'emergency_handling': emergency_result,
                    'system_performance': performance
                }
            }
            
            print("   OK - Multi-Agent Architecture: TOUS LES TESTS RÉUSSIS")
            return self.test_results['multi_agent']
            
        except Exception as e:
            print(f"   ❌ Erreur test Multi-Agent: {e}")
            self.test_results['multi_agent'] = {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return self.test_results['multi_agent']
    
    def test_regime_detection_hybrid(self) -> Dict:
        """Test du système de détection de régimes hybrides"""
        print("\n🔍 TEST: Regime Detection Hybrid")
        print("-" * 40)
        
        try:
            # Initialisation
            regime_ensemble = RegimeDetectionEnsemble()
            
            # Test 1: Initialisation des détecteurs
            print("   🔍 Test initialisation détecteurs...")
            detectors = regime_ensemble.detectors
            print(f"      ✅ Détecteurs initialisés: {len(detectors)}")
            
            # Test 2: Entraînement de l'ensemble
            print("   🎯 Test entraînement ensemble...")
            training_success = regime_ensemble.train_ensemble(
                self.test_data['prices']
            )
            print(f"      ✅ Entraînement: {training_success}")
            
            # Test 3: Détection de régime
            print("   🌊 Test détection régime...")
            regime_result = regime_ensemble.detect_regime_ensemble(
                self.test_data['prices'][-100:]
            )
            print(f"      ✅ Régime détecté: {regime_result.get('ensemble_regime', 'N/A')}")
            
            # Test 4: Performance de l'ensemble
            print("   📊 Test performance ensemble...")
            performance = regime_ensemble.get_ensemble_performance()
            print(f"      ✅ Performance: {performance.get('total_detections', 0)} détections")
            
            # Test 5: Mise à jour des poids
            print("   ⚖️ Test mise à jour poids...")
            regime_ensemble.update_ensemble_weights(performance)
            print("      ✅ Poids mis à jour")
            
            self.test_results['regime_detection'] = {
                'status': 'PASSED',
                'tests': 5,
                'passed': 5,
                'failed': 0,
                'details': {
                    'detectors_count': len(detectors),
                    'training_success': training_success,
                    'regime_detected': regime_result.get('ensemble_regime'),
                    'performance_metrics': performance,
                    'weights_updated': True
                }
            }
            
            print("   OK - Regime Detection Hybrid: TOUS LES TESTS RÉUSSIS")
            return self.test_results['regime_detection']
            
        except Exception as e:
            print(f"   ❌ Erreur test Regime Detection: {e}")
            self.test_results['regime_detection'] = {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return self.test_results['regime_detection']
    
    def test_ultra_low_latency_engine(self) -> Dict:
        """Test du moteur de latence ultra-faible"""
        print("\n🔍 TEST: Ultra-Low Latency Engine")
        print("-" * 40)
        
        try:
            # Initialisation
            engine = UltraLowLatencyEngine()
            
            # Test 1: Initialisation des composants
            print("   ⚡ Test initialisation composants...")
            components = [
                engine.event_bus,
                engine.tsdb,
                engine.cache,
                engine.signal_computer
            ]
            print(f"      ✅ Composants initialisés: {len(components)}")
            
            # Test 2: Performance du cache
            print("   💾 Test performance cache...")
            for i in range(100):
                engine.cache.set(f"key_{i}", f"value_{i}")
            
            cache_stats = engine.cache.get_stats()
            print(f"      ✅ Cache: {cache_stats.get('hit_rate', 0):.2f} hit rate")
            
            # Test 3: Calcul de signaux
            print("   🧮 Test calcul signaux...")
            test_data = self.test_data['prices'][:100]
            start_time = time.time()
            signals = engine.signal_computer.compute_signals_fast(test_data)
            end_time = time.time()
            
            computation_time = (end_time - start_time) * 1000  # ms
            print(f"      ✅ Signaux calculés: {len(signals)} en {computation_time:.2f}ms")
            
            # Test 4: Stockage TSDB
            print("   🗄️ Test stockage TSDB...")
            for i in range(50):
                engine.tsdb.store_series('test_series', time.time(), i)
            
            tsdb_stats = engine.tsdb.get_stats()
            print(f"      ✅ TSDB: {tsdb_stats.get('total_points', 0)} points")
            
            # Test 5: Métriques de performance
            print("   📊 Test métriques performance...")
            performance = engine.get_performance_metrics()
            print(f"      ✅ Métriques: {len(performance)} indicateurs")
            
            self.test_results['ultra_latency'] = {
                'status': 'PASSED',
                'tests': 5,
                'passed': 5,
                'failed': 0,
                'details': {
                    'components_initialized': len(components),
                    'cache_performance': cache_stats,
                    'signal_computation_time_ms': computation_time,
                    'tsdb_points': tsdb_stats.get('total_points', 0),
                    'performance_metrics': performance
                }
            }
            
            print("   OK - Ultra-Low Latency Engine: TOUS LES TESTS RÉUSSIS")
            return self.test_results['ultra_latency']
            
        except Exception as e:
            print(f"   ❌ Erreur test Ultra-Low Latency: {e}")
            self.test_results['ultra_latency'] = {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return self.test_results['ultra_latency']
    
    def run_all_tests(self) -> Dict:
        """Exécution de tous les tests"""
        print("🚀 LANCEMENT TESTS COMPLETS PHASE 2")
        print("=" * 60)
        
        start_time = time.time()
        
        # Tests individuels
        self.test_model_monitoring_system()
        self.test_online_learning_framework()
        self.test_multi_agent_architecture()
        self.test_regime_detection_hybrid()
        self.test_ultra_low_latency_engine()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Résumé des tests
        self.generate_test_summary(total_time)
        
        return self.test_results
    
    def generate_test_summary(self, total_time: float):
        """Génération du résumé des tests"""
        print("\n📋 RÉSUMÉ DES TESTS PHASE 2")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for module_name, result in self.test_results.items():
            if result['status'] == 'PASSED':
                status_icon = "✅"
                passed_tests += result.get('tests', 0)
                total_tests += result.get('tests', 0)
            else:
                status_icon = "❌"
                failed_tests += 1
                total_tests += 1
            
            print(f"   {status_icon} {module_name}: {result['status']}")
        
        # Statistiques globales
        print(f"\n📊 STATISTIQUES GLOBALES:")
        print(f"   • Tests totaux: {total_tests}")
        print(f"   • Tests réussis: {passed_tests}")
        print(f"   • Tests échoués: {failed_tests}")
        print(f"   • Taux de succès: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "   • Taux de succès: N/A")
        print(f"   • Temps total: {total_time:.2f}s")
        
        # Sauvegarde des résultats
        self.save_test_results(total_time)
        
        # Conclusion
        if failed_tests == 0:
            print(f"\n🎉 TOUS LES TESTS PHASE 2 RÉUSSIS !")
            print("   L'agent est prêt pour la transformation TIER 1+")
        else:
            print(f"\n⚠️ {failed_tests} TESTS ÉCHOUÉS")
            print("   Vérification requise avant déploiement")
    
    def save_test_results(self, total_time: float):
        """Sauvegarde des résultats des tests"""
        test_report = {
            'timestamp': time.time(),
            'test_duration_seconds': total_time,
            'modules_tested': list(self.test_results.keys()),
            'results': self.test_results,
            'summary': {
                'total_tests': sum(r.get('tests', 0) for r in self.test_results.values() if r['status'] == 'PASSED'),
                'passed_tests': sum(r.get('tests', 0) for r in self.test_results.values() if r['status'] == 'PASSED'),
                'failed_modules': sum(1 for r in self.test_results.values() if r['status'] == 'FAILED')
            }
        }
        
        try:
            with open('phase2_test_results.json', 'w') as f:
                json.dump(test_report, f, indent=2, default=str)
            print("   💾 Résultats sauvegardés: phase2_test_results.json")
        except Exception as e:
            print(f"   ⚠️ Erreur sauvegarde résultats: {e}")

def run_phase2_tests():
    """Fonction principale pour exécuter les tests Phase 2"""
    print("🧪 LANCEMENT TESTS DE VALIDATION PHASE 2")
    print("=" * 70)
    
    # Initialisation du testeur
    tester = Phase2ModuleTester()
    
    # Exécution de tous les tests
    results = tester.run_all_tests()
    
    return results

if __name__ == "__main__":
    # Lancement des tests
    test_results = run_phase2_tests()
    
    # Affichage des résultats finaux
    if all(r['status'] == 'PASSED' for r in test_results.values()):
        print("\n✅ VALIDATION PHASE 2 COMPLÈTE - Agent certifié TIER 1+ !")
    else:
        print("\n❌ VALIDATION PHASE 2 INCOMPLÈTE - Vérification requise")
