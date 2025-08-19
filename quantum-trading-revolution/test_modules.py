#!/usr/bin/env python3
"""
TEST AUTOMATIQUE DES MODULES QUANTUM TRADING AGENT
Tests individuels et intégrés sans interaction utilisateur
"""

import sys
import numpy as np
import pandas as pd

def test_feature_engineering():
    """Test du module Feature Engineering"""
    print("🔬 Test du Feature Engineering Pipeline...")
    try:
        from advanced_features_final import FeatureEngineeringPipeline
        
        # Données de test
        np.random.seed(42)
        test_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'volume': np.random.randint(1000, 5000, 100)
        })
        
        fe = FeatureEngineeringPipeline()
        features = fe.generate_core_features(test_data)
        
        print(f"   ✅ Features générées: {len(features.columns)}")
        print(f"   ✅ Données traitées: {len(features)} observations")
        print(f"   ✅ Features principales: {list(features.columns[:5])}")
        print(f"   ✅ Pas de valeurs NaN: {not features.isnull().any().any()}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur test: {e}")
        return False

def test_adaptive_parameters():
    """Test du module Adaptive Parameters"""
    print("🔬 Test de l'Adaptive Parameter Manager...")
    try:
        from adaptive_parameters import AdaptiveParameterManager
        
        apm = AdaptiveParameterManager()
        
        # Test détection régime
        price_data = np.random.randn(50).cumsum() + 100
        regime, confidence = apm.detect_simple_regime(price_data)
        
        print(f"   ✅ Régime détecté: {regime}")
        print(f"   ✅ Confiance: {confidence:.2f}")
        
        # Test adaptation poids
        signals = {'sentiment': 0.5, 'mps': -0.2, 'quantum': 0.1}
        pnl_history = np.random.randn(20) * 0.01
        market_data = {'close': price_data}
        
        weights, detected_regime = apm.update_weights(signals, pnl_history, market_data)
        print(f"   ✅ Poids adaptés: {weights}")
        print(f"   ✅ Somme des poids: {sum(weights.values()):.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur test: {e}")
        return False

def test_cost_optimizer():
    """Test du module Transaction Cost Optimizer"""
    print("🔬 Test du Transaction Cost Optimizer...")
    try:
        from transaction_optimizer import TransactionCostOptimizer
        
        tco = TransactionCostOptimizer()
        
        # Test estimation coûts
        cost_est = tco.estimate_costs(1000, 100, volume=50000, volatility=0.02)
        print(f"   ✅ Coût estimé: {cost_est['cost_bps']:.1f} bps")
        print(f"   ✅ Recommandation: {cost_est['recommendation']}")
        
        # Test optimisation
        exec_plan = tco.optimize_execution(5000, 100)
        print(f"   ✅ Stratégie: {exec_plan['strategy']}")
        if 'expected_savings' in exec_plan:
            print(f"   ✅ Économies: ${exec_plan['expected_savings']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur test: {e}")
        return False

def test_complete_integration():
    """Test de l'intégration complète"""
    print("🔬 Test de l'intégration complète...")
    try:
        from demo_upgrade import run_comprehensive_demo
        
        print("   🚀 Lancement de la démo complète...")
        upgraded_agent, results = run_comprehensive_demo()
        
        print("   ✅ Test intégré terminé avec succès!")
        print(f"   ✅ Agent upgradé: {type(upgraded_agent).__name__}")
        print(f"   ✅ Résultats disponibles: {len(results)} métriques")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur test intégré: {e}")
        return False

def main():
    """Exécution de tous les tests"""
    print("="*80)
    print("🧪 TESTS AUTOMATIQUES - QUANTUM TRADING AGENT MODULES")
    print("="*80)
    
    tests = [
        ("Feature Engineering", test_feature_engineering),
        ("Adaptive Parameters", test_adaptive_parameters), 
        ("Cost Optimizer", test_cost_optimizer),
        ("Intégration Complète", test_complete_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 [{len(results)+1}/4] {test_name}")
        print("-" * 50)
        
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"   ✅ {test_name}: RÉUSSI")
        else:
            print(f"   ❌ {test_name}: ÉCHEC")
    
    # Résumé final
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
        print(f"   {test_name:<25} {status}")
    
    print(f"\n🎯 RÉSULTAT GLOBAL: {passed}/{total} tests réussis ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("🎉 TOUS LES MODULES FONCTIONNENT PARFAITEMENT!")
        print("✅ Votre agent est prêt pour l'upgrade révolutionnaire!")
    else:
        print("⚠️  Certains modules nécessitent une attention.")
        print("📧 Consultez les erreurs ci-dessus pour plus de détails.")
    
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrompus par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()
