#!/usr/bin/env python3
"""
🧪 TEST D'INTÉGRATION : SYSTÈME DE RÉCUPÉRATION D'ERREURS
🎯 Valider l'intégration complète du gestionnaire de récupération avec l'agent principal
"""

import sys
import os
import unittest
import logging
from unittest.mock import Mock, patch

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestErrorRecoveryIntegration(unittest.TestCase):
    """Test d'intégration du système de récupération d'erreurs"""
    
    def setUp(self):
        """Configuration initiale des tests"""
        logger.info("🧪 Configuration test d'intégration récupération d'erreurs")
        
        # Mock des modules externes si nécessaire
        self.mock_modules = {}
        
    def test_import_error_recovery_manager(self):
        """Test import du gestionnaire de récupération"""
        try:
            from error_recovery import ErrorRecoveryManager
            logger.info("✅ Import ErrorRecoveryManager réussi")
            self.assertTrue(True)
        except ImportError as e:
            logger.error(f"❌ Import ErrorRecoveryManager échoué: {e}")
            self.fail(f"Import échoué: {e}")
    
    def test_import_trading_exceptions(self):
        """Test import des exceptions de trading"""
        try:
            from trading_exceptions import (
                TradingError, InsufficientCapitalError, StaleMarketDataError,
                ExcessiveRiskError, MemoryDecodingError, MemoryCapacityError
            )
            logger.info("✅ Import exceptions trading réussi")
            self.assertTrue(True)
        except ImportError as e:
            logger.error(f"❌ Import exceptions trading échoué: {e}")
            self.fail(f"Import échoué: {e}")
    
    def test_error_recovery_manager_creation(self):
        """Test création d'une instance du gestionnaire de récupération"""
        try:
            from error_recovery import ErrorRecoveryManager
            
            # Créer une config mock
            mock_config = Mock()
            mock_config.memory_size = 1000
            
            # Créer l'instance
            recovery_manager = ErrorRecoveryManager(mock_config)
            
            # Vérifier les attributs
            self.assertIsNotNone(recovery_manager)
            self.assertEqual(recovery_manager.max_recovery_attempts, 3)
            self.assertIsInstance(recovery_manager.recovery_attempts, dict)
            self.assertIsInstance(recovery_manager.recovery_success_rate, dict)
            self.assertIsInstance(recovery_manager.recovery_history, list)
            
            logger.info("✅ Création ErrorRecoveryManager réussie")
            
        except Exception as e:
            logger.error(f"❌ Création ErrorRecoveryManager échouée: {e}")
            self.fail(f"Création échouée: {e}")
    
    def test_insufficient_capital_recovery_strategy(self):
        """Test stratégie de récupération pour capital insuffisant"""
        try:
            from error_recovery import ErrorRecoveryManager
            from trading_exceptions import InsufficientCapitalError
            
            # Créer le gestionnaire
            mock_config = Mock()
            recovery_manager = ErrorRecoveryManager(mock_config)
            
            # Créer une erreur de capital insuffisant
            capital_error = InsufficientCapitalError(
                required=2000.0,
                available=1000.0,
                symbol="AAPL",
                action="BUY"
            )
            
            # Obtenir stratégie de récupération
            recovery_plan = recovery_manager.handle_insufficient_capital(capital_error)
            
            # Vérifier la structure
            self.assertIn('error_type', recovery_plan)
            self.assertIn('recovery_strategies', recovery_plan)
            self.assertIn('recommended_strategy', recovery_plan)
            self.assertEqual(recovery_plan['error_type'], 'INSUFFICIENT_CAPITAL')
            self.assertGreater(len(recovery_plan['recovery_strategies']), 0)
            
            logger.info("✅ Stratégie récupération capital insuffisant validée")
            
        except Exception as e:
            logger.error(f"❌ Test stratégie capital insuffisant échoué: {e}")
            self.fail(f"Test échoué: {e}")
    
    def test_stale_market_data_recovery_strategy(self):
        """Test stratégie de récupération pour données obsolètes"""
        try:
            from error_recovery import ErrorRecoveryManager
            from trading_exceptions import StaleMarketDataError
            
            # Créer le gestionnaire
            mock_config = Mock()
            recovery_manager = ErrorRecoveryManager(mock_config)
            
            # Créer une erreur de données obsolètes
            stale_error = StaleMarketDataError(
                data_age_seconds=600,  # 10 minutes
                max_age_seconds=300,   # 5 minutes
                symbol="AAPL"
            )
            
            # Obtenir stratégie de récupération
            recovery_plan = recovery_manager.handle_stale_market_data(stale_error)
            
            # Vérifier la structure
            self.assertIn('error_type', recovery_plan)
            self.assertIn('recovery_strategies', recovery_plan)
            self.assertEqual(recovery_plan['error_type'], 'STALE_MARKET_DATA')
            self.assertGreater(len(recovery_plan['recovery_strategies']), 0)
            
            logger.info("✅ Stratégie récupération données obsolètes validée")
            
        except Exception as e:
            logger.error(f"❌ Test stratégie données obsolètes échoué: {e}")
            self.fail(f"Test échoué: {e}")
    
    def test_excessive_risk_recovery_strategy(self):
        """Test stratégie de récupération pour risque excessif"""
        try:
            from error_recovery import ErrorRecoveryManager
            from trading_exceptions import ExcessiveRiskError
            
            # Créer le gestionnaire
            mock_config = Mock()
            recovery_manager = ErrorRecoveryManager(mock_config)
            
            # Créer une erreur de risque excessif
            risk_error = ExcessiveRiskError(
                risk_score=0.85,
                max_allowed=0.5,
                risk_factors=["volatilité élevée", "concentration excessive"],
                portfolio_value=50000.0
            )
            
            # Obtenir stratégie de récupération
            recovery_plan = recovery_manager.handle_excessive_risk(risk_error)
            
            # Vérifier la structure
            self.assertIn('error_type', recovery_plan)
            self.assertIn('recovery_strategies', recovery_plan)
            self.assertEqual(recovery_plan['error_type'], 'EXCESSIVE_RISK')
            self.assertGreater(len(recovery_plan['recovery_strategies']), 0)
            
            logger.info("✅ Stratégie récupération risque excessif validée")
            
        except Exception as e:
            logger.error(f"❌ Test stratégie risque excessif échoué: {e}")
            self.fail(f"Test échoué: {e}")
    
    def test_memory_decoding_recovery_strategy(self):
        """Test stratégie de récupération pour erreur de décodage mémoire"""
        try:
            from error_recovery import ErrorRecoveryManager
            from trading_exceptions import MemoryDecodingError
            
            # Créer le gestionnaire
            mock_config = Mock()
            recovery_manager = ErrorRecoveryManager(mock_config)
            
            # Créer une erreur de décodage mémoire
            memory_error = MemoryDecodingError(
                operation="context_encoding",
                memory_size=500,
                error_details="Invalid context format",
                input_data={"price": 100.0, "rsi": 50.0}
            )
            
            # Obtenir stratégie de récupération
            recovery_plan = recovery_manager.handle_memory_decoding_error(memory_error)
            
            # Vérifier la structure
            self.assertIn('error_type', recovery_plan)
            self.assertIn('recovery_strategies', recovery_plan)
            self.assertEqual(recovery_plan['error_type'], 'MEMORY_DECODING_ERROR')
            self.assertGreater(len(recovery_plan['recovery_strategies']), 0)
            
            logger.info("✅ Stratégie récupération décodage mémoire validée")
            
        except Exception as e:
            logger.error(f"❌ Test stratégie décodage mémoire échoué: {e}")
            self.fail(f"Test échoué: {e}")
    
    def test_memory_capacity_recovery_strategy(self):
        """Test stratégie de récupération pour erreur de capacité mémoire"""
        try:
            from error_recovery import ErrorRecoveryManager
            from trading_exceptions import MemoryCapacityError
            
            # Créer le gestionnaire
            mock_config = Mock()
            recovery_manager = ErrorRecoveryManager(mock_config)
            
            # Créer une erreur de capacité mémoire
            capacity_error = MemoryCapacityError(
                current_size=950,
                max_capacity=1000,
                operation="store_experience"
            )
            
            # Obtenir stratégie de récupération
            recovery_plan = recovery_manager.handle_memory_capacity_error(capacity_error)
            
            # Vérifier la structure
            self.assertIn('error_type', recovery_plan)
            self.assertIn('recovery_strategies', recovery_plan)
            self.assertEqual(recovery_plan['error_type'], 'MEMORY_CAPACITY_ERROR')
            self.assertGreater(len(recovery_plan['recovery_strategies']), 0)
            
            logger.info("✅ Stratégie récupération capacité mémoire validée")
            
        except Exception as e:
            logger.error(f"❌ Test stratégie capacité mémoire échoué: {e}")
            self.fail(f"Test échoué: {e}")
    
    def test_generic_error_recovery_strategy(self):
        """Test stratégie de récupération pour erreur générique"""
        try:
            from error_recovery import ErrorRecoveryManager
            from trading_exceptions import TradingError
            
            # Créer le gestionnaire
            mock_config = Mock()
            recovery_manager = ErrorRecoveryManager(mock_config)
            
            # Créer une erreur générique
            generic_error = TradingError(
                message="Erreur système inattendue",
                error_code="SYSTEM_ERROR",
                context={"component": "fusion_layer"},
                severity="ERROR"
            )
            
            # Obtenir stratégie de récupération
            recovery_plan = recovery_manager.handle_generic_error(generic_error)
            
            # Vérifier la structure
            self.assertIn('error_type', recovery_plan)
            self.assertIn('recovery_strategies', recovery_plan)
            self.assertEqual(recovery_plan['error_type'], 'GENERIC_ERROR')
            self.assertGreater(len(recovery_plan['recovery_strategies']), 0)
            
            logger.info("✅ Stratégie récupération erreur générique validée")
            
        except Exception as e:
            logger.error(f"❌ Test stratégie erreur générique échoué: {e}")
            self.fail(f"Test échoué: {e}")
    
    def test_recovery_strategy_execution(self):
        """Test exécution d'une stratégie de récupération"""
        try:
            from error_recovery import ErrorRecoveryManager
            from trading_exceptions import TradingError
            
            # Créer le gestionnaire
            mock_config = Mock()
            recovery_manager = ErrorRecoveryManager(mock_config)
            
            # Créer une erreur
            error = TradingError("Test error")
            
            # Créer une stratégie
            strategy = {
                'strategy': 'test_strategy',
                'action': 'Test action',
                'confidence': 0.8,
                'risk_level': 'LOW'
            }
            
            # Exécuter la stratégie
            result = recovery_manager.execute_recovery_strategy(error, strategy)
            
            # Vérifier la structure du résultat
            self.assertIn('strategy_executed', result)
            self.assertIn('success', result)
            self.assertIn('recovery_time_seconds', result)
            self.assertEqual(result['strategy_executed'], 'test_strategy')
            
            logger.info("✅ Exécution stratégie de récupération validée")
            
        except Exception as e:
            logger.error(f"❌ Test exécution stratégie échoué: {e}")
            self.fail(f"Test échoué: {e}")
    
    def test_recovery_statistics(self):
        """Test obtention des statistiques de récupération"""
        try:
            from error_recovery import ErrorRecoveryManager
            from trading_exceptions import TradingError
            
            # Créer le gestionnaire
            mock_config = Mock()
            recovery_manager = ErrorRecoveryManager(mock_config)
            
            # Exécuter quelques stratégies pour générer des statistiques
            error = TradingError("Test error")
            strategy = {'strategy': 'test_strategy', 'action': 'Test', 'confidence': 0.8, 'risk_level': 'LOW'}
            
            recovery_manager.execute_recovery_strategy(error, strategy)
            
            # Obtenir les statistiques
            stats = recovery_manager.get_recovery_statistics()
            
            # Vérifier la structure
            self.assertIn('total_recovery_attempts', stats)
            self.assertIn('successful_recoveries', stats)
            self.assertIn('overall_success_rate', stats)
            self.assertIn('strategy_success_rates', stats)
            self.assertIn('recent_recoveries', stats)
            
            # Vérifier les valeurs
            self.assertGreaterEqual(stats['total_recovery_attempts'], 1)
            self.assertGreaterEqual(stats['overall_success_rate'], 0.0)
            self.assertLessEqual(stats['overall_success_rate'], 1.0)
            
            logger.info("✅ Statistiques de récupération validées")
            
        except Exception as e:
            logger.error(f"❌ Test statistiques récupération échoué: {e}")
            self.fail(f"Test échoué: {e}")
    
    def test_recommended_strategy_selection(self):
        """Test sélection automatique de la stratégie recommandée"""
        try:
            from error_recovery import ErrorRecoveryManager
            from trading_exceptions import InsufficientCapitalError
            
            # Créer le gestionnaire
            mock_config = Mock()
            recovery_manager = ErrorRecoveryManager(mock_config)
            
            # Créer une erreur
            error = InsufficientCapitalError(
                required=2000.0,
                available=1000.0,
                symbol="AAPL",
                action="BUY"
            )
            
            # Obtenir la stratégie recommandée
            recommended = recovery_manager.get_recommended_strategy(error)
            
            # Vérifier que c'est la bonne stratégie
            self.assertEqual(recommended['error_type'], 'INSUFFICIENT_CAPITAL')
            self.assertIn('recommended_strategy', recommended)
            
            logger.info("✅ Sélection automatique stratégie recommandée validée")
            
        except Exception as e:
            logger.error(f"❌ Test sélection stratégie recommandée échoué: {e}")
            self.fail(f"Test échoué: {e}")

def run_integration_tests():
    """Exécuter tous les tests d'intégration"""
    logger.info("🚀 DÉMARRAGE TESTS D'INTÉGRATION - SYSTÈME DE RÉCUPÉRATION D'ERREURS")
    
    # Créer la suite de tests
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestErrorRecoveryIntegration)
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Résumé des résultats
    logger.info("="*80)
    logger.info("📊 RÉSULTATS TESTS D'INTÉGRATION")
    logger.info("="*80)
    logger.info(f"Tests exécutés: {result.testsRun}")
    logger.info(f"Échecs: {len(result.failures)}")
    logger.info(f"Erreurs: {len(result.errors)}")
    
    if result.failures:
        logger.error("❌ ÉCHECS DÉTECTÉS:")
        for test, traceback in result.failures:
            logger.error(f"  - {test}: {traceback}")
    
    if result.errors:
        logger.error("❌ ERREURS DÉTECTÉES:")
        for test, traceback in result.errors:
            logger.error(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        logger.info("✅ TOUS LES TESTS D'INTÉGRATION ONT RÉUSSI!")
        logger.info("🎯 Le système de récupération d'erreurs est parfaitement intégré!")
    else:
        logger.error("❌ CERTAINS TESTS D'INTÉGRATION ONT ÉCHOUÉ!")
        logger.error("🔧 Vérifiez les erreurs ci-dessus et corrigez-les")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Exécuter les tests
    success = run_integration_tests()
    
    # Code de sortie approprié
    sys.exit(0 if success else 1)
