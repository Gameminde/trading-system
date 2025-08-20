"""
Tests complets pour les validateurs et constantes de trading
Validation de niveau production institutionnel
"""

import unittest
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from validators import TradingValidators
from trading_constants import TradingConstants, TradingActions, MarketRegimes


class TestTradingValidators(unittest.TestCase):
    """Tests complets pour TradingValidators"""
    
    def test_validate_trade_action_valid(self):
        """Test validation actions valides"""
        valid_actions = ['BUY', 'SELL', 'HOLD']
        for action in valid_actions:
            try:
                TradingValidators.validate_trade_action(action)
                self.assertTrue(True)  # Pas d'exception
            except ValueError:
                self.fail(f"Action valide {action} a levé une exception")
    
    def test_validate_trade_action_invalid(self):
        """Test validation actions invalides"""
        invalid_actions = ['', 'INVALID', 'buy', 'sell', None, 123]
        for action in invalid_actions:
            with self.assertRaises(ValueError):
                TradingValidators.validate_trade_action(action)
    
    def test_validate_quantity_valid(self):
        """Test validation quantités valides"""
        valid_quantities = [1, 100, 1000.5, 0.1]
        for qty in valid_quantities:
            try:
                TradingValidators.validate_quantity(qty)
                self.assertTrue(True)
            except ValueError:
                self.fail(f"Quantité valide {qty} a levé une exception")
    
    def test_validate_quantity_invalid(self):
        """Test validation quantités invalides"""
        invalid_quantities = [0, -1, -100.5, None, "100", []]
        for qty in invalid_quantities:
            with self.assertRaises(ValueError):
                TradingValidators.validate_quantity(qty)
    
    def test_validate_price_valid(self):
        """Test validation prix valides"""
        valid_prices = [0.01, 100, 1000.50, 50000]
        for price in valid_prices:
            try:
                TradingValidators.validate_price(price)
                self.assertTrue(True)
            except ValueError:
                self.fail(f"Prix valide {price} a levé une exception")
    
    def test_validate_price_invalid(self):
        """Test validation prix invalides"""
        invalid_prices = [0, -1, -100.50, None, "100", [], 1000001]  # Dépasse le seuil
        for price in invalid_prices:
            with self.assertRaises(ValueError):
                TradingValidators.validate_price(price)
    
    def test_validate_symbol_valid(self):
        """Test validation symboles valides"""
        valid_symbols = ['AAPL', 'BTC', 'ETH', 'TSLA', 'GOOGL']
        for symbol in valid_symbols:
            try:
                TradingValidators.validate_symbol(symbol)
                self.assertTrue(True)
            except ValueError:
                self.fail(f"Symbole valide {symbol} a levé une exception")
    
    def test_validate_symbol_invalid(self):
        """Test validation symboles invalides"""
        invalid_symbols = ['', 'TOOLONGSYMBOL', 'AAPL@', 'BTC-', None, 123, []]
        for symbol in invalid_symbols:
            with self.assertRaises(ValueError):
                TradingValidators.validate_symbol(symbol)
    
    def test_validate_capital_valid(self):
        """Test validation capital valide"""
        valid_capitals = [100, 1000, 10000.50, 5000000]
        for capital in valid_capitals:
            try:
                TradingValidators.validate_capital(capital)
                self.assertTrue(True)
            except ValueError:
                self.fail(f"Capital valide {capital} a levé une exception")
    
    def test_validate_capital_invalid(self):
        """Test validation capital invalide"""
        invalid_capitals = [0, -1, -1000, None, "1000", [], 10000001]  # Dépasse le seuil
        for capital in invalid_capitals:
            with self.assertRaises(ValueError):
                TradingValidators.validate_capital(capital)
    
    def test_validate_context_valid(self):
        """Test validation contexte valide"""
        valid_context = {
            'timestamp': '2024-01-01T00:00:00',
            'price': 100.0,
            'symbols': ['AAPL']
        }
        try:
            TradingValidators.validate_context(valid_context)
            self.assertTrue(True)
        except ValueError:
            self.fail("Contexte valide a levé une exception")
    
    def test_validate_context_invalid(self):
        """Test validation contexte invalide"""
        invalid_contexts = [
            None,
            {},
            {'timestamp': '2024-01-01'},
            {'price': 100, 'symbols': ['AAPL']},
            {'timestamp': '2024-01-01', 'price': 100}
        ]
        for context in invalid_contexts:
            with self.assertRaises(ValueError):
                TradingValidators.validate_context(context)
    
    def test_validate_confidence_valid(self):
        """Test validation confiance valide"""
        valid_confidences = [0.0, 0.5, 1.0, 0.123]
        for conf in valid_confidences:
            try:
                TradingValidators.validate_confidence(conf)
                self.assertTrue(True)
            except ValueError:
                self.fail(f"Confiance valide {conf} a levé une exception")
    
    def test_validate_confidence_invalid(self):
        """Test validation confiance invalide"""
        invalid_confidences = [-0.1, 1.1, None, "0.5", [], -1, 2]
        for conf in invalid_confidences:
            with self.assertRaises(ValueError):
                TradingValidators.validate_confidence(conf)
    
    def test_validate_risk_score_valid(self):
        """Test validation score de risque valide"""
        valid_risk_scores = [0.0, 0.5, 1.0, 0.789]
        for risk in valid_risk_scores:
            try:
                TradingValidators.validate_risk_score(risk)
                self.assertTrue(True)
            except ValueError:
                self.fail(f"Score de risque valide {risk} a levé une exception")
    
    def test_validate_risk_score_invalid(self):
        """Test validation score de risque invalide"""
        invalid_risk_scores = [-0.1, 1.1, None, "0.5", [], -1, 2]
        for risk in invalid_risk_scores:
            with self.assertRaises(ValueError):
                TradingValidators.validate_risk_score(risk)
    
    def test_validate_trade_parameters_complete(self):
        """Test validation complète des paramètres de trade"""
        valid_params = {
            'action': 'BUY',
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.0,
            'context': {
                'timestamp': '2024-01-01T00:00:00',
                'price': 150.0,
                'symbols': ['AAPL']
            }
        }
        
        try:
            TradingValidators.validate_trade_parameters(
                valid_params['action'],
                valid_params['symbol'],
                valid_params['quantity'],
                valid_params['price'],
                valid_params['context']
            )
            self.assertTrue(True)
        except ValueError:
            self.fail("Validation complète des paramètres a échoué")
    
    def test_validate_portfolio_limits_valid(self):
        """Test validation limites portfolio valides"""
        valid_positions = [
            {'symbol': 'AAPL', 'quantity': 100, 'entry_price': 150.0},  # 15,000
            {'symbol': 'GOOGL', 'quantity': 5, 'entry_price': 2800.0}   # 14,000
        ]
        
        try:
            TradingValidators.validate_portfolio_limits(valid_positions, max_concentration=0.6)
            self.assertTrue(True)
        except ValueError:
            self.fail("Validation portfolio valide a échoué")
    
    def test_validate_portfolio_limits_invalid(self):
        """Test validation limites portfolio invalides"""
        invalid_positions = [
            {'symbol': 'AAPL', 'quantity': 1000, 'entry_price': 150.0},  # 100% concentration
        ]
        
        with self.assertRaises(ValueError):
            TradingValidators.validate_portfolio_limits(invalid_positions, max_concentration=0.5)


class TestTradingConstants(unittest.TestCase):
    """Tests pour les constantes de trading"""
    
    def test_rsi_thresholds(self):
        """Test seuils RSI"""
        self.assertEqual(TradingConstants.RSI_OVERSOLD, 30)
        self.assertEqual(TradingConstants.RSI_OVERBOUGHT, 70)
        self.assertEqual(TradingConstants.RSI_NEUTRAL, 50)
    
    def test_risk_thresholds(self):
        """Test seuils de risque"""
        self.assertEqual(TradingConstants.RISK_CRITICAL_THRESHOLD, 0.8)
        self.assertEqual(TradingConstants.RISK_HIGH_THRESHOLD, 0.6)
        self.assertEqual(TradingConstants.RISK_MODERATE_THRESHOLD, 0.4)
        self.assertEqual(TradingConstants.RISK_LOW_THRESHOLD, 0.2)
    
    def test_confidence_thresholds(self):
        """Test seuils de confiance"""
        self.assertEqual(TradingConstants.MIN_CONFIDENCE_THRESHOLD, 0.08)
        self.assertEqual(TradingConstants.HIGH_CONFIDENCE_THRESHOLD, 0.8)
        self.assertEqual(TradingConstants.EXTREME_CONFIDENCE_THRESHOLD, 0.95)
    
    def test_portfolio_limits(self):
        """Test limites portfolio"""
        self.assertEqual(TradingConstants.MAX_POSITION_CONCENTRATION, 0.5)
        self.assertEqual(TradingConstants.MAX_DAILY_LOSS_RATIO, 0.05)
        self.assertEqual(TradingConstants.MAX_DRAWDOWN_LIMIT, 0.15)
        self.assertEqual(TradingConstants.MAX_POSITIONS_COUNT, 4)
    
    def test_quantum_scoring(self):
        """Test scores quantiques"""
        self.assertEqual(TradingConstants.QUANTUM_HIGH_SCORE, 0.6)
        self.assertEqual(TradingConstants.QUANTUM_LOW_SCORE, 0.4)
        self.assertEqual(TradingConstants.QUANTUM_BOOST_FACTOR, 0.8)
    
    def test_price_change_thresholds(self):
        """Test seuils changement prix"""
        self.assertEqual(TradingConstants.SIGNIFICANT_PRICE_CHANGE, 0.01)
        self.assertEqual(TradingConstants.MAJOR_PRICE_CHANGE, 0.02)
        self.assertEqual(TradingConstants.EXTREME_PRICE_CHANGE, 0.05)
    
    def test_trading_actions(self):
        """Test actions de trading"""
        self.assertIn('BUY', TradingActions.VALID_ACTIONS)
        self.assertIn('SELL', TradingActions.VALID_ACTIONS)
        self.assertIn('HOLD', TradingActions.VALID_ACTIONS)
        self.assertEqual(len(TradingActions.VALID_ACTIONS), 5)
    
    def test_market_regimes(self):
        """Test régimes de marché"""
        self.assertIn('BULL', MarketRegimes.VALID_REGIMES)
        self.assertIn('BEAR', MarketRegimes.VALID_REGIMES)
        self.assertIn('SIDEWAYS', MarketRegimes.VALID_REGIMES)
        self.assertEqual(len(MarketRegimes.VALID_REGIMES), 5)


def run_comprehensive_tests():
    """Exécuter tous les tests"""
    print("🧪 DÉMARRAGE TESTS COMPLETS VALIDATEURS + CONSTANTES")
    print("=" * 60)
    
    # Créer suite de tests
    test_suite = unittest.TestSuite()
    
    # Ajouter tests validateurs
    test_suite.addTest(unittest.makeSuite(TestTradingValidators))
    
    # Ajouter tests constantes
    test_suite.addTest(unittest.makeSuite(TestTradingConstants))
    
    # Exécuter tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS TESTS COMPLETS")
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ TOUS LES TESTS ONT RÉUSSI!")
        print("🎯 VALIDATEURS ET CONSTANTES OPÉRATIONNELS")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        if result.failures:
            print("\nÉchecs:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        if result.errors:
            print("\nErreurs:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
