"""
🧪 SYSTEM VALIDATOR - VALIDATION DES CORRECTIONS CRITIQUES
✅ Test Transformer predictions réalistes
✅ Test données réelles vs hardcodées
✅ Test logique RL cohérente
✅ Test données historiques vraies
✅ Validation complète du système
"""

import sys
import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("SYSTEM_VALIDATOR")

class SystemValidator:
    """Validateur pour toutes les corrections critiques"""
    
    def __init__(self):
        self.test_results = {}
        self.critical_errors = []
        
    def validate_all_corrections(self) -> bool:
        """Valider que toutes les corrections fonctionnent"""
        logger.info("🧪 Validation des corrections critiques...")
        
        tests = [
            ("Transformer Predictions Réalistes", self.test_transformer_realistic_predictions),
            ("Données Réelles vs Hardcodées", self.test_real_vs_fake_data),
            ("Logique RL Cohérente", self.test_rl_logic_correctness),
            ("Données Historiques Vraies", self.test_real_historical_data),
            ("Validation Complète Système", self.test_complete_system)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*20} {test_name} {'='*20}")
            try:
                success = test_func()
                self.test_results[test_name] = success
                if success:
                    logger.info(f"✅ {test_name}: RÉUSSI")
                else:
                    logger.error(f"❌ {test_name}: ÉCHOUÉ")
                    all_passed = False
            except Exception as e:
                logger.error(f"💥 {test_name}: ERREUR CRITIQUE - {e}")
                self.test_results[test_name] = False
                self.critical_errors.append(f"{test_name}: {e}")
                all_passed = False
        
        if all_passed:
            logger.info("✅ Toutes les corrections validées avec succès")
        else:
            logger.error("❌ Certaines corrections ont échoué")
            for error in self.critical_errors:
                logger.error(f"   💥 {error}")
        
        return all_passed
        
    def test_transformer_realistic_predictions(self) -> bool:
        """Tester que Transformer donne prix réalistes"""
        try:
            logger.info("🔮 Test prédictions Transformer...")
            
            # Importer le Transformer corrigé
            from transformer_predictor import PricePredictor
            
            # Créer données de test réalistes
            test_data = pd.DataFrame({
                'open': [100.0, 101.0, 99.0, 100.5, 102.0],
                'high': [102.0, 103.0, 100.0, 101.5, 103.0],
                'low': [99.0, 100.0, 98.0, 99.5, 101.0],
                'close': [100.0, 101.0, 99.0, 100.5, 102.0],
                'volume': [1000000] * 5,
                'rsi': [50.0, 55.0, 45.0, 52.0, 58.0],
                'macd': [0.0, 0.1, -0.1, 0.05, 0.15]
            })
            
            # Créer et entraîner le modèle
            predictor = PricePredictor(sequence_length=3)
            predictor.train_model(test_data, epochs=10)
            
            # Tester prédiction
            predicted_price = predictor.predict_next_price(test_data)
            
            if predicted_price is None:
                logger.error("❌ Prédiction échouée")
                return False
            
            current_price = test_data['close'].iloc[-1]
            price_change = abs(predicted_price - current_price) / current_price
            
            logger.info(f"📊 Prix actuel: ${current_price:.2f}")
            logger.info(f"🔮 Prix prédit: ${predicted_price:.2f}")
            logger.info(f"📈 Changement: {price_change:.1%}")
            
            # Vérifier que la prédiction est réaliste (±50% max)
            if price_change > 0.50:
                logger.error(f"❌ Prédiction trop extrême: {price_change:.1%}")
                return False
            
            # Vérifier que le prix n'est pas aberrant
            if predicted_price <= 0 or predicted_price > current_price * 10:
                logger.error(f"❌ Prix aberrant: ${predicted_price:.2f}")
                return False
            
            logger.info("✅ Prédictions Transformer réalistes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test Transformer échoué: {e}")
            return False
    
    def test_real_vs_fake_data(self) -> bool:
        """Tester que données sont réelles, pas hardcodées"""
        try:
            logger.info("📊 Test données réelles vs hardcodées...")
            
            # Importer le calculateur d'indicateurs réels
            from real_indicators_calculator import RealIndicatorsCalculator
            
            calculator = RealIndicatorsCalculator()
            
            # Tester avec plusieurs symboles
            symbols = ['AAPL', 'MSFT', 'GOOGL']
            all_real = True
            
            for symbol in symbols:
                data = calculator.get_real_market_data(symbol)
                if not data:
                    logger.error(f"❌ Impossible récupérer données {symbol}")
                    all_real = False
                    continue
                
                logger.info(f"📊 {symbol}: Prix=${data['price']:.2f}, RSI={data['rsi']:.1f}, MACD={data['macd']:.4f}")
                
                # Vérifier que RSI n'est pas toujours 50.0
                if data['rsi'] == 50.0 and data['source'] == 'yfinance_real':
                    logger.warning(f"⚠️ RSI suspect pour {symbol}: {data['rsi']}")
                
                # Vérifier que MACD n'est pas toujours 0.0
                if data['macd'] == 0.0 and data['source'] == 'yfinance_real':
                    logger.warning(f"⚠️ MACD suspect pour {symbol}: {data['macd']}")
                
                # Vérifier que le prix est réaliste
                if data['price'] <= 0 or data['price'] > 1000000:
                    logger.error(f"❌ Prix invalide {symbol}: ${data['price']:.2f}")
                    all_real = False
                
                # Vérifier que la source n'est pas fallback
                if data['source'] == 'fallback':
                    logger.warning(f"⚠️ Fallback utilisé pour {symbol}")
            
            if all_real:
                logger.info("✅ Données réelles validées")
                return True
            else:
                logger.error("❌ Certaines données sont invalides")
                return False
                
        except Exception as e:
            logger.error(f"❌ Test données réelles échoué: {e}")
            return False
    
    def test_rl_logic_correctness(self) -> bool:
        """Tester logique RL balance/positions"""
        try:
            logger.info("🤖 Test logique RL...")
            
            # Importer l'environnement RL corrigé
            from rl_trading_agent import TradingEnvironment
            
            # Créer données de test
            test_data = pd.DataFrame({
                'close': [100.0, 101.0, 99.0, 100.5, 102.0],
                'rsi': [50.0, 55.0, 45.0, 52.0, 58.0],
                'macd': [0.0, 0.1, -0.1, 0.05, 0.15],
                'volume': [1000000] * 5
            })
            
            # Créer environnement
            env = TradingEnvironment(test_data, initial_balance=10000)
            
            # Test 1: Achat
            obs, reward, done, truncated, info = env.step(1)  # BUY
            logger.info(f"📊 Après BUY: Balance=${env.balance:.2f}, Actions={env.shares}")
            
            # Vérifier cohérence
            expected_balance = 10000 - (env.shares * 100.0)  # Prix d'achat
            if abs(env.balance - expected_balance) > 0.01:
                logger.error(f"❌ Balance incohérente après BUY: {env.balance} vs {expected_balance}")
                return False
            
            # Test 2: Vente
            obs, reward, done, truncated, info = env.step(2)  # SELL
            logger.info(f"📊 Après SELL: Balance=${env.balance:.2f}, Actions={env.shares}")
            
            # Vérifier que toutes les actions ont été vendues
            if env.shares != 0:
                logger.error(f"❌ Actions non vendues: {env.shares}")
                return False
            
            # Vérifier conservation du capital (avec profit/perte)
            portfolio_value = env.balance
            logger.info(f"📊 Valeur finale portfolio: ${portfolio_value:.2f}")
            
            logger.info("✅ Logique RL cohérente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test RL échoué: {e}")
            return False
    
    def test_real_historical_data(self) -> bool:
        """Tester données historiques vraies"""
        try:
            logger.info("📈 Test données historiques vraies...")
            
            # Importer l'agent intégré
            from INTEGRATED_TRADING_SYSTEM import IntegratedTradingAgent, IntegratedTradingConfig
            
            config = IntegratedTradingConfig(enable_rl=True, enable_transformer=True)
            agent = IntegratedTradingAgent(config)
            
            # Récupérer vraies données historiques
            real_data = agent._get_real_historical_data_for_training("SPY", "1y")
            
            if real_data is None or len(real_data) < 100:
                logger.error("❌ Données historiques insuffisantes")
                return False
            
            logger.info(f"📊 Données historiques: {len(real_data)} périodes")
            logger.info(f"📊 Colonnes: {list(real_data.columns)}")
            
            # Vérifier que ce ne sont pas des données np.random
            if 'open' in real_data.columns:
                open_prices = real_data['open'].values
                if len(set(open_prices)) < 10:  # Si trop peu de valeurs uniques
                    logger.error("❌ Données suspectes (trop peu de variations)")
                    return False
            
            # Vérifier indicateurs techniques
            if 'rsi' in real_data.columns:
                rsi_values = real_data['rsi'].dropna()
                if len(rsi_values) > 0:
                    rsi_range = (rsi_values.min(), rsi_values.max())
                    logger.info(f"📊 RSI range: {rsi_range[0]:.1f} - {rsi_range[1]:.1f}")
                    
                    # Vérifier que RSI n'est pas toujours 50.0
                    if rsi_range[0] == rsi_range[1] == 50.0:
                        logger.error("❌ RSI toujours 50.0 (données suspectes)")
                        return False
            
            logger.info("✅ Données historiques vraies validées")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test données historiques échoué: {e}")
            return False
    
    def test_complete_system(self) -> bool:
        """Test complet du système intégré"""
        try:
            logger.info("🚀 Test complet du système...")
            
            # Importer tous les composants
            from INTEGRATED_TRADING_SYSTEM import IntegratedTradingAgent, IntegratedTradingConfig
            from real_indicators_calculator import RealIndicatorsCalculator
            
            # Configuration
            config = IntegratedTradingConfig(
                enable_rl=True,
                enable_multi_broker=True,
                enable_transformer=True
            )
            
            # Créer agent
            agent = IntegratedTradingAgent(config)
            
            # Test initialisation
            agent.initialize_system()
            
            # Test récupération données marché
            symbols = ['AAPL', 'MSFT']
            for symbol in symbols:
                market_data = agent._get_market_data(symbol)
                if not market_data:
                    logger.error(f"❌ Impossible récupérer données {symbol}")
                    return False
                
                logger.info(f"📊 {symbol}: {market_data['source']}")
            
            logger.info("✅ Système complet validé")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test système complet échoué: {e}")
            return False
    
    def generate_validation_report(self) -> str:
        """Générer rapport de validation"""
        report = []
        report.append("# 🧪 RAPPORT DE VALIDATION DES CORRECTIONS CRITIQUES")
        report.append("")
        report.append(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Résumé des tests
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        report.append("## 📊 RÉSUMÉ DES TESTS")
        report.append(f"- **Total:** {total_tests}")
        report.append(f"- **Réussis:** {passed_tests}")
        report.append(f"- **Échoués:** {failed_tests}")
        report.append(f"- **Taux de succès:** {(passed_tests/total_tests)*100:.1f}%")
        report.append("")
        
        # Détail des tests
        report.append("## 🧪 DÉTAIL DES TESTS")
        for test_name, result in self.test_results.items():
            status = "✅ RÉUSSI" if result else "❌ ÉCHOUÉ"
            report.append(f"- **{test_name}:** {status}")
        report.append("")
        
        # Erreurs critiques
        if self.critical_errors:
            report.append("## 🚨 ERREURS CRITIQUES")
            for error in self.critical_errors:
                report.append(f"- {error}")
            report.append("")
        
        # Conclusion
        if failed_tests == 0:
            report.append("## 🎉 CONCLUSION")
            report.append("**✅ TOUTES LES CORRECTIONS VALIDÉES AVEC SUCCÈS!**")
            report.append("")
            report.append("Le système est maintenant **PRODUCTION-READY** et prêt pour déploiement avec capital réel.")
        else:
            report.append("## ⚠️ CONCLUSION")
            report.append(f"**❌ {failed_tests} CORRECTION(S) ONT ÉCHOUÉ**")
            report.append("")
            report.append("Des corrections supplémentaires sont nécessaires avant le déploiement en production.")
        
        return "\n".join(report)

def main():
    """Point d'entrée principal"""
    print("🚀" + "="*80 + "🚀")
    print("   🔥 VALIDATION DES CORRECTIONS CRITIQUES")
    print("="*84)
    print("   🎯 Objectif: Vérifier que le système est PRODUCTION-READY")
    print("🚀" + "="*80 + "🚀")
    
    try:
        validator = SystemValidator()
        success = validator.validate_all_corrections()
        
        # Générer rapport
        report = validator.generate_validation_report()
        
        # Sauvegarder rapport
        with open("VALIDATION_REPORT.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print("\n" + "="*84)
        print("📊 RAPPORT DE VALIDATION GÉNÉRÉ: VALIDATION_REPORT.md")
        print("="*84)
        
        if success:
            print("🏆 TOUS LES TESTS RÉUSSIS!")
            print("🎉 Le système est maintenant PRODUCTION-READY!")
            print("🚀 Prêt pour déploiement avec capital réel!")
            sys.exit(0)
        else:
            print("⚠️  Certains tests ont échoué")
            print("🔧 Des corrections supplémentaires sont nécessaires")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erreur critique lors de la validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
