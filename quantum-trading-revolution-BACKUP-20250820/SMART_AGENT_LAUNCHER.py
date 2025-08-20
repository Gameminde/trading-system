"""
🚀 SMART AGENT LAUNCHER - CHOIX INTELLIGENT

MISSION: Lancer l'agent optimal selon les données disponibles
OPTIONS: Agent standard vs Agent entraîné historiquement
RÉSULTAT: Performance maximale basée sur vraie connaissance du marché

💎 L'INTERFACE PARFAITE POUR CHOISIR SON AGENT !
"""

import os
import sys
import json
from datetime import datetime
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SMART_LAUNCHER")

class SmartAgentLauncher:
    """Lanceur intelligent pour choisir le meilleur agent"""
    
    def __init__(self):
        self.models_dir = "models/trained"
        self.has_trained_agent = False
        self.training_info = None
        
        self.check_trained_models()
    
    def check_trained_models(self):
        """Vérifier si un agent entraîné existe"""
        
        trained_agent_path = f"{self.models_dir}/trained_agent.pkl"
        training_results_path = f"{self.models_dir}/training_results.json"
        
        if os.path.exists(trained_agent_path) and os.path.exists(training_results_path):
            self.has_trained_agent = True
            
            try:
                with open(training_results_path, 'r') as f:
                    self.training_info = json.load(f)
                    
                logger.info("✅ Trained agent found")
                
            except Exception as e:
                logger.warning(f"⚠️ Error reading training info: {e}")
        else:
            logger.info("📊 No trained agent found")
    
    def display_options(self):
        """Afficher les options disponibles"""
        
        print("🚀" + "="*70 + "🚀")
        print("   🧠 SMART AGENT LAUNCHER - CHOISISSEZ VOTRE ARME")
        print("="*74)
        
        print(f"\n📊 OPTIONS DISPONIBLES :")
        
        # Option 1: Agent Standard
        print(f"\n1️⃣ AGENT OPTIMIZED (Standard)")
        print(f"   ⚡ Puissance: 95% modules actifs")
        print(f"   🧠 Intelligence: LLM + MPS + Quantum")
        print(f"   📊 Données: Temps réel uniquement")
        print(f"   🎯 Performance: Optimisé anti-overtrading")
        print(f"   ✅ Status: Toujours disponible")
        
        # Option 2: Agent Entraîné (si disponible)
        if self.has_trained_agent:
            print(f"\n2️⃣ AGENT TRAINED (Historiquement Intelligent) ⭐ RECOMMANDÉ")
            print(f"   🎓 Entraîné sur: {self.training_info.get('data_periods', {}).get('crypto_years', 5)} ans crypto + {self.training_info.get('data_periods', {}).get('stocks_years', 5)} ans stocks")
            print(f"   📈 Patterns appris: {self.training_info.get('patterns_found', 0)} stratégies gagnantes")
            print(f"   🔍 Régimes identifiés: {self.training_info.get('regimes_identified', 0)} contextes de marché")
            print(f"   🧠 Config optimisée: Basée sur backtest historique")
            print(f"   📅 Entraîné le: {self.training_info.get('training_date', 'N/A')[:10]}")
            print(f"   ✅ Status: Prêt pour performance supérieure")
        else:
            print(f"\n2️⃣ AGENT TRAINED (Non disponible)")
            print(f"   ❌ Pas encore entraîné")
            print(f"   🎓 Exécutez 'python HISTORICAL_DATA_TRAINER.py' d'abord")
            print(f"   ⏱️ Temps d'entraînement: ~10-15 minutes")
            print(f"   💡 Collecte 5-10 ans de données BTC, ETH, TSLA, AAPL")
        
        # Option 3: Entraînement
        print(f"\n3️⃣ ENTRAÎNER NOUVEAU AGENT")
        print(f"   📚 Collecte données historiques complètes")
        print(f"   🔍 Analyse patterns gagnants sur 5-10 ans") 
        print(f"   🎯 Optimise configuration automatiquement")
        print(f"   💾 Sauvegarde agent super-intelligent")
        print(f"   ⏱️ Durée: 10-15 minutes")
        
        # Option 4: Dashboard seul
        print(f"\n4️⃣ DASHBOARD SEUL (Interface graphique)")
        print(f"   🎨 Interface web moderne")
        print(f"   📊 Graphiques temps réel")
        print(f"   🎛️ Contrôles START/STOP manuels")
        print(f"   🌐 http://localhost:8050")
    
    def get_user_choice(self):
        """Obtenir le choix de l'utilisateur"""
        
        while True:
            try:
                print(f"\n🎯 VOTRE CHOIX :")
                
                if self.has_trained_agent:
                    print(f"   💡 RECOMMANDÉ: Option 2 (Agent Trained) pour performance max")
                else:
                    print(f"   💡 RECOMMANDÉ: Option 3 (Entraîner) puis Option 2")
                
                choice = input(f"\n➡️ Entrez votre choix [1-4]: ").strip()
                
                if choice in ['1', '2', '3', '4']:
                    return int(choice)
                else:
                    print(f"❌ Choix invalide. Entrez 1, 2, 3 ou 4.")
                    
            except KeyboardInterrupt:
                print(f"\n👋 Au revoir !")
                sys.exit(0)
    
    def launch_option_1(self):
        """Lancer Agent Optimized Standard"""
        
        print(f"\n🚀 Lancement Agent OPTIMIZED Standard...")
        print(f"   ⚡ Puissance maximale + Intelligence anti-overtrading")
        
        try:
            from AGENT_OPTIMIZED_MAXIMUM_POWER import main as launch_optimized
            launch_optimized()
            
        except Exception as e:
            logger.error(f"❌ Erreur lancement agent standard: {e}")
    
    def launch_option_2(self):
        """Lancer Agent Trained (si disponible)"""
        
        if not self.has_trained_agent:
            print(f"❌ Agent entraîné non disponible")
            print(f"💡 Choisissez option 3 pour entraîner d'abord")
            return
        
        print(f"\n🎓 Lancement Agent TRAINED (Historiquement Intelligent)...")
        print(f"   📊 Patterns: {self.training_info.get('patterns_found', 0)} stratégies")
        print(f"   🧠 Optimisé sur {self.training_info.get('data_periods', {}).get('crypto_years', 5)} ans de données")
        
        try:
            from HISTORICAL_DATA_TRAINER import HistoricalDataTrainer
            
            trainer = HistoricalDataTrainer()
            trained_agent = trainer.load_trained_agent()
            
            if trained_agent:
                # Lancer session avec agent entraîné
                symbols = ["BTC", "ETH", "AAPL", "TSLA", "MSFT", "GOOGL", "BNB"]
                results = trained_agent.run_optimized_session(symbols, duration_minutes=15)
                
                print(f"\n🏆 RÉSULTATS AGENT ENTRAÎNÉ:")
                print(f"   📊 Performance: {results.get('return_percent', 0):.1f}%")
                print(f"   🎯 Success rate: {results.get('success_rate', 0):.1f}%")
                print(f"   📈 Trades: {results.get('trades_executed', 0)}")
                
            else:
                print(f"❌ Impossible de charger l'agent entraîné")
                
        except Exception as e:
            logger.error(f"❌ Erreur lancement agent entraîné: {e}")
            import traceback
            traceback.print_exc()
    
    def launch_option_3(self):
        """Entraîner nouveau agent"""
        
        print(f"\n🎓 Lancement ENTRAÎNEMENT Agent...")
        print(f"   📚 Cela va prendre 10-15 minutes")
        print(f"   📊 Collecte BTC (5-10 ans), ETH, TSLA, AAPL...")
        
        try:
            from HISTORICAL_DATA_TRAINER import main as launch_training
            launch_training()
            
            # Après entraînement, relancer launcher
            self.check_trained_models()
            
            if self.has_trained_agent:
                print(f"\n🎉 ENTRAÎNEMENT TERMINÉ !")
                print(f"   💡 Vous pouvez maintenant choisir Option 2")
                
                retry = input(f"\n🚀 Lancer directement l'agent entraîné ? [y/N]: ").lower()
                if retry == 'y':
                    self.launch_option_2()
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement: {e}")
            import traceback
            traceback.print_exc()
    
    def launch_option_4(self):
        """Lancer Dashboard seul"""
        
        print(f"\n🎨 Lancement DASHBOARD Interface...")
        print(f"   🌐 URL: http://localhost:8050")
        print(f"   🎛️ Contrôles manuels dans l'interface")
        
        try:
            from AGENT_TRADING_DASHBOARD import main as launch_dashboard
            launch_dashboard()
            
        except Exception as e:
            logger.error(f"❌ Erreur lancement dashboard: {e}")
    
    def run(self):
        """Exécuter le launcher principal"""
        
        while True:
            self.display_options()
            choice = self.get_user_choice()
            
            if choice == 1:
                self.launch_option_1()
            elif choice == 2:
                self.launch_option_2()
            elif choice == 3:
                self.launch_option_3()
            elif choice == 4:
                self.launch_option_4()
            
            # Demander si continuer
            print(f"\n" + "="*50)
            continue_choice = input(f"🔄 Revenir au menu principal ? [y/N]: ").lower()
            if continue_choice != 'y':
                break
        
        print(f"\n👋 Merci d'avoir utilisé Smart Agent Launcher !")

def main():
    """Fonction principale"""
    
    launcher = SmartAgentLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
