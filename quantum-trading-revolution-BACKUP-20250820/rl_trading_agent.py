"""
🚀 REINFORCEMENT LEARNING TRADING AGENT
✅ Intégration PPO pour optimisation automatique des décisions
🎯 Objectif: +10-15% optimisation des signaux trading
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("RL_TRADING_AGENT")

# Vérifier si stable-baselines3 est disponible
try:
    from stable_baselines3 import PPO
    PPO_AVAILABLE = True
    logger.info("✅ Stable-Baselines3 disponible")
except ImportError:
    PPO_AVAILABLE = False
    logger.warning("⚠️ Stable-Baselines3 non disponible - Mode simulation activé")

class TradingEnvironment(gym.Env):
    """Environnement de trading pour Reinforcement Learning"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000):
        super(TradingEnvironment, self).__init__()
        
        # ACTIONS: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        
        # OBSERVATIONS: Prix, RSI, MACD, Volume, Position actuelle, Balance, Profit, Drawdown
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        
        logger.info(f"🚀 Environnement RL créé avec {len(data)} périodes")
        logger.info(f"   Balance initiale: ${initial_balance:,.2f}")
    
    def reset(self, seed=None, options=None):
        """Réinitialiser l'environnement"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0=pas de position, 1=long, -1=short
        self.entry_price = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.max_balance = self.initial_balance
        self.min_balance = self.initial_balance
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Retourner l'observation actuelle"""
        if self.current_step >= len(self.data):
            return np.zeros(8, dtype=np.float32)
        
        current_data = self.data.iloc[self.current_step]
        
        # RETOURNER: [prix, rsi, macd, volume, position, balance, profit, drawdown]
        obs = np.array([
            current_data.get('close', 100.0),
            current_data.get('rsi', 50.0),
            current_data.get('macd', 0.0),
            current_data.get('volume', 1000000),
            self.position,
            self.balance,
            self._calculate_profit(),
            self._calculate_drawdown()
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_profit(self):
        """Calculer le profit actuel"""
        if self.position == 0:
            return 0.0
        
        current_price = self.data.iloc[self.current_step]['close']
        if self.position == 1:  # Long position
            return current_price - self.entry_price
        else:  # Short position
            return self.entry_price - current_price
    
    def _calculate_drawdown(self):
        """Calculer le drawdown actuel"""
        if self.balance > self.max_balance:
            self.max_balance = self.balance
        
        if self.balance < self.min_balance:
            self.min_balance = self.balance
        
        if self.max_balance == 0:
            return 0.0
        
        return (self.max_balance - self.balance) / self.max_balance
    
    def step(self, action):
        """Exécuter une action et retourner le nouvel état"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
        
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        info = {}
        
        # Exécuter l'action
        if action == 1 and self.position == 0:  # BUY
            self.position = 1
            self.entry_price = current_price
            self.total_trades += 1
            info['action'] = 'BUY'
            info['entry_price'] = current_price
            
        elif action == 2 and self.position == 1:  # SELL
            profit = current_price - self.entry_price
            self.balance += profit
            reward = profit
            
            if profit > 0:
                self.winning_trades += 1
            
            self.position = 0
            self.entry_price = 0
            info['action'] = 'SELL'
            info['profit'] = profit
            info['win_rate'] = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Mettre à jour le step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Calculer le reward final si terminé
        if done:
            final_profit = self.balance - self.initial_balance
            reward += final_profit * 0.1  # Bonus final
            info['final_balance'] = self.balance
            info['total_return'] = (self.balance / self.initial_balance - 1) * 100
            info['win_rate'] = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        return self._get_observation(), reward, done, False, info

class RLTrader:
    """Trader utilisant le Reinforcement Learning"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.environment = None
        self.model_path = model_path
        self.is_trained = False
        
        logger.info("🤖 RL Trader initialisé")
    
    def train_model(self, historical_data: pd.DataFrame, total_timesteps: int = 10000):
        """Entraîner le modèle RL sur données historiques"""
        if not PPO_AVAILABLE:
            logger.warning("⚠️ PPO non disponible - Mode simulation activé")
            self.is_trained = True
            return
        
        try:
            logger.info(f"🚀 Début entraînement RL sur {len(historical_data)} périodes")
            logger.info(f"   Timesteps: {total_timesteps:,}")
            
            # Créer l'environnement
            self.environment = TradingEnvironment(historical_data)
            
            # Créer et entraîner le modèle
            self.model = PPO("MlpPolicy", self.environment, verbose=1)
            self.model.learn(total_timesteps=total_timesteps)
            
            # Sauvegarder le modèle
            if self.model_path:
                self.model.save(self.model_path)
                logger.info(f"💾 Modèle sauvegardé: {self.model_path}")
            
            self.is_trained = True
            logger.info("✅ Entraînement RL terminé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement RL: {e}")
            self.is_trained = False
    
    def load_model(self, model_path: str):
        """Charger un modèle pré-entraîné"""
        if not PPO_AVAILABLE:
            logger.warning("⚠️ PPO non disponible - Impossible de charger le modèle")
            return False
        
        try:
            self.model = PPO.load(model_path)
            self.model_path = model_path
            self.is_trained = True
            logger.info(f"✅ Modèle chargé: {model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            return False
    
    def get_rl_decision(self, current_state: List[float]) -> int:
        """Obtenir décision du modèle RL"""
        if not self.is_trained or self.model is None:
            return 0  # HOLD par défaut
        
        try:
            obs = np.array(current_state, dtype=np.float32)
            action, _ = self.model.predict(obs)
            return int(action)
        except Exception as e:
            logger.error(f"❌ Erreur prédiction RL: {e}")
            return 0  # HOLD en cas d'erreur
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict:
        """Évaluer la performance du modèle sur données de test"""
        if not self.is_trained or self.model is None:
            return {'error': 'Modèle non entraîné'}
        
        try:
            test_env = TradingEnvironment(test_data)
            obs = test_env.reset()
            done = False
            total_reward = 0
            trades = []
            
            while not done:
                action = self.get_rl_decision(obs)
                obs, reward, done, truncated, info = test_env.step(action)
                total_reward += reward
                
                if 'action' in info:
                    trades.append(info)
            
            # Calculer métriques
            final_balance = test_env.balance
            total_return = (final_balance / test_env.initial_balance - 1) * 100
            win_rate = test_env.winning_trades / test_env.total_trades if test_env.total_trades > 0 else 0
            
            results = {
                'total_reward': total_reward,
                'final_balance': final_balance,
                'total_return_pct': total_return,
                'total_trades': test_env.total_trades,
                'winning_trades': test_env.winning_trades,
                'win_rate': win_rate,
                'max_drawdown': test_env._calculate_drawdown()
            }
            
            logger.info(f"📊 Évaluation RL - Retour: {total_return:.2f}%, Win Rate: {win_rate:.1%}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur évaluation RL: {e}")
            return {'error': str(e)}

def create_sample_data(n_periods: int = 1000) -> pd.DataFrame:
    """Créer des données d'exemple pour test"""
    np.random.seed(42)
    
    # Prix simulés avec tendance
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, n_periods):
        # Tendance + bruit
        trend = 0.0001 * i  # Tendance légèrement haussière
        noise = np.random.normal(0, 0.02)  # 2% volatilité
        new_price = prices[-1] * (1 + trend + noise)
        prices.append(max(new_price, 1.0))  # Prix minimum $1
    
    # Créer DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(100000, 1000000, n_periods),
        'rsi': np.random.uniform(20, 80, n_periods),
        'macd': np.random.uniform(-2, 2, n_periods)
    })
    
    return df

def main():
    """Test du système RL"""
    print("🚀" + "="*80 + "🚀")
    print("   🔥 REINFORCEMENT LEARNING TRADING AGENT")
    print("="*84)
    print("   ✅ Environnement de trading gym")
    print("   ✅ Modèle PPO (si disponible)")
    print("   🎯 Objectif: +10-15% optimisation trading")
    print("🚀" + "="*80 + "🚀")
    
    # Créer données d'exemple
    print("\n📊 Création données d'exemple...")
    sample_data = create_sample_data(500)
    print(f"   ✅ {len(sample_data)} périodes créées")
    
    # Diviser en train/test
    split_idx = int(len(sample_data) * 0.8)
    train_data = sample_data[:split_idx]
    test_data = sample_data[split_idx:]
    
    print(f"   📈 Entraînement: {len(train_data)} périodes")
    print(f"   🧪 Test: {len(test_data)} périodes")
    
    # Initialiser le trader RL
    rl_trader = RLTrader()
    
    # Entraîner le modèle
    print("\n🤖 Entraînement du modèle RL...")
    rl_trader.train_model(train_data, total_timesteps=5000)
    
    if rl_trader.is_trained:
        # Évaluer le modèle
        print("\n📊 Évaluation du modèle...")
        results = rl_trader.evaluate_model(test_data)
        
        if 'error' not in results:
            print(f"   🎯 Retour total: {results['total_return_pct']:.2f}%")
            print(f"   📈 Trades: {results['total_trades']}")
            print(f"   ✅ Win Rate: {results['win_rate']:.1%}")
            print(f"   💰 Balance finale: ${results['final_balance']:,.2f}")
        else:
            print(f"   ❌ Erreur: {results['error']}")
    else:
        print("   ⚠️ Modèle non entraîné - Mode simulation")
    
    print("\n✅ Test RL terminé!")

if __name__ == "__main__":
    main()
