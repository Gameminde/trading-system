"""
🚀 REINFORCEMENT LEARNING TRADING AGENT - CORRIGÉ
✅ Intégration PPO pour optimisation automatique des décisions
✅ CORRECTION CRITIQUE: Logique balance/positions cohérente
✅ Gestion correcte du capital et des actions
🎯 Objectif: +10-15% optimisation des signaux trading
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("RL_TRADING_AGENT")

class TradingEnvironment(gym.Env):
    """Environnement de trading corrigé avec gestion correcte"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000):
        super(TradingEnvironment, self).__init__()
        
        self.action_space = spaces.Discrete(3)  # 0=HOLD, 1=BUY, 2=SELL
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Réinitialiser l'environnement"""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0  # Nombre d'actions détenues
        self.entry_price = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.max_portfolio_value = self.initial_balance
        return self._get_observation(), {}
    
    def step(self, action):
        """LOGIQUE CORRIGÉE pour gestion portfolio"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, False, {}

        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        info = {}

        # LOGIQUE CORRIGÉE : Gestion du capital
        if action == 1 and self.shares == 0:  # BUY (si pas de position)
            # Calculer combien d'actions on peut acheter
            max_shares = int(self.balance / current_price)
            
            if max_shares > 0:
                # Acheter le maximum possible
                cost = max_shares * current_price
                self.balance -= cost
                self.shares = max_shares
                self.entry_price = current_price
                self.total_trades += 1
                
                info['action'] = 'BUY'
                info['shares'] = max_shares
                info['cost'] = cost
                
                logger.debug(f"BUY: {max_shares} actions @ ${current_price:.2f}")

        elif action == 2 and self.shares > 0:  # SELL (si on a des actions)
            # Vendre toutes les actions
            proceeds = self.shares * current_price
            profit = proceeds - (self.shares * self.entry_price)
            
            self.balance += proceeds
            
            if profit > 0:
                self.winning_trades += 1
                reward = profit / self.initial_balance  # Reward normalisé
            else:
                reward = profit / self.initial_balance
            
            info['action'] = 'SELL'
            info['shares'] = self.shares
            info['proceeds'] = proceeds
            info['profit'] = profit
            
            logger.debug(f"SELL: {self.shares} actions @ ${current_price:.2f}, P&L: ${profit:.2f}")
            
            self.shares = 0
            self.entry_price = 0

        # Calculer la valeur totale du portfolio
        portfolio_value = self.balance + (self.shares * current_price)
        
        # Bonus/Malus pour performance du portfolio
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
            reward += 0.01  # Bonus pour nouveau high
        
        # Malus pour drawdown
        drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
        if drawdown > 0.1:  # Plus de 10% de drawdown
            reward -= drawdown * 0.1

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        if done:
            # Vendre automatiquement à la fin
            if self.shares > 0:
                final_proceeds = self.shares * current_price
                self.balance += final_proceeds
                self.shares = 0
            
            final_return = (self.balance / self.initial_balance - 1) * 100
            info['final_return'] = final_return
            info['win_rate'] = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            info['total_trades'] = self.total_trades

        return self._get_observation(), reward, done, False, info

    def _get_observation(self):
        """Observation corrigée avec vraies métriques"""
        if self.current_step >= len(self.data):
            return np.zeros(8, dtype=np.float32)

        current_data = self.data.iloc[self.current_step]
        current_price = current_data['close']
        
        # Portfolio value actuel
        portfolio_value = self.balance + (self.shares * current_price)
        
        obs = np.array([
            current_price,
            current_data.get('rsi', 50.0),
            current_data.get('macd', 0.0),
            current_data.get('volume', 1000000),
            float(self.shares),  # Nombre d'actions détenues
            self.balance,
            portfolio_value,
            (portfolio_value / self.initial_balance - 1) * 100  # Return %
        ], dtype=np.float32)

        return obs

class RLTrader:
    """Agent de trading RL avec PPO"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.environment = None
        self.model_path = model_path
        
        logger.info("🤖 RLTrader initialisé")
    
    def train_model(self, historical_data: pd.DataFrame, total_timesteps: int = 10000):
        """Entraîner le modèle RL sur données historiques"""
        try:
            logger.info(f"🚀 Entraînement modèle RL ({total_timesteps} timesteps)...")
            
            # Créer environnement
            self.environment = TradingEnvironment(historical_data)
            
            # Créer modèle PPO
            self.model = PPO(
                "MlpPolicy",
                self.environment,
                verbose=1,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01
            )
            
            # Entraînement
            self.model.learn(total_timesteps=total_timesteps)
            
            # Sauvegarder le modèle
            if self.model_path:
                self.model.save(self.model_path)
                logger.info(f"💾 Modèle sauvegardé: {self.model_path}")
            
            logger.info("✅ Modèle RL entraîné avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement RL: {e}")
            raise
    
    def load_model(self, model_path: str):
        """Charger un modèle pré-entraîné"""
        try:
            logger.info(f"📥 Chargement modèle: {model_path}")
            self.model = PPO.load(model_path)
            logger.info("✅ Modèle chargé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            raise
    
    def get_rl_decision(self, current_state: List[float]) -> int:
        """Obtenir décision du modèle RL"""
        if self.model is None:
            logger.warning("⚠️ Modèle RL non disponible")
            return 0  # HOLD par défaut
        
        try:
            # Préparer observation
            obs = np.array(current_state, dtype=np.float32)
            
            # Prédiction
            action, _ = self.model.predict(obs, deterministic=True)
            
            return int(action)
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction RL: {e}")
            return 0  # HOLD en cas d'erreur
    
    def evaluate_performance(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Évaluer les performances du modèle RL"""
        try:
            if self.model is None:
                return {'error': 'Modèle non disponible'}
            
            # Créer environnement de test
            test_env = TradingEnvironment(test_data)
            
            # Test complet
            obs = test_env.reset()[0]
            done = False
            total_reward = 0
            trades = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(action)
                total_reward += reward
                
                if 'action' in info and info['action'] in ['BUY', 'SELL']:
                    trades.append(info)
            
            # Calculer métriques
            final_return = info.get('final_return', 0)
            win_rate = info.get('win_rate', 0)
            total_trades = info.get('total_trades', 0)
            
            metrics = {
                'total_reward': float(total_reward),
                'final_return': float(final_return),
                'win_rate': float(win_rate),
                'total_trades': int(total_trades),
                'avg_reward_per_trade': float(total_reward / total_trades) if total_trades > 0 else 0
            }
            
            logger.info(f"📊 Performance RL: Return={final_return:.2f}%, Win Rate={win_rate:.1%}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur évaluation RL: {e}")
            return {'error': str(e)}

def main():
    """Test de l'agent RL corrigé"""
    print("🚀 TEST RL TRADING AGENT CORRIGÉ")
    print("="*50)
    
    try:
        # Créer données de test
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        test_data = pd.DataFrame({
            'close': np.random.uniform(95, 105, len(dates)),
            'rsi': np.random.uniform(30, 70, len(dates)),
            'macd': np.random.uniform(-2, 2, len(dates)),
            'volume': np.random.uniform(900000, 1100000, len(dates))
        }, index=dates)
        
        # Créer agent RL
        rl_trader = RLTrader()
        
        # Entraîner modèle
        rl_trader.train_model(test_data, total_timesteps=1000)
        
        # Tester décision
        current_state = [100.0, 50.0, 0.0, 1000000, 0, 100000, 100000, 0.0]
        decision = rl_trader.get_rl_decision(current_state)
        
        action_names = ['HOLD', 'BUY', 'SELL']
        print(f"✅ Décision RL: {action_names[decision]}")
        
        # Évaluer performance
        metrics = rl_trader.evaluate_performance(test_data.tail(100))
        print(f"📊 Métriques: {metrics}")
        
        print("\n🎯 RL Trading Agent corrigé testé avec succès!")
        
    except Exception as e:
        print(f"❌ Test échoué: {e}")

if __name__ == "__main__":
    main()



