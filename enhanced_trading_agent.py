"""
🚀 ENHANCED TRADING AGENT WITH MEMORY DECODER
Agent de trading amélioré avec Memory Decoder intégré
Capital réaliste de 1000$ et métriques avancées
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Any
from stable_baselines3 import PPO
from datetime import datetime
import json

# Import des modules custom
from memory_decoder_trading import TradingMemoryDecoder, TradingMemoryLoss
from rl_trading_agent import TradingEnvironment, RLTrader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("ENHANCED_AGENT")

class EnhancedTradingEnvironment(TradingEnvironment):
    """Environnement de trading amélioré avec métriques avancées"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 1000):
        super().__init__(data, initial_balance)
        
        # Métriques avancées
        self.metrics = {
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'win_loss_ratio': 0.0,
            'avg_trade_duration': 0.0,
            'total_fees': 0.0
        }
        
        # Configuration réaliste
        self.trading_fee = 0.001  # 0.1% de frais par trade
        self.slippage = 0.0005    # 0.05% de slippage
        
        # Historique pour calculs
        self.portfolio_history = []
        self.trade_durations = []
        self.returns_history = []
        
        logger.info(f"💰 Environnement amélioré initialisé avec ${initial_balance}")
    
    def step(self, action):
        """Step amélioré avec frais et slippage"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, False, self._get_final_metrics()
        
        current_price = self.data.iloc[self.current_step]['close']
        
        # Appliquer slippage
        execution_price = current_price * (1 + self.slippage * (1 if action == 1 else -1 if action == 2 else 0))
        
        reward = 0
        info = {}
        
        # LOGIQUE DE TRADING AVEC FRAIS
        if action == 1 and self.shares == 0:  # BUY
            max_shares = int(self.balance / (execution_price * (1 + self.trading_fee)))
            
            if max_shares > 0:
                cost = max_shares * execution_price * (1 + self.trading_fee)
                self.balance -= cost
                self.shares = max_shares
                self.entry_price = execution_price
                self.entry_step = self.current_step
                self.total_trades += 1
                self.metrics['total_fees'] += max_shares * execution_price * self.trading_fee
                
                info['action'] = 'BUY'
                info['shares'] = max_shares
                info['price'] = execution_price
                info['fees'] = max_shares * execution_price * self.trading_fee
                
                logger.debug(f"BUY: {max_shares} @ ${execution_price:.2f} (fees: ${info['fees']:.2f})")
        
        elif action == 2 and self.shares > 0:  # SELL
            proceeds = self.shares * execution_price * (1 - self.trading_fee)
            fees = self.shares * execution_price * self.trading_fee
            profit = proceeds - (self.shares * self.entry_price)
            
            self.balance += proceeds
            self.metrics['total_fees'] += fees
            
            # Calculer durée du trade
            trade_duration = self.current_step - self.entry_step
            self.trade_durations.append(trade_duration)
            
            if profit > 0:
                self.winning_trades += 1
                reward = profit / self.initial_balance
            else:
                reward = profit / self.initial_balance
            
            info['action'] = 'SELL'
            info['shares'] = self.shares
            info['price'] = execution_price
            info['profit'] = profit
            info['fees'] = fees
            info['duration'] = trade_duration
            
            logger.debug(f"SELL: {self.shares} @ ${execution_price:.2f}, P&L: ${profit:.2f}")
            
            self.shares = 0
            self.entry_price = 0
        
        # Calculer valeur du portfolio
        portfolio_value = self.balance + (self.shares * current_price)
        self.portfolio_history.append(portfolio_value)
        
        # Calculer return journalier
        if len(self.portfolio_history) > 1:
            daily_return = (portfolio_value - self.portfolio_history[-2]) / self.portfolio_history[-2]
            self.returns_history.append(daily_return)
        
        # Mise à jour métriques
        self._update_metrics()
        
        # Reward amélioré
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
            reward += 0.01
        
        # Pénalité pour drawdown
        drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
        if drawdown > 0.1:
            reward -= drawdown * 0.1
        
        # Pénalité pour inactivité
        if action == 0 and len(self.portfolio_history) > 10:
            reward -= 0.001
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, False, info
    
    def _update_metrics(self):
        """Calculer métriques avancées"""
        if len(self.returns_history) > 1:
            # Sharpe Ratio
            returns_array = np.array(self.returns_history)
            if returns_array.std() > 0:
                self.metrics['sharpe_ratio'] = (returns_array.mean() * 252) / (returns_array.std() * np.sqrt(252))
            
            # Maximum Drawdown
            portfolio_array = np.array(self.portfolio_history)
            running_max = np.maximum.accumulate(portfolio_array)
            drawdown = (running_max - portfolio_array) / running_max
            self.metrics['max_drawdown'] = np.max(drawdown)
            
            # Calmar Ratio
            if self.metrics['max_drawdown'] > 0:
                annual_return = (portfolio_array[-1] / self.initial_balance - 1) * (252 / len(portfolio_array))
                self.metrics['calmar_ratio'] = annual_return / self.metrics['max_drawdown']
            
            # Win/Loss Ratio
            if self.total_trades > 0:
                self.metrics['win_loss_ratio'] = self.winning_trades / self.total_trades
            
            # Average Trade Duration
            if self.trade_durations:
                self.metrics['avg_trade_duration'] = np.mean(self.trade_durations)
    
    def _get_final_metrics(self) -> Dict:
        """Obtenir métriques finales"""
        final_value = self.balance + (self.shares * self.data.iloc[-1]['close'])
        
        return {
            'final_value': final_value,
            'total_return': (final_value / self.initial_balance - 1) * 100,
            'sharpe_ratio': self.metrics['sharpe_ratio'],
            'max_drawdown': self.metrics['max_drawdown'],
            'calmar_ratio': self.metrics['calmar_ratio'],
            'win_rate': self.metrics['win_loss_ratio'],
            'avg_trade_duration': self.metrics['avg_trade_duration'],
            'total_fees': self.metrics['total_fees'],
            'total_trades': self.total_trades
        }

class IntegratedMemoryTrader:
    """Trader intégrant PPO et Memory Decoder"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'use_memory_decoder': True,
            'memory_weight': 0.6,
            'initial_capital': 1000,
            'memory_size': 50000,
            'update_frequency': 100
        }
        
        # Composants principaux
        self.rl_trader = RLTrader()
        self.memory_decoder = TradingMemoryDecoder() if self.config['use_memory_decoder'] else None
        self.environment = None
        
        # Tracking
        self.trade_counter = 0
        self.memory_update_counter = 0
        
        logger.info("🤖 Integrated Memory Trader initialisé")
        logger.info(f"   Capital: ${self.config['initial_capital']}")
        logger.info(f"   Memory Decoder: {'Activé' if self.config['use_memory_decoder'] else 'Désactivé'}")
    
    def train(self, historical_data: pd.DataFrame, epochs: int = 10):
        """Entraîner le système complet"""
        logger.info("🚀 Début de l'entraînement intégré...")
        
        # Créer environnement amélioré
        self.environment = EnhancedTradingEnvironment(
            historical_data, 
            initial_balance=self.config['initial_capital']
        )
        
        # Phase 1: Entraîner RL de base
        logger.info("📚 Phase 1: Entraînement RL de base...")
        self.rl_trader.train_model(historical_data, total_timesteps=5000)
        
        # Phase 2: Pré-entraîner Memory Decoder si activé
        if self.memory_decoder:
            logger.info("🧠 Phase 2: Pré-entraînement Memory Decoder...")
            self._pretrain_memory_decoder(historical_data)
        
        # Phase 3: Entraînement conjoint
        logger.info("🔄 Phase 3: Entraînement conjoint...")
        for epoch in range(epochs):
            logger.info(f"Époque {epoch + 1}/{epochs}")
            
            # Reset environnement
            obs, _ = self.environment.reset()
            done = False
            episode_trades = []
            
            while not done:
                # Décision hybride
                action = self._get_hybrid_decision(obs)
                
                # Step
                next_obs, reward, done, truncated, info = self.environment.step(action)
                
                # Mise à jour mémoire
                if self.memory_decoder and 'action' in info:
                    self._update_memory(obs, action, reward, info)
                    episode_trades.append(info)
                
                obs = next_obs
            
            # Métriques d'époque
            metrics = self.environment._get_final_metrics()
            logger.info(f"   Return: {metrics['total_return']:.2f}%")
            logger.info(f"   Sharpe: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"   Max DD: {metrics['max_drawdown']:.2%}")
            logger.info(f"   Trades: {metrics['total_trades']}")
        
        logger.info("✅ Entraînement terminé")
    
    def _pretrain_memory_decoder(self, data: pd.DataFrame):
        """Pré-entraîner le Memory Decoder sur données historiques"""
        if not self.memory_decoder:
            return
        
        # Créer dataset d'entraînement
        sequences = []
        targets = []
        
        for i in range(len(data) - 10):
            # Créer séquence de contexte
            context_window = data.iloc[i:i+10]
            
            # Encoder le contexte
            market_context = {
                'price': context_window['close'].iloc[-1],
                'rsi': context_window['rsi'].iloc[-1] if 'rsi' in context_window else 50,
                'macd': context_window['macd'].iloc[-1] if 'macd' in context_window else 0,
                'volume': context_window['volume'].iloc[-1] if 'volume' in context_window else 1000000
            }
            
            encoded_context = self.memory_decoder.encode_market_context(market_context)
            sequences.append(encoded_context)
            
            # Déterminer target (action optimale basée sur prix futur)
            if i < len(data) - 11:
                future_price = data.iloc[i+11]['close']
                current_price = data.iloc[i+10]['close']
                price_change = (future_price - current_price) / current_price
                
                # Mapper à action
                if price_change > 0.01:
                    target = 4  # STRONG_BUY
                elif price_change > 0.002:
                    target = 3  # BUY
                elif price_change < -0.01:
                    target = 0  # STRONG_SELL
                elif price_change < -0.002:
                    target = 1  # SELL
                else:
                    target = 2  # HOLD
                
                targets.append(target)
        
        # Entraînement basique (simplifié pour l'exemple)
        logger.info(f"   Pré-entraînement sur {len(sequences)} séquences...")
        
        # Stocker quelques exemples dans la mémoire
        for i in range(min(100, len(sequences))):
            if i < len(targets):
                self.memory_decoder.update_memory(
                    sequences[i].squeeze(),
                    targets[i],
                    0.0  # Reward placeholder
                )
    
    def _get_hybrid_decision(self, observation: np.ndarray) -> int:
        """Obtenir décision hybride RL + Memory Decoder"""
        # Décision RL
        rl_action = self.rl_trader.get_rl_decision(observation.tolist())
        
        if not self.memory_decoder or not self.config['use_memory_decoder']:
            return rl_action
        
        # Encoder observation pour Memory Decoder
        market_context = {
            'price': float(observation[0]),
            'rsi': float(observation[1]) if len(observation) > 1 else 50,
            'macd': float(observation[2]) if len(observation) > 2 else 0,
            'volume': float(observation[3]) if len(observation) > 3 else 1000000
        }
        
        encoded = self.memory_decoder.encode_market_context(market_context)
        
        # Obtenir prédiction Memory Decoder
        with torch.no_grad():
            memory_output = self.memory_decoder(encoded.unsqueeze(0))
            memory_probs = memory_output['action_probs'].squeeze()
        
        # Mapper les 5 actions du Memory Decoder aux 3 actions de l'environnement
        # STRONG_SELL(0) + SELL(1) -> SELL(2)
        # HOLD(2) -> HOLD(0)
        # BUY(3) + STRONG_BUY(4) -> BUY(1)
        
        sell_prob = memory_probs[0] + memory_probs[1]
        hold_prob = memory_probs[2]
        buy_prob = memory_probs[3] + memory_probs[4]
        
        memory_action_probs = torch.tensor([hold_prob, buy_prob, sell_prob])
        memory_action = torch.argmax(memory_action_probs).item()
        
        # Combiner décisions avec pondération
        if np.random.random() < self.config['memory_weight']:
            return memory_action
        else:
            return rl_action
    
    def _update_memory(self, observation: np.ndarray, action: int, reward: float, info: Dict):
        """Mettre à jour la mémoire du decoder"""
        if not self.memory_decoder:
            return
        
        self.memory_update_counter += 1
        
        # Mise à jour périodique seulement
        if self.memory_update_counter % self.config['update_frequency'] != 0:
            return
        
        # Encoder contexte
        market_context = {
            'price': float(observation[0]),
            'rsi': float(observation[1]) if len(observation) > 1 else 50,
            'macd': float(observation[2]) if len(observation) > 2 else 0,
            'volume': float(observation[3]) if len(observation) > 3 else 1000000
        }
        
        encoded = self.memory_decoder.encode_market_context(market_context)
        
        # Ajouter métadonnées
        metadata = {
            'profit': info.get('profit', 0),
            'trade_type': info.get('action', 'HOLD'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Mettre à jour mémoire
        self.memory_decoder.update_memory(
            encoded.squeeze(),
            action,
            reward,
            metadata
        )
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict:
        """Évaluer performance sur données de test"""
        logger.info("📊 Évaluation sur données de test...")
        
        # Créer environnement de test
        test_env = EnhancedTradingEnvironment(
            test_data,
            initial_balance=self.config['initial_capital']
        )
        
        # Simulation
        obs, _ = test_env.reset()
        done = False
        
        while not done:
            action = self._get_hybrid_decision(obs)
            obs, reward, done, truncated, info = test_env.step(action)
        
        # Résultats
        metrics = test_env._get_final_metrics()
        
        logger.info("📈 Résultats d'évaluation:")
        logger.info(f"   Return: {metrics['total_return']:.2f}%")
        logger.info(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"   Win Rate: {metrics['win_rate']:.2%}")
        logger.info(f"   Total Trades: {metrics['total_trades']}")
        logger.info(f"   Total Fees: ${metrics['total_fees']:.2f}")
        
        return metrics
    
    def save_system(self, filepath: str):
        """Sauvegarder le système complet"""
        try:
            # Sauvegarder RL model
            if self.rl_trader.model:
                self.rl_trader.model.save(f"{filepath}_rl.zip")
            
            # Sauvegarder Memory Decoder
            if self.memory_decoder:
                self.memory_decoder.save_memory(f"{filepath}_memory.pt")
            
            # Sauvegarder config
            with open(f"{filepath}_config.json", 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"💾 Système sauvegardé: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")
    
    def load_system(self, filepath: str):
        """Charger le système complet"""
        try:
            # Charger config
            with open(f"{filepath}_config.json", 'r') as f:
                self.config = json.load(f)
            
            # Charger RL model
            self.rl_trader.load_model(f"{filepath}_rl.zip")
            
            # Charger Memory Decoder
            if self.memory_decoder:
                self.memory_decoder.load_memory(f"{filepath}_memory.pt")
            
            logger.info(f"📂 Système chargé: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement: {e}")

def main():
    """Test du système intégré"""
    print("="*70)
    print("🚀 TEST DU SYSTÈME DE TRADING AMÉLIORÉ AVEC MEMORY DECODER")
    print("="*70)
    
    # Configuration
    config = {
        'use_memory_decoder': True,
        'memory_weight': 0.6,
        'initial_capital': 1000,  # Capital réaliste
        'memory_size': 10000,
        'update_frequency': 10
    }
    
    # Créer données de test réalistes
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Simuler tendance avec volatilité
    trend = np.linspace(100, 120, len(dates))  # Tendance haussière
    noise = np.random.randn(len(dates)) * 2    # Volatilité
    prices = trend + noise
    
    test_data = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.uniform(900000, 1100000, len(dates)),
        'rsi': 50 + np.sin(np.arange(len(dates)) * 0.1) * 20,
        'macd': np.sin(np.arange(len(dates)) * 0.05) * 2
    }, index=dates)
    
    # Créer trader intégré
    trader = IntegratedMemoryTrader(config)
    
    # Entraîner
    print("\n📚 Entraînement du système...")
    train_data = test_data.iloc[:250]
    trader.train(train_data, epochs=3)
    
    # Évaluer
    print("\n📊 Évaluation sur données de test...")
    test_data_eval = test_data.iloc[250:]
    metrics = trader.evaluate(test_data_eval)
    
    # Afficher résumé
    print("\n" + "="*70)
    print("📈 RÉSUMÉ DES PERFORMANCES")
    print("="*70)
    print(f"Capital Initial: ${config['initial_capital']:,.2f}")
    print(f"Return Total: {metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Nombre de Trades: {metrics['total_trades']}")
    print(f"Frais Totaux: ${metrics['total_fees']:.2f}")
    
    # Comparaison avec/sans Memory Decoder
    print("\n🔄 Comparaison avec système de base...")
    trader_base = IntegratedMemoryTrader({'use_memory_decoder': False, 'initial_capital': 1000})
    trader_base.rl_trader = trader.rl_trader  # Utiliser même modèle RL
    metrics_base = trader_base.evaluate(test_data_eval)
    
    print(f"\nSans Memory Decoder: Return = {metrics_base['total_return']:.2f}%")
    print(f"Avec Memory Decoder: Return = {metrics['total_return']:.2f}%")
    
    improvement = metrics['total_return'] - metrics_base['total_return']
    print(f"\n{'✅' if improvement > 0 else '❌'} Amélioration: {improvement:+.2f}%")
    
    print("\n✅ Test terminé avec succès!")

if __name__ == "__main__":
    main()
