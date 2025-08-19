"""
💰 REAL MONEY TRADING SYSTEM - VERSION CORRIGÉE
✅ Problèmes résolus :
- Seuils de décision ajustés
- Logique de scoring améliorée
- Indicateurs techniques ajoutés
- Confiance mieux calibrée
"""

import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("REAL_TRADING_FIXED")

@dataclass
class RealTradingConfig:
    """Configuration optimisée pour trading réel"""
    initial_capital: float = 100.0  # $100 RÉEL
    max_positions: int = 5
    max_position_size: float = 0.20  # 20% max par position
    stop_loss: float = 0.05  # 5% stop loss
    take_profit: float = 0.10  # 10% take profit
    trailing_stop: float = 0.03  # 3% trailing stop
    trading_fee: float = 0.001  # 0.1% frais
    
    # NOUVEAUX PARAMÈTRES OPTIMISÉS
    min_confidence_buy: float = 0.15  # Réduit de 0.25 à 0.15
    min_confidence_sell: float = 0.15  # Réduit de 0.25 à 0.15
    score_threshold_buy: float = 0.60  # Réduit de 0.65 à 0.60
    score_threshold_sell: float = 0.40  # Augmenté de 0.35 à 0.40
    
    memory_file: str = "logs/real_trading_memory_fixed.json"
    real_data_sources: List[str] = None

    def __post_init__(self):
        if self.real_data_sources is None:
            self.real_data_sources = [
                "yfinance",  # Actions réelles
                "binance_api",  # Crypto réelles  
                "alpha_vantage",  # Données fondamentales
            ]

@dataclass
class RealMarketData:
    """Données réelles de marché"""
    symbol: str
    price: float
    volume: int
    change_24h: float
    market_cap: Optional[float]
    timestamp: datetime
    source: str
    bid: float
    ask: float
    spread: float

@dataclass
class RealTrade:
    """Trade réel avec vraies données"""
    trade_id: str
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    pnl: float
    pnl_percent: float
    fees: float
    reason: str
    market_data_snapshot: Dict
    agent_confidence: float
    successful: bool

class TechnicalIndicators:
    """Indicateurs techniques pour analyse avancée"""
    
    def __init__(self):
        self.price_history = {}  # Historique des prix par symbole
        self.rsi_period = 14
        self.sma_short = 20
        self.sma_long = 50
        
    def update_price_history(self, symbol: str, price: float):
        """Met à jour l'historique des prix"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=100)
        self.price_history[symbol].append(price)
    
    def calculate_rsi(self, symbol: str) -> Optional[float]:
        """Calcule le RSI (Relative Strength Index)"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < self.rsi_period + 1:
            return None
        
        prices = list(self.price_history[symbol])[-self.rsi_period-1:]
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_sma(self, symbol: str, period: int) -> Optional[float]:
        """Calcule la moyenne mobile simple"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < period:
            return None
        
        prices = list(self.price_history[symbol])[-period:]
        return np.mean(prices)
    
    def calculate_ema(self, symbol: str, period: int) -> Optional[float]:
        """Calcule la moyenne mobile exponentielle"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < period:
            return None
        
        prices = list(self.price_history[symbol])[-period:]
        weights = np.exp(np.linspace(-1, 0, period))
        weights /= weights.sum()
        
        return np.dot(prices, weights)
    
    def calculate_bollinger_bands(self, symbol: str, period: int = 20, std_dev: int = 2) -> Optional[Dict]:
        """Calcule les bandes de Bollinger"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < period:
            return None
        
        prices = list(self.price_history[symbol])[-period:]
        sma = np.mean(prices)
        std = np.std(prices)
        
        return {
            'upper': sma + (std_dev * std),
            'middle': sma,
            'lower': sma - (std_dev * std),
            'current_price': prices[-1]
        }
    
    def calculate_macd(self, symbol: str) -> Optional[Dict]:
        """Calcule le MACD"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 26:
            return None
        
        ema_12 = self.calculate_ema(symbol, 12)
        ema_26 = self.calculate_ema(symbol, 26)
        
        if ema_12 is None or ema_26 is None:
            return None
        
        macd_line = ema_12 - ema_26
        
        return {
            'macd': macd_line,
            'signal': macd_line * 0.8,  # Approximation simplifiée
            'histogram': macd_line * 0.2
        }

class RealDataProvider:
    """Fournisseur de données réelles avec cache"""
    
    def __init__(self):
        self.sources_active = {}
        self.cache = {}  # Cache pour éviter trop d'appels API
        self.cache_duration = 60  # 60 secondes
        self._initialize_sources()
        
    def _initialize_sources(self):
        """Initialisation des sources de données réelles"""
        logger.info("🌐 Initialisation des sources de données RÉELLES...")
        
        try:
            test = yf.Ticker("AAPL").info
            self.sources_active["yfinance"] = True
            logger.info("✅ YFinance: CONNECTÉ")
        except Exception as e:
            self.sources_active["yfinance"] = False
            logger.error(f"❌ YFinance: {e}")
    
    def get_real_stock_price(self, symbol: str) -> Optional[RealMarketData]:
        """Récupère le prix réel d'une action via YFinance avec cache"""
        
        # Vérifier le cache
        cache_key = f"stock_{symbol}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_duration:
                return cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            if current_price == 0:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
            
            volume = info.get('volume', 0)
            prev_close = info.get('previousClose', current_price)
            change_24h = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
            
            bid = info.get('bid', current_price * 0.999)
            ask = info.get('ask', current_price * 1.001)
            spread = ask - bid
            
            market_data = RealMarketData(
                symbol=symbol,
                price=current_price,
                volume=volume,
                change_24h=change_24h,
                market_cap=info.get('marketCap'),
                timestamp=datetime.now(),
                source="YFinance",
                bid=bid,
                ask=ask,
                spread=spread
            )
            
            # Mettre en cache
            self.cache[cache_key] = (market_data, datetime.now())
            
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération stock {symbol}: {e}")
            return None
    
    def get_real_crypto_price(self, symbol: str) -> Optional[RealMarketData]:
        """Récupère le prix réel d'une crypto via Binance avec cache"""
        
        # Vérifier le cache
        cache_key = f"crypto_{symbol}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_duration:
                return cached_data
        
        try:
            ticker_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
            response = requests.get(ticker_url, timeout=5)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            current_price = float(data['price'])
            
            stats_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}USDT"
            stats_response = requests.get(stats_url, timeout=5)
            
            if stats_response.status_code == 200:
                stats = stats_response.json()
                volume = int(float(stats['volume']))
                change_24h = float(stats['priceChangePercent'])
            else:
                volume = 0
                change_24h = 0.0
            
            book_url = f"https://api.binance.com/api/v3/depth?symbol={symbol}USDT&limit=5"
            book_response = requests.get(book_url, timeout=5)
            
            if book_response.status_code == 200:
                book = book_response.json()
                if book['bids'] and book['asks']:
                    bid = float(book['bids'][0][0])
                    ask = float(book['asks'][0][0])
                    spread = ask - bid
                else:
                    bid = current_price * 0.999
                    ask = current_price * 1.001
                    spread = ask - bid
            else:
                bid = current_price * 0.999
                ask = current_price * 1.001  
                spread = ask - bid
                
            market_data = RealMarketData(
                symbol=symbol,
                price=current_price,
                volume=volume,
                change_24h=change_24h,
                market_cap=None,
                timestamp=datetime.now(),
                source="Binance",
                bid=bid,
                ask=ask,
                spread=spread
            )
            
            # Mettre en cache
            self.cache[cache_key] = (market_data, datetime.now())
            
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération crypto {symbol}: {e}")
            return None

class RealTradingMemory:
    """Système de mémoire amélioré pour l'agent"""
    
    def __init__(self, memory_file: str):
        self.memory_file = memory_file
        self.memory = {
            "trades": [],
            "market_data_history": [],
            "decisions": [],
            "performance": [],
            "learning_patterns": [],
            "errors": [],
            "successful_strategies": [],
            "technical_indicators": {}  # Nouveau : historique des indicateurs
        }
        self._load_memory()
        logger.info(f"🧠 Mémoire agent initialisée: {memory_file}")
    
    def _load_memory(self):
        """Charger la mémoire existante"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    saved_memory = json.load(f)
                    self.memory.update(saved_memory)
                logger.info(f"📚 Mémoire chargée: {len(self.memory['trades'])} trades historiques")
            except Exception as e:
                logger.warning(f"⚠️ Erreur chargement mémoire: {e}")
    
    def save_memory(self):
        """Sauvegarder la mémoire"""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            memory_copy = {}
            for key, value in self.memory.items():
                if isinstance(value, list):
                    memory_copy[key] = []
                    for item in value:
                        if isinstance(item, dict):
                            item_copy = {}
                            for k, v in item.items():
                                if isinstance(v, datetime):
                                    item_copy[k] = v.isoformat()
                                else:
                                    item_copy[k] = v
                            memory_copy[key].append(item_copy)
                        else:
                            memory_copy[key].append(item)
                else:
                    memory_copy[key] = value
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_copy, f, indent=2, ensure_ascii=False)
            logger.info("💾 Mémoire sauvegardée avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde mémoire: {e}")
    
    def remember_trade(self, trade: RealTrade):
        """Mémoriser un trade"""
        trade_dict = asdict(trade)
        for key, value in trade_dict.items():
            if isinstance(value, datetime):
                trade_dict[key] = value.isoformat()
        
        self.memory["trades"].append(trade_dict)
        logger.info(f"🧠 Trade mémorisé: {trade.symbol} {trade.side} ${trade.pnl:.2f}")
    
    def remember_decision(self, symbol: str, action: str, reasoning: str, confidence: float, market_data: RealMarketData, technical_indicators: Dict = None):
        """Mémoriser une décision de trading avec indicateurs techniques"""
        decision = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "reasoning": reasoning,
            "confidence": confidence,
            "market_price": market_data.price,
            "market_change_24h": market_data.change_24h,
            "market_volume": market_data.volume,
            "spread": market_data.spread,
            "source": market_data.source,
            "technical_indicators": technical_indicators or {}
        }
        self.memory["decisions"].append(decision)
        logger.info(f"🤔 Décision mémorisée: {symbol} {action} (confiance: {confidence:.1%})")

class RealMoneyTradingAgent:
    """Agent de trading amélioré avec logique corrigée"""
    
    def __init__(self, config: RealTradingConfig):
        self.config = config
        self.data_provider = RealDataProvider()
        self.memory = RealTradingMemory(config.memory_file)
        self.technical_indicators = TechnicalIndicators()
        
        # Portfolio
        self.cash = config.initial_capital
        self.positions = {}
        self.trade_history = []
        self.portfolio_history = []
        self.total_fees_paid = 0.0
        
        # Trailing stops
        self.position_high_prices = {}  # Pour trailing stop
        
        logger.info("🤖 Agent de Trading CORRIGÉ initialisé")
        logger.info(f"   Capital: ${config.initial_capital:.2f}")
        logger.info(f"   Seuils BUY/SELL: {config.score_threshold_buy:.2f}/{config.score_threshold_sell:.2f}")
        logger.info(f"   Confiance min: {config.min_confidence_buy:.2f}")
    
    def analyze_market_advanced(self, symbol: str, market_data: RealMarketData) -> Tuple[str, float, str]:
        """Analyse avancée avec indicateurs techniques et seuils optimisés"""
        
        # Mettre à jour l'historique des prix
        self.technical_indicators.update_price_history(symbol, market_data.price)
        
        score = 0.5
        reasoning_factors = []
        technical_data = {}
        
        # 1. ANALYSE TECHNIQUE (40% du score)
        technical_weight = 0.4
        technical_score = 0.0
        
        # RSI
        rsi = self.technical_indicators.calculate_rsi(symbol)
        if rsi is not None:
            technical_data['rsi'] = rsi
            if rsi < 30:
                technical_score += 0.3
                reasoning_factors.append(f"RSI survente ({rsi:.1f})")
            elif rsi > 70:
                technical_score -= 0.3
                reasoning_factors.append(f"RSI surachat ({rsi:.1f})")
            elif rsi < 40:
                technical_score += 0.15
                reasoning_factors.append(f"RSI bas ({rsi:.1f})")
            elif rsi > 60:
                technical_score -= 0.15
                reasoning_factors.append(f"RSI élevé ({rsi:.1f})")
        
        # Moving Averages
        sma_20 = self.technical_indicators.calculate_sma(symbol, 20)
        sma_50 = self.technical_indicators.calculate_sma(symbol, 50)
        
        if sma_20 is not None and sma_50 is not None:
            technical_data['sma_20'] = sma_20
            technical_data['sma_50'] = sma_50
            
            if market_data.price > sma_20 > sma_50:
                technical_score += 0.25
                reasoning_factors.append("Tendance haussière (MA)")
            elif market_data.price < sma_20 < sma_50:
                technical_score -= 0.25
                reasoning_factors.append("Tendance baissière (MA)")
            elif market_data.price > sma_20:
                technical_score += 0.1
                reasoning_factors.append("Au-dessus SMA20")
        
        # Bollinger Bands
        bb = self.technical_indicators.calculate_bollinger_bands(symbol)
        if bb is not None:
            technical_data['bollinger'] = bb
            if market_data.price < bb['lower']:
                technical_score += 0.2
                reasoning_factors.append("Sous bande Bollinger inférieure")
            elif market_data.price > bb['upper']:
                technical_score -= 0.2
                reasoning_factors.append("Au-dessus bande Bollinger supérieure")
        
        # MACD
        macd = self.technical_indicators.calculate_macd(symbol)
        if macd is not None:
            technical_data['macd'] = macd
            if macd['macd'] > 0 and macd['histogram'] > 0:
                technical_score += 0.15
                reasoning_factors.append("MACD positif")
            elif macd['macd'] < 0 and macd['histogram'] < 0:
                technical_score -= 0.15
                reasoning_factors.append("MACD négatif")
        
        score += technical_score * technical_weight
        
        # 2. ANALYSE DE MOMENTUM (30% du score)
        momentum_weight = 0.3
        momentum_score = 0.0
        
        if market_data.change_24h > 3:
            momentum_score += 0.4
            reasoning_factors.append(f"Fort momentum +{market_data.change_24h:.1f}%")
        elif market_data.change_24h > 1:
            momentum_score += 0.2
            reasoning_factors.append(f"Momentum positif +{market_data.change_24h:.1f}%")
        elif market_data.change_24h < -3:
            momentum_score -= 0.4
            reasoning_factors.append(f"Fort momentum négatif {market_data.change_24h:.1f}%")
        elif market_data.change_24h < -1:
            momentum_score -= 0.2
            reasoning_factors.append(f"Momentum négatif {market_data.change_24h:.1f}%")
        
        score += momentum_score * momentum_weight
        
        # 3. ANALYSE DE VOLUME (20% du score)
        volume_weight = 0.2
        volume_score = 0.0
        
        if market_data.volume > 0:
            # Volume élevé renforce le signal
            if market_data.volume > 1000000:
                if market_data.change_24h > 0:
                    volume_score += 0.3
                    reasoning_factors.append(f"Volume élevé + hausse")
                else:
                    volume_score -= 0.3
                    reasoning_factors.append(f"Volume élevé + baisse")
            elif market_data.volume > 500000:
                if market_data.change_24h > 0:
                    volume_score += 0.15
                    reasoning_factors.append(f"Bon volume + hausse")
                else:
                    volume_score -= 0.15
                    reasoning_factors.append(f"Bon volume + baisse")
        
        score += volume_score * volume_weight
        
        # 4. ANALYSE DE LIQUIDITÉ (10% du score)
        liquidity_weight = 0.1
        liquidity_score = 0.0
        
        spread_percent = (market_data.spread / market_data.price) * 100
        if spread_percent < 0.05:
            liquidity_score += 0.3
            reasoning_factors.append("Excellente liquidité")
        elif spread_percent < 0.1:
            liquidity_score += 0.15
            reasoning_factors.append("Bonne liquidité")
        elif spread_percent > 0.5:
            liquidity_score -= 0.3
            reasoning_factors.append("Faible liquidité")
        
        score += liquidity_score * liquidity_weight
        
        # DÉCISION FINALE avec seuils optimisés
        if score >= self.config.score_threshold_buy:
            action = "BUY"
        elif score <= self.config.score_threshold_sell:
            action = "SELL"
        else:
            action = "HOLD"
        
        # Calcul de confiance amélioré
        confidence = min(abs(score - 0.5) * 3, 1.0)  # Multiplicateur augmenté, plafonné à 1.0
        
        # Mémoriser la décision avec indicateurs
        self.memory.remember_decision(symbol, action, " | ".join(reasoning_factors), confidence, market_data, technical_data)
        
        reasoning = f"Score: {score:.2f} | " + " | ".join(reasoning_factors) if reasoning_factors else f"Score: {score:.2f}"
        
        return action, confidence, reasoning
    
    def calculate_position_size(self, symbol: str, confidence: float) -> float:
        """Calcul intelligent de la taille de position basé sur Kelly Criterion"""
        
        # Position de base selon la confiance
        if confidence > 0.7:
            base_size = self.config.max_position_size
        elif confidence > 0.5:
            base_size = self.config.max_position_size * 0.75
        elif confidence > 0.3:
            base_size = self.config.max_position_size * 0.5
        else:
            base_size = self.config.max_position_size * 0.25
        
        # Ajustement selon le nombre de positions ouvertes
        positions_adjustment = 1.0 - (len(self.positions) * 0.1)
        positions_adjustment = max(positions_adjustment, 0.5)
        
        # Taille finale
        position_size = base_size * positions_adjustment
        max_dollar_amount = self.cash * position_size
        
        return max(max_dollar_amount, 10.0)  # Minimum $10
    
    def execute_real_buy(self, symbol: str, market_data: RealMarketData, reasoning: str, confidence: float) -> Optional[RealTrade]:
        """Exécution BUY avec gestion améliorée"""
        
        position_size_dollars = self.calculate_position_size(symbol, confidence)
        
        if position_size_dollars < 10.0:
            logger.warning(f"⚠️ Position trop petite pour {symbol}: ${position_size_dollars:.2f}")
            return None
        
        if position_size_dollars > self.cash:
            position_size_dollars = self.cash * 0.95
        
        execution_price = market_data.ask
        quantity = position_size_dollars / execution_price
        
        fees = position_size_dollars * self.config.trading_fee
        total_cost = position_size_dollars + fees
        
        if total_cost > self.cash:
            logger.warning(f"⚠️ Fonds insuffisants pour {symbol}")
            return None
        
        trade_id = f"{symbol}_BUY_{int(datetime.now().timestamp())}"
        
        self.cash -= total_cost
        self.total_fees_paid += fees
        
        if symbol in self.positions:
            old_qty = self.positions[symbol]["quantity"]
            old_avg = self.positions[symbol]["avg_price"]
            new_avg = ((old_qty * old_avg) + (quantity * execution_price)) / (old_qty + quantity)
            
            self.positions[symbol]["quantity"] += quantity
            self.positions[symbol]["avg_price"] = new_avg
        else:
            self.positions[symbol] = {
                "quantity": quantity,
                "avg_price": execution_price,
                "stop_loss": execution_price * (1 - self.config.stop_loss),
                "take_profit": execution_price * (1 + self.config.take_profit)
            }
        
        # Initialiser le prix le plus haut pour trailing stop
        self.position_high_prices[symbol] = execution_price
        
        trade = RealTrade(
            trade_id=trade_id,
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            entry_price=execution_price,
            exit_price=None,
            entry_time=datetime.now(),
            exit_time=None,
            pnl=0.0,
            pnl_percent=0.0,
            fees=fees,
            reason=reasoning,
            market_data_snapshot={
                "price": market_data.price,
                "bid": market_data.bid,
                "ask": market_data.ask,
                "spread": market_data.spread,
                "volume": market_data.volume,
                "change_24h": market_data.change_24h,
                "source": market_data.source
            },
            agent_confidence=confidence,
            successful=True
        )
        
        self.memory.remember_trade(trade)
        self.trade_history.append(trade)
        
        logger.info(f"🟢 BUY EXÉCUTÉ: {symbol}")
        logger.info(f"   Quantité: {quantity:.6f}")
        logger.info(f"   Prix: ${execution_price:.4f}")
        logger.info(f"   Confiance: {confidence:.1%}")
        logger.info(f"   Cash restant: ${self.cash:.2f}")
        
        return trade
    
    def execute_real_sell(self, symbol: str, market_data: RealMarketData, reasoning: str, confidence: float) -> Optional[RealTrade]:
        """Exécution SELL avec gestion améliorée"""
        
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        quantity = position["quantity"]
        avg_buy_price = position["avg_price"]
        
        execution_price = market_data.bid
        gross_proceeds = quantity * execution_price
        
        fees = gross_proceeds * self.config.trading_fee
        net_proceeds = gross_proceeds - fees
        
        total_cost = quantity * avg_buy_price
        pnl = net_proceeds - total_cost
        pnl_percent = (pnl / total_cost) * 100 if total_cost > 0 else 0
        
        trade_id = f"{symbol}_SELL_{int(datetime.now().timestamp())}"
        
        self.cash += net_proceeds
        self.total_fees_paid += fees
        del self.positions[symbol]
        
        # Nettoyer le trailing stop
        if symbol in self.position_high_prices:
            del self.position_high_prices[symbol]
        
        trade = RealTrade(
            trade_id=trade_id,
            symbol=symbol,
            side="SELL",
            quantity=quantity,
            entry_price=avg_buy_price,
            exit_price=execution_price,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            pnl=pnl,
            pnl_percent=pnl_percent,
            fees=fees,
            reason=reasoning,
            market_data_snapshot={
                "price": market_data.price,
                "bid": market_data.bid,
                "ask": market_data.ask,
                "spread": market_data.spread,
                "volume": market_data.volume,
                "change_24h": market_data.change_24h,
                "source": market_data.source
            },
            agent_confidence=confidence,
            successful=pnl > 0
        )
        
        self.memory.remember_trade(trade)
        self.trade_history.append(trade)
        
        emoji = "🟢" if pnl > 0 else "🔴"
        logger.info(f"{emoji} SELL EXÉCUTÉ: {symbol}")
        logger.info(f"   Quantité: {quantity:.6f}")
        logger.info(f"   Prix: ${execution_price:.4f}")
        logger.info(f"   P&L: ${pnl:.2f} ({pnl_percent:.1f}%)")
        logger.info(f"   Cash total: ${self.cash:.2f}")
        
        return trade
    
    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[str]:
        """Vérifie les stop loss et take profit"""
        
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Mise à jour du trailing stop
        if symbol in self.position_high_prices:
            if current_price > self.position_high_prices[symbol]:
                self.position_high_prices[symbol] = current_price
                # Ajuster le stop loss (trailing)
                new_stop = current_price * (1 - self.config.trailing_stop)
                if new_stop > position["stop_loss"]:
                    position["stop_loss"] = new_stop
                    logger.info(f"📈 Trailing stop ajusté pour {symbol}: ${new_stop:.2f}")
        
        # Vérifier stop loss
        if current_price <= position["stop_loss"]:
            return "STOP_LOSS"
        
        # Vérifier take profit
        if current_price >= position["take_profit"]:
            return "TAKE_PROFIT"
        
        return None
    
    def get_portfolio_value(self) -> float:
        """Calcule la valeur totale du portfolio"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in ["BTC", "ETH", "BNB", "ADA", "DOT"]:
                market_data = self.data_provider.get_real_crypto_price(symbol)
            else:
                market_data = self.data_provider.get_real_stock_price(symbol)
            
            if market_data:
                position_value = position["quantity"] * market_data.price
                total_value += position_value
        
        return total_value
    
    def run_trading_session(self, symbols: List[str], duration_minutes: int = 60):
        """Session de trading améliorée avec gestion active"""
        
        logger.info("🚀 DÉBUT SESSION DE TRADING CORRIGÉE")
        logger.info(f"   Symboles: {symbols}")
        logger.info(f"   Durée: {duration_minutes} minutes")
        logger.info(f"   Budget: ${self.config.initial_capital:.2f}")
        logger.info(f"   Seuils: BUY>{self.config.score_threshold_buy:.2f}, SELL<{self.config.score_threshold_sell:.2f}")
        logger.info(f"   Confiance min: {self.config.min_confidence_buy:.2f}")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        session_trades = 0
        cycle_count = 0
        
        try:
            while datetime.now() < end_time:
                cycle_count += 1
                logger.info(f"\n📊 === CYCLE {cycle_count} ===")
                
                for symbol in symbols:
                    try:
                        # Récupération données réelles
                        if symbol in ["BTC", "ETH", "BNB", "ADA", "DOT"]:
                            market_data = self.data_provider.get_real_crypto_price(symbol)
                        else:
                            market_data = self.data_provider.get_real_stock_price(symbol)
                        
                        if not market_data:
                            continue
                        
                        # Vérifier stop loss / take profit pour positions existantes
                        if symbol in self.positions:
                            exit_signal = self.check_stop_loss_take_profit(symbol, market_data.price)
                            if exit_signal:
                                reason = f"{exit_signal}: Prix actuel ${market_data.price:.2f}"
                                trade = self.execute_real_sell(symbol, market_data, reason, 0.9)
                                if trade:
                                    session_trades += 1
                                    logger.info(f"⚡ {exit_signal} déclenché pour {symbol}")
                                continue
                        
                        # Analyse avancée du marché
                        action, confidence, reasoning = self.analyze_market_advanced(symbol, market_data)
                        
                        logger.info(f"📊 {symbol}: ${market_data.price:.2f} | {action} (conf: {confidence:.1%})")
                        logger.info(f"   Raison: {reasoning}")
                        
                        # Exécution selon action avec seuils optimisés
                        if action == "BUY" and confidence >= self.config.min_confidence_buy:
                            if len(self.positions) < self.config.max_positions:
                                trade = self.execute_real_buy(symbol, market_data, reasoning, confidence)
                                if trade:
                                    session_trades += 1
                        
                        elif action == "SELL" and confidence >= self.config.min_confidence_sell:
                            if symbol in self.positions:
                                trade = self.execute_real_sell(symbol, market_data, reasoning, confidence)
                                if trade:
                                    session_trades += 1
                        
                        time.sleep(2)  # Pause courte entre analyses
                        
                    except Exception as e:
                        logger.error(f"❌ Erreur analyse {symbol}: {e}")
                        continue
                
                # Affichage status portfolio
                portfolio_value = self.get_portfolio_value()
                pnl = portfolio_value - self.config.initial_capital
                pnl_percent = (pnl / self.config.initial_capital) * 100
                
                logger.info(f"\n💼 === STATUS PORTFOLIO ===")
                logger.info(f"   Valeur: ${portfolio_value:.2f}")
                logger.info(f"   P&L: ${pnl:.2f} ({pnl_percent:+.1f}%)")
                logger.info(f"   Cash: ${self.cash:.2f}")
                logger.info(f"   Positions: {len(self.positions)}")
                logger.info(f"   Trades session: {session_trades}")
                
                # Sauvegarder périodiquement
                if cycle_count % 5 == 0:
                    self.memory.save_memory()
                
                # Pause entre cycles (réduite pour plus d'activité)
                time.sleep(15)
                
        except KeyboardInterrupt:
            logger.info("🛑 Session interrompue par utilisateur")
        
        # Fin de session
        final_portfolio_value = self.get_portfolio_value()
        total_return = final_portfolio_value - self.config.initial_capital
        return_percent = (total_return / self.config.initial_capital) * 100
        
        # Calculer les statistiques
        winning_trades = [t for t in self.trade_history if t.pnl > 0]
        losing_trades = [t for t in self.trade_history if t.pnl < 0]
        
        logger.info("\n🏁 === FIN SESSION DE TRADING ===")
        logger.info(f"   Durée: {(datetime.now() - start_time).seconds // 60} minutes")
        logger.info(f"   Trades exécutés: {session_trades}")
        logger.info(f"   Trades gagnants: {len(winning_trades)}")
        logger.info(f"   Trades perdants: {len(losing_trades)}")
        if session_trades > 0:
            logger.info(f"   Win rate: {len(winning_trades)/session_trades*100:.1f}%")
        logger.info(f"   Portfolio final: ${final_portfolio_value:.2f}")
        logger.info(f"   Return total: ${total_return:.2f} ({return_percent:+.1f}%)")
        logger.info(f"   Frais payés: ${self.total_fees_paid:.2f}")
        logger.info(f"   Positions ouvertes: {len(self.positions)}")
        
        # Sauvegarder la mémoire finale
        self.memory.save_memory()
        
        return {
            "session_trades": session_trades,
            "final_portfolio_value": final_portfolio_value,
            "total_return": total_return,
            "return_percent": return_percent,
            "total_fees": self.total_fees_paid,
            "positions_count": len(self.positions),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades)/session_trades*100 if session_trades > 0 else 0
        }

def main():
    """Fonction principale - Trading réel corrigé"""
    
    print("💰" + "="*80 + "💰")
    print("   🔥 REAL MONEY TRADING SYSTEM - VERSION CORRIGÉE")
    print("="*84)
    print("   ✅ Seuils optimisés")
    print("   ✅ Indicateurs techniques ajoutés")
    print("   ✅ Gestion active des risques")
    print("   ✅ Trailing stop implémenté")
    print("💰" + "="*80 + "💰")
    
    # Configuration optimisée
    config = RealTradingConfig()
    
    # Agent de trading corrigé
    agent = RealMoneyTradingAgent(config)
    
    # Symboles pour trading
    trading_symbols = [
        # Actions principales
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        # Crypto principales  
        "BTC", "ETH", "BNB"
    ]
    
    print(f"\n🎯 CONFIGURATION OPTIMISÉE:")
    print(f"   Budget: ${config.initial_capital:.2f}")
    print(f"   Max positions: {config.max_positions}")
    print(f"   Seuils BUY/SELL: {config.score_threshold_buy:.2f}/{config.score_threshold_sell:.2f}")
    print(f"   Confiance min: {config.min_confidence_buy:.2f}")
    print(f"   Stop loss: {config.stop_loss:.1%}")
    print(f"   Take profit: {config.take_profit:.1%}")
    print(f"   Trailing stop: {config.trailing_stop:.1%}")
    
    print(f"\n📊 SYMBOLES DE TRADING:")
    for symbol in trading_symbols:
        print(f"   • {symbol}")
    
    print("\n" + "="*84)
    input("Appuyez sur ENTRÉE pour démarrer la session de trading...")
    print("="*84)
    
    # Lancer la session de trading
    results = agent.run_trading_session(trading_symbols, duration_minutes=60)
    
    print("\n📊 RÉSULTATS FINAUX:")
    print(f"   Trades: {results['session_trades']}")
    print(f"   Win rate: {results['win_rate']:.1f}%")
    print(f"   Return: ${results['total_return']:.2f} ({results['return_percent']:+.1f}%)")
    print(f"   Portfolio: ${results['final_portfolio_value']:.2f}")
    
    print("\n✅ Session terminée avec succès!")

if __name__ == "__main__":
    main()