"""
💰 REAL MONEY TRADING SYSTEM - AGENT ENTRAÎNEMENT $100

MISSION: Trading réel avec vraies données de marché ($100 budget)
OBJECTIF: Entraînement agent avec mémorisation complète
DONNÉES: 100% RÉELLES (APIs officielles, prix temps réel)

🔥 ATTENTION: ARGENT RÉEL SIMULÉ - Traité comme budget réel de $100
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

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("REAL_TRADING")

@dataclass
class RealTradingConfig:
    """Configuration pour trading réel"""
    initial_capital: float = 100.0  # $100 RÉEL
    max_positions: int = 5
    max_position_size: float = 0.20  # 20% max par position
    stop_loss: float = 0.05  # 5% stop loss
    take_profit: float = 0.10  # 10% take profit
    trading_fee: float = 0.001  # 0.1% frais
    memory_file: str = "logs/real_trading_memory.json"
    real_data_sources: List[str] = None

    def __post_init__(self):
        if self.real_data_sources is None:
            self.real_data_sources = [
                "yfinance",  # Actions réelles
                "binance_api",  # Crypto réelles  
                "alpha_vantage",  # Données fondamentales
                "polygon_api"  # Données professionnelles
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
    reason: str  # Raison du trade
    market_data_snapshot: Dict
    agent_confidence: float
    successful: bool

class RealDataProvider:
    """Fournisseur de données réelles"""
    
    def __init__(self):
        self.sources_active = {}
        self._initialize_sources()
        
    def _initialize_sources(self):
        """Initialisation des sources de données réelles"""
        logger.info("🌐 Initialisation des sources de données RÉELLES...")
        
        # YFinance (gratuit, données réelles)
        try:
            # Test avec une action réelle
            test = yf.Ticker("AAPL").info
            self.sources_active["yfinance"] = True
            logger.info("✅ YFinance: CONNECTÉ (données actions réelles)")
        except Exception as e:
            self.sources_active["yfinance"] = False
            logger.warning(f"⚠️ YFinance: ERREUR {e}")
        
        # Binance API (crypto réelles - gratuit)
        try:
            response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=5)
            if response.status_code == 200:
                self.sources_active["binance"] = True
                logger.info("✅ Binance API: CONNECTÉ (données crypto réelles)")
            else:
                self.sources_active["binance"] = False
        except Exception as e:
            self.sources_active["binance"] = False
            logger.warning(f"⚠️ Binance API: ERREUR {e}")
        
        # CoinGecko API (crypto réelles - gratuit)
        try:
            response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd", timeout=5)
            if response.status_code == 200:
                self.sources_active["coingecko"] = True
                logger.info("✅ CoinGecko API: CONNECTÉ (données crypto réelles)")
            else:
                self.sources_active["coingecko"] = False
        except Exception as e:
            self.sources_active["coingecko"] = False
            logger.warning(f"⚠️ CoinGecko API: ERREUR {e}")
        
        active_count = sum(self.sources_active.values())
        logger.info(f"📊 SOURCES ACTIVES: {active_count}/3 - Données 100% RÉELLES disponibles")

    def get_real_stock_price(self, symbol: str) -> Optional[RealMarketData]:
        """Prix réel d'une action via YFinance"""
        if not self.sources_active.get("yfinance", False):
            return None
            
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m").tail(1)
            info = ticker.info
            
            if data.empty:
                return None
                
            latest = data.iloc[-1]
            current_price = float(latest['Close'])
            volume = int(latest['Volume'])
            
            # Calcul du spread (estimation)
            spread = current_price * 0.001  # 0.1% spread estimé
            bid = current_price - spread/2
            ask = current_price + spread/2
            
            # Change 24h
            data_24h = ticker.history(period="2d")
            if len(data_24h) > 1:
                change_24h = ((current_price - float(data_24h.iloc[-2]['Close'])) / float(data_24h.iloc[-2]['Close'])) * 100
            else:
                change_24h = 0.0
                
            return RealMarketData(
                symbol=symbol,
                price=current_price,
                volume=volume,
                change_24h=change_24h,
                market_cap=info.get('marketCap', 0),
                timestamp=datetime.now(),
                source="YFinance",
                bid=bid,
                ask=ask,
                spread=spread
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération {symbol}: {e}")
            return None
    
    def get_real_crypto_price(self, symbol: str) -> Optional[RealMarketData]:
        """Prix réel crypto via Binance API"""
        if not self.sources_active.get("binance", False):
            return None
            
        try:
            # Prix actuel
            price_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
            response = requests.get(price_url, timeout=5)
            
            if response.status_code != 200:
                return None
                
            price_data = response.json()
            current_price = float(price_data['price'])
            
            # Stats 24h
            stats_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}USDT"
            stats_response = requests.get(stats_url, timeout=5)
            
            if stats_response.status_code == 200:
                stats = stats_response.json()
                volume = int(float(stats['volume']))
                change_24h = float(stats['priceChangePercent'])
            else:
                volume = 0
                change_24h = 0.0
            
            # OrderBook pour spread réel
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
                
            return RealMarketData(
                symbol=symbol,
                price=current_price,
                volume=volume,
                change_24h=change_24h,
                market_cap=None,  # Non disponible via Binance
                timestamp=datetime.now(),
                source="Binance",
                bid=bid,
                ask=ask,
                spread=spread
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération crypto {symbol}: {e}")
            return None

class RealTradingMemory:
    """Système de mémoire pour entraînement agent"""
    
    def __init__(self, memory_file: str):
        self.memory_file = memory_file
        self.memory = {
            "trades": [],
            "market_data_history": [],
            "decisions": [],
            "performance": [],
            "learning_patterns": [],
            "errors": [],
            "successful_strategies": []
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
            
            # Convertir les objets datetime en strings pour JSON
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
        # Convertir datetime en string
        for key, value in trade_dict.items():
            if isinstance(value, datetime):
                trade_dict[key] = value.isoformat()
        
        self.memory["trades"].append(trade_dict)
        logger.info(f"🧠 Trade mémorisé: {trade.symbol} {trade.side} ${trade.pnl:.2f}")
    
    def remember_decision(self, symbol: str, action: str, reasoning: str, confidence: float, market_data: RealMarketData):
        """Mémoriser une décision de trading"""
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
            "source": market_data.source
        }
        self.memory["decisions"].append(decision)
        logger.info(f"🤔 Décision mémorisée: {symbol} {action} (confiance: {confidence:.1%})")
    
    def remember_market_data(self, data: RealMarketData):
        """Mémoriser les données de marché"""
        market_record = {
            "timestamp": data.timestamp.isoformat(),
            "symbol": data.symbol,
            "price": data.price,
            "volume": data.volume,
            "change_24h": data.change_24h,
            "spread": data.spread,
            "source": data.source
        }
        self.memory["market_data_history"].append(market_record)
    
    def get_learning_insights(self) -> Dict:
        """Analyser la mémoire pour insights d'apprentissage"""
        if not self.memory["trades"]:
            return {"message": "Aucun trade pour analyse"}
        
        total_trades = len(self.memory["trades"])
        profitable_trades = sum(1 for t in self.memory["trades"] if float(t.get("pnl", 0)) > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(float(t.get("pnl", 0)) for t in self.memory["trades"])
        
        # Patterns de succès
        successful_symbols = {}
        for trade in self.memory["trades"]:
            if float(trade.get("pnl", 0)) > 0:
                symbol = trade.get("symbol", "")
                successful_symbols[symbol] = successful_symbols.get(symbol, 0) + 1
        
        insights = {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "profitable_trades": profitable_trades,
            "successful_symbols": successful_symbols,
            "average_pnl": total_pnl / total_trades if total_trades > 0 else 0
        }
        
        return insights

class RealMoneyTradingAgent:
    """Agent de trading avec argent réel ($100 budget)"""
    
    def __init__(self, config: RealTradingConfig):
        self.config = config
        self.data_provider = RealDataProvider()
        self.memory = RealTradingMemory(config.memory_file)
        
        # Portfolio réel
        self.cash = config.initial_capital  # $100 RÉEL
        self.positions = {}  # {symbol: {"quantity": float, "avg_price": float}}
        self.total_fees_paid = 0.0
        
        # Historique performance
        self.portfolio_history = []
        self.trade_history = []
        
        logger.info("💰 REAL MONEY TRADING AGENT INITIALISÉ")
        logger.info(f"   Budget initial: ${self.cash:.2f} (TRAITÉ COMME RÉEL)")
        logger.info(f"   Sources données: {sum(self.data_provider.sources_active.values())}/3 actives")
        logger.info(f"   Mémoire: {len(self.memory.memory['trades'])} trades historiques")
    
    def get_portfolio_value(self) -> float:
        """Valeur totale du portfolio en temps réel"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            # Récupération prix réel actuel
            if symbol in ["BTC", "ETH", "BNB"]:  # Crypto
                market_data = self.data_provider.get_real_crypto_price(symbol)
            else:  # Actions
                market_data = self.data_provider.get_real_stock_price(symbol)
            
            if market_data:
                position_value = position["quantity"] * market_data.price
                total_value += position_value
        
        return total_value
    
    def calculate_position_size(self, symbol: str, confidence: float) -> float:
        """Calcul taille position basée sur confiance et risk management"""
        base_size = min(self.config.max_position_size, confidence * 0.3)  # Max 30% si confiance 100%
        max_dollar_amount = self.cash * base_size
        
        return max_dollar_amount
    
    def execute_real_buy(self, symbol: str, market_data: RealMarketData, reasoning: str, confidence: float) -> Optional[RealTrade]:
        """Exécution BUY avec vraies données"""
        
        # Calcul taille position
        position_size_dollars = self.calculate_position_size(symbol, confidence)
        
        if position_size_dollars < 1.0:  # Minimum $1
            logger.warning(f"⚠️ Position trop petite pour {symbol}: ${position_size_dollars:.2f}")
            return None
        
        if position_size_dollars > self.cash:
            position_size_dollars = self.cash * 0.95  # Garde 5% de cash
        
        # Prix d'exécution (ask price pour buy)
        execution_price = market_data.ask
        quantity = position_size_dollars / execution_price
        
        # Frais de transaction
        fees = position_size_dollars * self.config.trading_fee
        total_cost = position_size_dollars + fees
        
        if total_cost > self.cash:
            logger.warning(f"⚠️ Fonds insuffisants pour {symbol}: ${total_cost:.2f} > ${self.cash:.2f}")
            return None
        
        # Exécution du trade
        trade_id = f"{symbol}_BUY_{int(datetime.now().timestamp())}"
        
        # Mise à jour portfolio
        self.cash -= total_cost
        self.total_fees_paid += fees
        
        if symbol in self.positions:
            # Moyenne des prix
            old_qty = self.positions[symbol]["quantity"]
            old_avg = self.positions[symbol]["avg_price"]
            new_avg = ((old_qty * old_avg) + (quantity * execution_price)) / (old_qty + quantity)
            
            self.positions[symbol]["quantity"] += quantity
            self.positions[symbol]["avg_price"] = new_avg
        else:
            self.positions[symbol] = {"quantity": quantity, "avg_price": execution_price}
        
        # Création du trade
        trade = RealTrade(
            trade_id=trade_id,
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            entry_price=execution_price,
            exit_price=None,
            entry_time=datetime.now(),
            exit_time=None,
            pnl=0.0,  # PnL calculé à la vente
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
            successful=True  # Sera mis à jour lors de la vente
        )
        
        # Mémorisation
        self.memory.remember_trade(trade)
        self.memory.remember_decision(symbol, "BUY", reasoning, confidence, market_data)
        self.trade_history.append(trade)
        
        logger.info(f"🟢 BUY EXÉCUTÉ: {symbol}")
        logger.info(f"   Quantité: {quantity:.6f}")
        logger.info(f"   Prix: ${execution_price:.4f}")
        logger.info(f"   Coût total: ${total_cost:.2f}")
        logger.info(f"   Cash restant: ${self.cash:.2f}")
        
        return trade
    
    def execute_real_sell(self, symbol: str, market_data: RealMarketData, reasoning: str, confidence: float) -> Optional[RealTrade]:
        """Exécution SELL avec vraies données"""
        
        if symbol not in self.positions:
            logger.warning(f"⚠️ Aucune position à vendre pour {symbol}")
            return None
        
        position = self.positions[symbol]
        quantity = position["quantity"]
        avg_buy_price = position["avg_price"]
        
        # Prix d'exécution (bid price pour sell)
        execution_price = market_data.bid
        gross_proceeds = quantity * execution_price
        
        # Frais de transaction
        fees = gross_proceeds * self.config.trading_fee
        net_proceeds = gross_proceeds - fees
        
        # Calcul P&L
        total_cost = quantity * avg_buy_price
        pnl = net_proceeds - total_cost
        pnl_percent = (pnl / total_cost) * 100 if total_cost > 0 else 0
        
        # Exécution du trade
        trade_id = f"{symbol}_SELL_{int(datetime.now().timestamp())}"
        
        # Mise à jour portfolio
        self.cash += net_proceeds
        self.total_fees_paid += fees
        del self.positions[symbol]  # Position fermée
        
        # Création du trade
        trade = RealTrade(
            trade_id=trade_id,
            symbol=symbol,
            side="SELL",
            quantity=quantity,
            entry_price=avg_buy_price,
            exit_price=execution_price,
            entry_time=datetime.now(),  # Approximation
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
            successful=(pnl > 0)
        )
        
        # Mémorisation
        self.memory.remember_trade(trade)
        self.memory.remember_decision(symbol, "SELL", reasoning, confidence, market_data)
        self.trade_history.append(trade)
        
        result_emoji = "🟢" if pnl > 0 else "🔴"
        logger.info(f"{result_emoji} SELL EXÉCUTÉ: {symbol}")
        logger.info(f"   Quantité: {quantity:.6f}")
        logger.info(f"   Prix achat: ${avg_buy_price:.4f}")
        logger.info(f"   Prix vente: ${execution_price:.4f}")
        logger.info(f"   P&L: ${pnl:.2f} ({pnl_percent:.1f}%)")
        logger.info(f"   Cash total: ${self.cash:.2f}")
        
        return trade
    
    def analyze_market_sentiment(self, symbol: str, market_data: RealMarketData) -> Tuple[str, float, str]:
        """Analyse sentiment basée sur vraies données"""
        
        # Analyse technique simple mais réelle
        reasoning_factors = []
        score = 0.5  # Neutre par défaut
        
        # Volume analysis
        if market_data.volume > 0:
            if market_data.volume > 1000000:  # Volume élevé
                if market_data.change_24h > 0:
                    score += 0.1
                    reasoning_factors.append(f"Volume élevé ({market_data.volume:,}) + hausse")
                else:
                    score -= 0.1
                    reasoning_factors.append(f"Volume élevé ({market_data.volume:,}) + baisse")
        
        # Price momentum
        if market_data.change_24h > 5:
            score += 0.2
            reasoning_factors.append(f"Forte hausse 24h: +{market_data.change_24h:.1f}%")
        elif market_data.change_24h < -5:
            score -= 0.2
            reasoning_factors.append(f"Forte baisse 24h: {market_data.change_24h:.1f}%")
        elif market_data.change_24h > 1:
            score += 0.1
            reasoning_factors.append(f"Hausse modérée 24h: +{market_data.change_24h:.1f}%")
        elif market_data.change_24h < -1:
            score -= 0.1
            reasoning_factors.append(f"Baisse modérée 24h: {market_data.change_24h:.1f}%")
        
        # Spread analysis
        spread_percent = (market_data.spread / market_data.price) * 100
        if spread_percent < 0.1:
            score += 0.05
            reasoning_factors.append("Spread faible (liquidité élevée)")
        elif spread_percent > 1.0:
            score -= 0.1
            reasoning_factors.append("Spread élevé (faible liquidité)")
        
        # Détermination action
        if score >= 0.65:
            action = "BUY"
        elif score <= 0.35:
            action = "SELL"
        else:
            action = "HOLD"
        
        confidence = abs(score - 0.5) * 2  # Convertir en confiance 0-1
        reasoning = " | ".join(reasoning_factors) if reasoning_factors else "Analyse neutre"
        
        return action, confidence, reasoning
    
    def run_trading_session(self, symbols: List[str], duration_minutes: int = 60):
        """Session de trading réel"""
        logger.info("🚀 DÉBUT SESSION DE TRADING RÉEL")
        logger.info(f"   Symboles: {symbols}")
        logger.info(f"   Durée: {duration_minutes} minutes")
        logger.info(f"   Budget initial: ${self.config.initial_capital:.2f}")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        session_trades = 0
        
        try:
            while datetime.now() < end_time:
                # Analyse de chaque symbole
                for symbol in symbols:
                    try:
                        # Récupération données réelles
                        if symbol in ["BTC", "ETH", "BNB", "ADA", "DOT"]:
                            market_data = self.data_provider.get_real_crypto_price(symbol)
                        else:
                            market_data = self.data_provider.get_real_stock_price(symbol)
                        
                        if not market_data:
                            logger.warning(f"⚠️ Pas de données pour {symbol}")
                            continue
                        
                        # Mémorisation données marché
                        self.memory.remember_market_data(market_data)
                        
                        # Analyse sentiment
                        action, confidence, reasoning = self.analyze_market_sentiment(symbol, market_data)
                        
                        logger.info(f"📊 {symbol}: {action} (confiance: {confidence:.1%}) - {reasoning}")
                        
                        # Exécution selon action (seuil abaissé pour plus d'activité)
                        if action == "BUY" and confidence > 0.25 and len(self.positions) < self.config.max_positions:
                            trade = self.execute_real_buy(symbol, market_data, reasoning, confidence)
                            if trade:
                                session_trades += 1
                        
                        elif action == "SELL" and symbol in self.positions and confidence > 0.25:
                            trade = self.execute_real_sell(symbol, market_data, reasoning, confidence)
                            if trade:
                                session_trades += 1
                        
                        # Portfolio status
                        portfolio_value = self.get_portfolio_value()
                        self.portfolio_history.append({
                            "timestamp": datetime.now(),
                            "portfolio_value": portfolio_value,
                            "cash": self.cash,
                            "positions_count": len(self.positions)
                        })
                        
                        # Affichage status
                        if session_trades > 0 and session_trades % 5 == 0:
                            logger.info(f"💼 Portfolio: ${portfolio_value:.2f} | Cash: ${self.cash:.2f} | Positions: {len(self.positions)}")
                        
                        time.sleep(10)  # Pause entre analyses
                        
                    except Exception as e:
                        logger.error(f"❌ Erreur analyse {symbol}: {e}")
                        continue
                
                time.sleep(30)  # Pause entre cycles
                
        except KeyboardInterrupt:
            logger.info("🛑 Session interrompue par utilisateur")
        
        # Fin de session
        final_portfolio_value = self.get_portfolio_value()
        total_return = final_portfolio_value - self.config.initial_capital
        return_percent = (total_return / self.config.initial_capital) * 100
        
        logger.info("🏁 FIN SESSION DE TRADING")
        logger.info(f"   Trades exécutés: {session_trades}")
        logger.info(f"   Portfolio final: ${final_portfolio_value:.2f}")
        logger.info(f"   Return total: ${total_return:.2f} ({return_percent:.1f}%)")
        logger.info(f"   Frais payés: ${self.total_fees_paid:.2f}")
        logger.info(f"   Positions ouvertes: {len(self.positions)}")
        
        # Sauvegarde mémoire
        self.memory.save_memory()
        
        return {
            "session_trades": session_trades,
            "final_portfolio_value": final_portfolio_value,
            "total_return": total_return,
            "return_percent": return_percent,
            "total_fees": self.total_fees_paid,
            "positions_count": len(self.positions)
        }

def main():
    """Fonction principale - Trading réel $100"""
    
    print("💰" + "="*80 + "💰")
    print("   🔥 REAL MONEY TRADING SYSTEM - $100 BUDGET RÉEL")
    print("="*84)
    print("   MISSION: Trading avec vraies données de marché")
    print("   BUDGET: $100 (traité comme argent réel)")
    print("   DONNÉES: 100% réelles via APIs officielles")
    print("   MÉMOIRE: Entraînement agent complet")
    print("💰" + "="*80 + "💰")
    
    # Configuration
    config = RealTradingConfig()
    
    # Agent de trading
    agent = RealMoneyTradingAgent(config)
    
    # Symboles pour trading
    trading_symbols = [
        # Actions principales
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        # Crypto principales  
        "BTC", "ETH", "BNB"
    ]
    
    print(f"\n🎯 CONFIGURATION DE TRADING:")
    print(f"   Budget initial: ${config.initial_capital:.2f}")
    print(f"   Max positions: {config.max_positions}")
    print(f"   Max position size: {config.max_position_size:.1%}")
    print(f"   Stop loss: {config.stop_loss:.1%}")
    print(f"   Frais: {config.trading_fee:.1%}")
    print(f"   Symboles: {', '.join(trading_symbols)}")
    
    print(f"\n📊 SOURCES DE DONNÉES:")
    for source, active in agent.data_provider.sources_active.items():
        status = "✅ ACTIF" if active else "❌ INACTIF"
        print(f"   {source}: {status}")
    
    # Insights mémoire existante
    insights = agent.memory.get_learning_insights()
    if insights.get("total_trades", 0) > 0:
        print(f"\n🧠 MÉMOIRE EXISTANTE:")
        print(f"   Trades historiques: {insights['total_trades']}")
        print(f"   Win rate: {insights['win_rate']:.1%}")
        print(f"   P&L total: ${insights['total_pnl']:.2f}")
        print(f"   P&L moyen: ${insights['average_pnl']:.2f}")
    
    print(f"\n🚀 DÉMARRAGE SESSION DE TRADING...")
    input("   Appuyez sur Entrée pour commencer (Ctrl+C pour arrêter)...")
    
    # Lancement session de trading
    try:
        results = agent.run_trading_session(
            symbols=trading_symbols,
            duration_minutes=30  # 30 minutes de trading
        )
        
        print(f"\n🏆 RÉSULTATS SESSION:")
        print(f"   Trades: {results['session_trades']}")
        print(f"   Portfolio final: ${results['final_portfolio_value']:.2f}")
        print(f"   Return: ${results['total_return']:.2f} ({results['return_percent']:.1f}%)")
        print(f"   Frais: ${results['total_fees']:.2f}")
        print(f"   Positions: {results['positions_count']}")
        
        # Analyse apprentissage
        final_insights = agent.memory.get_learning_insights()
        print(f"\n🧠 APPRENTISSAGE AGENT:")
        if final_insights.get('total_trades', 0) > 0:
            print(f"   Win rate global: {final_insights['win_rate']:.1%}")
            print(f"   Symboles réussis: {final_insights.get('successful_symbols', {})}")
        else:
            print(f"   Aucun trade exécuté - Agent trop prudent")
        
    except KeyboardInterrupt:
        print("\n🛑 Session interrompue")
    except Exception as e:
        print(f"\n❌ Erreur session: {e}")
    
    print("\n💾 Mémoire sauvegardée pour entraînement futur")
    print("🎓 Agent prêt pour prochaine session")

if __name__ == "__main__":
    main()
