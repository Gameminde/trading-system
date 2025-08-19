"""
🚀 MULTI-BROKER MANAGER - DIVERSIFICATION GÉOGRAPHIQUE
✅ Interactive Brokers + Alpaca + Yahoo Finance
🎯 Objectif: +15-25% diversification et meilleurs prix
"""

import yfinance as yf
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("MULTI_BROKER_MANAGER")

class YahooBroker:
    """Broker Yahoo Finance (gratuit, données temps réel)"""
    
    def __init__(self):
        self.name = "Yahoo Finance"
        self.connected = True
        self.rate_limit = 100  # Appels par minute
        self.last_call = 0
        
    def connect(self):
        """Yahoo Finance est toujours connecté"""
        self.connected = True
        logger.info(f"✅ {self.name} connecté")
        
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Récupérer le prix actuel"""
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_call < 60 / self.rate_limit:
                time.sleep(60 / self.rate_limit)
            
            ticker = yf.Ticker(symbol)
            price = ticker.info.get('regularMarketPrice')
            self.last_call = current_time
            
            if price:
                logger.info(f"📊 {self.name} - {symbol}: ${price:.2f}")
                return float(price)
            else:
                logger.warning(f"⚠️ {self.name} - Pas de prix pour {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"❌ {self.name} - Erreur prix {symbol}: {e}")
            return None
    
    def execute_trade(self, symbol: str, action: str, quantity: int) -> bool:
        """Simuler l'exécution d'un trade (Yahoo = lecture seule)"""
        logger.info(f"📝 {self.name} - Simulation {action} {quantity} {symbol}")
        return True  # Simulation réussie

class IBBroker:
    """Broker Interactive Brokers (connexion locale)"""
    
    def __init__(self):
        self.name = "Interactive Brokers"
        self.connected = False
        self.ib = None
        
    def connect(self):
        """Connecter à Interactive Brokers"""
        try:
            # Vérifier si IB Gateway est disponible
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', 7497))
            sock.close()
            
            if result == 0:
                # Port ouvert - essayer de se connecter
                try:
                    from ib_insync import IB, Stock, MarketOrder
                    self.ib = IB()
                    self.ib.connect('127.0.0.1', 7497, clientId=1, timeout=5)
                    self.connected = True
                    logger.info(f"✅ {self.name} connecté via IB Gateway")
                except ImportError:
                    logger.warning(f"⚠️ {self.name} - ib_insync non installé")
                except Exception as e:
                    logger.warning(f"⚠️ {self.name} - Connexion échouée: {e}")
            else:
                logger.warning(f"⚠️ {self.name} - Port 7497 fermé (IB Gateway non démarré)")
                
        except Exception as e:
            logger.warning(f"⚠️ {self.name} - Erreur connexion: {e}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Récupérer le prix actuel via IB"""
        if not self.connected or not self.ib:
            return None
            
        try:
            stock = Stock(symbol, 'SMART', 'USD')
            ticker = self.ib.reqMktData(stock, '', False, False)
            self.ib.sleep(1)  # Attendre les données
            
            if ticker.marketPrice():
                logger.info(f"📊 {self.name} - {symbol}: ${ticker.marketPrice():.2f}")
                return ticker.marketPrice()
            else:
                logger.warning(f"⚠️ {self.name} - Pas de prix IB pour {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"❌ {self.name} - Erreur prix {symbol}: {e}")
            return None
    
    def execute_trade(self, symbol: str, action: str, quantity: int) -> bool:
        """Exécuter un trade via IB"""
        if not self.connected or not self.ib:
            logger.warning(f"⚠️ {self.name} - Non connecté")
            return False
            
        try:
            stock = Stock(symbol, 'SMART', 'USD')
            order = MarketOrder(action.upper(), quantity)
            trade = self.ib.placeOrder(stock, order)
            
            # Attendre la confirmation
            self.ib.sleep(2)
            
            if trade.orderStatus.status == 'Filled':
                logger.info(f"🟢 {self.name} - {action} {quantity} {symbol} exécuté")
                return True
            else:
                logger.warning(f"⚠️ {self.name} - Trade non exécuté: {trade.orderStatus.status}")
                return False
                
        except Exception as e:
            logger.error(f"❌ {self.name} - Erreur trade {symbol}: {e}")
            return False

class AlpacaBroker:
    """Broker Alpaca (paper trading)"""
    
    def __init__(self):
        self.name = "Alpaca"
        self.connected = False
        self.api = None
        
        # Configuration par défaut (à remplacer par vos clés)
        self.api_key = "YOUR_ALPACA_KEY"
        self.secret_key = "YOUR_ALPACA_SECRET"
        self.base_url = "https://paper-api.alpaca.markets"
        
    def connect(self):
        """Connecter à Alpaca"""
        try:
            import alpaca_trade_api as tradeapi
            
            # Vérifier si les clés sont configurées
            if self.api_key == "YOUR_ALPACA_KEY":
                logger.warning(f"⚠️ {self.name} - Clés API non configurées")
                return
            
            self.api = tradeapi.REST(
                self.api_key,
                self.secret_key,
                base_url=self.base_url
            )
            
            # Vérifier la connexion
            account = self.api.get_account()
            if account.status == 'ACTIVE':
                self.connected = True
                logger.info(f"✅ {self.name} connecté - Compte: {account.status}")
                logger.info(f"   Balance: ${float(account.cash):,.2f}")
            else:
                logger.warning(f"⚠️ {self.name} - Compte non actif: {account.status}")
                
        except ImportError:
            logger.warning(f"⚠️ {self.name} - alpaca_trade_api non installé")
        except Exception as e:
            logger.warning(f"⚠️ {self.name} - Erreur connexion: {e}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Récupérer le prix actuel via Alpaca"""
        if not self.connected or not self.api:
            return None
            
        try:
            barset = self.api.get_barset(symbol, 'minute', limit=1)
            if symbol in barset and len(barset[symbol]) > 0:
                price = barset[symbol][0].c
                logger.info(f"📊 {self.name} - {symbol}: ${price:.2f}")
                return price
            else:
                logger.warning(f"⚠️ {self.name} - Pas de données pour {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"❌ {self.name} - Erreur prix {symbol}: {e}")
            return None
    
    def execute_trade(self, symbol: str, action: str, quantity: int) -> bool:
        """Exécuter un trade via Alpaca"""
        if not self.connected or not self.api:
            logger.warning(f"⚠️ {self.name} - Non connecté")
            return False
            
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=action.lower(),
                type='market',
                time_in_force='gtc'
            )
            
            # Attendre la confirmation
            time.sleep(2)
            
            if order.status in ['filled', 'partially_filled']:
                logger.info(f"🟢 {self.name} - {action} {quantity} {symbol} exécuté")
                return True
            else:
                logger.warning(f"⚠️ {self.name} - Trade non exécuté: {order.status}")
                return False
                
        except Exception as e:
            logger.error(f"❌ {self.name} - Erreur trade {symbol}: {e}")
            return False

class MultiBrokerManager:
    """Gestionnaire multi-broker pour diversification"""
    
    def __init__(self):
        self.brokers = {
            'yahoo': YahooBroker(),
            'interactive_brokers': IBBroker(),
            'alpaca': AlpacaBroker()
        }
        self.active_broker = 'yahoo'  # Par défaut
        self.connection_status = {}
        
        logger.info("🚀 Multi-Broker Manager initialisé")
        logger.info(f"   Brokers: {list(self.brokers.keys())}")
    
    def connect_brokers(self):
        """Connecter tous les brokers disponibles"""
        logger.info("🔌 Connexion aux brokers...")
        
        for name, broker in self.brokers.items():
            try:
                broker.connect()
                self.connection_status[name] = broker.connected
                
                if broker.connected:
                    logger.info(f"✅ {name} connecté")
                else:
                    logger.warning(f"⚠️ {name} non disponible")
                    
            except Exception as e:
                logger.error(f"❌ Erreur connexion {name}: {e}")
                self.connection_status[name] = False
        
        # Afficher le statut
        connected_count = sum(self.connection_status.values())
        logger.info(f"📊 Statut connexion: {connected_count}/{len(self.brokers)} brokers connectés")
    
    def get_best_price(self, symbol: str) -> Tuple[Optional[str], Optional[float]]:
        """Comparer prix entre brokers et retourner le meilleur"""
        prices = {}
        
        for name, broker in self.brokers.items():
            if self.connection_status.get(name, False):
                try:
                    price = broker.get_current_price(symbol)
                    if price and price > 0:
                        prices[name] = price
                except Exception as e:
                    logger.debug(f"⚠️ Erreur prix {name} pour {symbol}: {e}")
                    continue
        
        if not prices:
            logger.warning(f"⚠️ Aucun prix disponible pour {symbol}")
            return None, None
        
        # Retourner le meilleur prix (le plus bas pour achat)
        best_broker = min(prices, key=prices.get)
        best_price = prices[best_broker]
        
        logger.info(f"🏆 Meilleur prix {symbol}: {best_broker} à ${best_price:.2f}")
        return best_broker, best_price
    
    def execute_trade(self, symbol: str, action: str, quantity: int) -> bool:
        """Exécuter trade sur le meilleur broker disponible"""
        logger.info(f"🚀 Exécution {action} {quantity} {symbol}")
        
        # Essayer d'abord le broker actif
        active_broker = self.brokers.get(self.active_broker)
        if active_broker and self.connection_status.get(self.active_broker, False):
            try:
                success = active_broker.execute_trade(symbol, action, quantity)
                if success:
                    logger.info(f"✅ Trade exécuté via {self.active_broker}")
                    return True
            except Exception as e:
                logger.warning(f"⚠️ Erreur broker actif: {e}")
        
        # Fallback sur le premier broker disponible
        for name, broker in self.brokers.items():
            if self.connection_status.get(name, False) and name != self.active_broker:
                try:
                    success = broker.execute_trade(symbol, action, quantity)
                    if success:
                        logger.info(f"✅ Trade exécuté via {name} (fallback)")
                        return True
                except Exception as e:
                    logger.warning(f"⚠️ Erreur fallback {name}: {e}")
                    continue
        
        logger.error(f"❌ Aucun broker disponible pour exécuter le trade")
        return False
    
    def get_broker_status(self) -> Dict[str, bool]:
        """Retourner le statut de tous les brokers"""
        return self.connection_status.copy()
    
    def set_active_broker(self, broker_name: str):
        """Définir le broker actif par défaut"""
        if broker_name in self.brokers:
            self.active_broker = broker_name
            logger.info(f"🎯 Broker actif changé: {broker_name}")
        else:
            logger.warning(f"⚠️ Broker inconnu: {broker_name}")
    
    def get_available_brokers(self) -> List[str]:
        """Retourner la liste des brokers connectés"""
        return [name for name, status in self.connection_status.items() if status]

def main():
    """Test du système multi-broker"""
    print("🚀" + "="*80 + "🚀")
    print("   🔥 MULTI-BROKER MANAGER - DIVERSIFICATION GÉOGRAPHIQUE")
    print("="*84)
    print("   ✅ Yahoo Finance (gratuit)")
    print("   ✅ Interactive Brokers (local)")
    print("   ✅ Alpaca (paper trading)")
    print("   🎯 Objectif: +15-25% diversification")
    print("🚀" + "="*80 + "🚀")
    
    # Initialiser le gestionnaire
    manager = MultiBrokerManager()
    
    # Connecter les brokers
    print("\n🔌 Connexion aux brokers...")
    manager.connect_brokers()
    
    # Afficher le statut
    print("\n📊 Statut des brokers:")
    status = manager.get_broker_status()
    for name, connected in status.items():
        status_icon = "✅" if connected else "❌"
        print(f"   {status_icon} {name}: {'Connecté' if connected else 'Non connecté'}")
    
    # Test de récupération de prix
    test_symbols = ["AAPL", "MSFT", "GOOGL"]
    
    print(f"\n📊 Test récupération prix:")
    for symbol in test_symbols:
        best_broker, best_price = manager.get_best_price(symbol)
        if best_broker and best_price:
            print(f"   {symbol}: {best_broker} à ${best_price:.2f}")
        else:
            print(f"   {symbol}: ❌ Prix non disponible")
    
    # Test d'exécution de trade
    print(f"\n🚀 Test exécution trade:")
    success = manager.execute_trade("AAPL", "BUY", 1)
    if success:
        print("   ✅ Trade simulé réussi")
    else:
        print("   ❌ Trade simulé échoué")
    
    print("\n✅ Test multi-broker terminé!")

if __name__ == "__main__":
    main()
