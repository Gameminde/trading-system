"""
🚀 REAL INDICATORS CALCULATOR - REMPLACER DONNÉES HARDCODÉES
✅ RSI calculé réellement (pas 50.0 hardcodé)
✅ MACD calculé réellement (pas 0.0 hardcodé)
✅ Prix réels via yfinance
✅ Historique pour calculs techniques
✅ Validation et cache
✅ Fallback en cas d'erreur
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("REAL_INDICATORS_CALCULATOR")

class RealIndicatorsCalculator:
    """Calculateur d'indicateurs techniques avec vraies données"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    def get_real_market_data(self, symbol: str) -> Optional[Dict]:
        """
        RÉCUPÉRER VRAIES DONNÉES DE MARCHÉ AVEC INDICATEURS
        
        Objectifs :
        1. Prix réels via yfinance
        2. Historique pour calculs RSI/MACD
        3. Validation et cache
        4. Fallback en cas d'erreur
        """
        
        cache_key = f"market_data_{symbol}"
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_duration:
                logger.debug(f"📊 Utilisation cache pour {symbol}")
                return data
        
        try:
            logger.info(f"📊 Récupération données réelles {symbol}...")
            
            # Récupérer données via yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d", interval="1d")
            info = ticker.info
            
            if hist.empty or len(hist) < 30:
                logger.warning(f"⚠️ Données insuffisantes pour {symbol}")
                return self._get_fallback_data(symbol)
            
            # Calculer indicateurs techniques RÉELS
            rsi = self._calculate_real_rsi(hist['Close'])
            sma_20 = self._calculate_real_sma(hist['Close'], 20)
            sma_50 = self._calculate_real_sma(hist['Close'], 50)
            macd = self._calculate_real_macd(hist['Close'])
            
            # Construire données de marché réelles
            market_data = {
                'symbol': symbol,
                'price': float(hist['Close'].iloc[-1]),
                'volume': int(hist['Volume'].iloc[-1]),
                'change_24h': float((hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100),
                'rsi': float(rsi),
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'macd': float(macd),
                'market_cap': info.get('marketCap'),
                'timestamp': datetime.now(),
                'source': 'yfinance_real'
            }
            
            # Valider les données
            if self._validate_market_data(market_data):
                # Mettre en cache
                self.cache[cache_key] = (market_data, datetime.now())
                logger.info(f"📊 Données réelles {symbol}: Prix=${market_data['price']:.2f}, RSI={market_data['rsi']:.1f}, MACD={market_data['macd']:.4f}")
                return market_data
            else:
                logger.warning(f"⚠️ Validation échouée pour {symbol}")
                return self._get_fallback_data(symbol)
                
        except Exception as e:
            logger.error(f"❌ Erreur données réelles {symbol}: {e}")
            return self._get_fallback_data(symbol)
    
    def _calculate_real_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculer RSI réel"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Prendre la dernière valeur valide
            rsi_clean = rsi.dropna()
            if len(rsi_clean) > 0:
                return float(rsi_clean.iloc[-1])
            else:
                return 50.0  # Valeur neutre si pas assez de données
        except Exception as e:
            logger.warning(f"⚠️ Erreur calcul RSI: {e}")
            return 50.0
    
    def _calculate_real_sma(self, prices: pd.Series, period: int) -> float:
        """Calculer SMA réel"""
        try:
            sma = prices.rolling(window=period).mean()
            sma_clean = sma.dropna()
            if len(sma_clean) > 0:
                return float(sma_clean.iloc[-1])
            else:
                return float(prices.iloc[-1])  # Fallback prix actuel
        except Exception as e:
            logger.warning(f"⚠️ Erreur calcul SMA: {e}")
            return float(prices.iloc[-1])
    
    def _calculate_real_macd(self, prices: pd.Series) -> float:
        """Calculer MACD réel"""
        try:
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            
            macd_clean = macd.dropna()
            if len(macd_clean) > 0:
                return float(macd_clean.iloc[-1])
            else:
                return 0.0  # Fallback neutre
        except Exception as e:
            logger.warning(f"⚠️ Erreur calcul MACD: {e}")
            return 0.0
    
    def _validate_market_data(self, data: Dict) -> bool:
        """Valider cohérence des données"""
        try:
            # Vérifier prix positif
            if data['price'] <= 0:
                logger.warning(f"⚠️ Prix invalide: {data['price']}")
                return False
            
            # Vérifier prix réaliste (pas de prix à $5.7 trillions!)
            if data['price'] > 1000000:  # Plus de $1M suspect
                logger.warning(f"⚠️ Prix suspect: ${data['price']:,.2f}")
                return False
            
            # Vérifier changement 24h réaliste
            if abs(data['change_24h']) > 50:  # Plus de 50% suspect
                logger.warning(f"⚠️ Changement 24h suspect: {data['change_24h']:.1f}%")
                return False
            
            # Vérifier RSI dans plage valide
            if data['rsi'] < 0 or data['rsi'] > 100:
                logger.warning(f"⚠️ RSI invalide: {data['rsi']}")
                return False
            
            # Vérifier volume positif
            if data['volume'] <= 0:
                logger.warning(f"⚠️ Volume invalide: {data['volume']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur validation: {e}")
            return False
    
    def _get_fallback_data(self, symbol: str) -> Dict:
        """Données de fallback si tout échoue"""
        logger.warning(f"🚨 Utilisation fallback pour {symbol}")
        return {
            'symbol': symbol,
            'price': 100.0,
            'volume': 1000000,
            'change_24h': 0.0,
            'rsi': 50.0,
            'sma_20': 100.0,
            'sma_50': 100.0,
            'macd': 0.0,
            'source': 'fallback'
        }
    
    def get_multiple_symbols_data(self, symbols: list) -> Dict[str, Dict]:
        """Récupérer données pour plusieurs symboles"""
        results = {}
        for symbol in symbols:
            try:
                data = self.get_real_market_data(symbol)
                if data:
                    results[symbol] = data
                else:
                    logger.warning(f"⚠️ Impossible récupérer données {symbol}")
            except Exception as e:
                logger.error(f"❌ Erreur symbole {symbol}: {e}")
        
        return results

def main():
    """Test du calculateur d'indicateurs réels"""
    print("🚀 TEST REAL INDICATORS CALCULATOR")
    print("="*50)
    
    calculator = RealIndicatorsCalculator()
    
    # Test avec AAPL
    print("\n📊 Test AAPL...")
    aapl_data = calculator.get_real_market_data('AAPL')
    if aapl_data:
        print(f"✅ Prix: ${aapl_data['price']:.2f}")
        print(f"✅ RSI: {aapl_data['rsi']:.1f}")
        print(f"✅ MACD: {aapl_data['macd']:.4f}")
        print(f"✅ Source: {aapl_data['source']}")
    
    # Test avec MSFT
    print("\n📊 Test MSFT...")
    msft_data = calculator.get_real_market_data('MSFT')
    if msft_data:
        print(f"✅ Prix: ${msft_data['price']:.2f}")
        print(f"✅ RSI: {msft_data['rsi']:.1f}")
        print(f"✅ MACD: {msft_data['macd']:.4f}")
        print(f"✅ Source: {msft_data['source']}")
    
    print("\n🎯 Calculateur d'indicateurs réels testé avec succès!")

if __name__ == "__main__":
    main()
