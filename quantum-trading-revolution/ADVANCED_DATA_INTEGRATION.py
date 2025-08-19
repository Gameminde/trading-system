"""
🚀 ADVANCED DATA INTEGRATION - APIs CRITIQUES
✅ Alpha Vantage API: Données financières avancées
✅ Federal Reserve API: Données macro-économiques
🎯 Objectif: +15-25% précision des signaux trading
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import asyncio
import aiohttp

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("ADVANCED_DATA_INTEGRATION")

@dataclass
class APIConfig:
    """Configuration des APIs critiques"""
    alpha_vantage_key: str = "Y2EO4DVTTPMTDTMB"
    federal_reserve_key: str = "400836fda21776d4c9ba5efb6ab0a389"
    
    # Alpha Vantage endpoints
    alpha_vantage_base: str = "https://www.alphavantage.co/query"
    
    # Federal Reserve endpoints
    fed_base: str = "https://api.stlouisfed.org/fred/series/observations"
    
    # Rate limiting
    alpha_vantage_rate_limit: int = 5  # 5 calls per minute (free tier)
    fed_rate_limit: int = 120  # 120 calls per minute

class AlphaVantageIntegration:
    """Intégration Alpha Vantage - Données financières avancées"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.last_call_time = 0
        self.call_count = 0
        self.rate_limit_reset = time.time() + 60
        
    def _rate_limit_check(self):
        """Vérification du rate limiting Alpha Vantage"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time > self.rate_limit_reset:
            self.call_count = 0
            self.rate_limit_reset = current_time + 60
        
        if self.call_count >= 5:  # Free tier limit
            wait_time = self.rate_limit_reset - current_time
            if wait_time > 0:
                logger.warning(f"⚠️ Rate limit Alpha Vantage atteint. Attente {wait_time:.1f}s")
                time.sleep(wait_time)
                self.call_count = 0
                self.rate_limit_reset = time.time() + 60
        
        self.call_count += 1
    
    def get_real_time_quote(self, symbol: str) -> Optional[Dict]:
        """Récupère les données temps réel d'un symbole"""
        self._rate_limit_check()
        
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'Global Quote' in data:
                    quote = data['Global Quote']
                    return {
                        'symbol': quote.get('01. symbol'),
                        'price': float(quote.get('05. price', 0)),
                        'change': float(quote.get('09. change', 0)),
                        'change_percent': float(quote.get('10. change percent', '0').replace('%', '')),
                        'volume': int(quote.get('06. volume', 0)),
                        'market_cap': quote.get('07. market cap'),
                        'timestamp': datetime.now()
                    }
                else:
                    logger.warning(f"⚠️ Pas de données pour {symbol}: {data}")
                    return None
            else:
                logger.error(f"❌ Erreur API Alpha Vantage: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur récupération quote {symbol}: {e}")
            return None
    
    def get_technical_indicators(self, symbol: str, function: str = 'RSI', interval: str = 'daily', 
                                time_period: int = 14) -> Optional[Dict]:
        """Récupère les indicateurs techniques avancés"""
        self._rate_limit_check()
        
        try:
            params = {
                'function': function,
                'symbol': symbol,
                'interval': interval,
                'time_period': time_period,
                'series_type': 'close',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if function in data:
                    # Extraire les dernières valeurs
                    time_series = data[function]
                    if isinstance(time_series, dict):
                        # Prendre les 2 dernières valeurs pour calculer la tendance
                        dates = sorted(time_series.keys())[-2:]
                        values = [float(time_series[date]) for date in dates]
                        
                        return {
                            'function': function,
                            'symbol': symbol,
                            'current_value': values[-1],
                            'previous_value': values[0] if len(values) > 1 else values[-1],
                            'trend': 'UP' if len(values) > 1 and values[-1] > values[0] else 'DOWN',
                            'timestamp': datetime.now()
                        }
                
                logger.warning(f"⚠️ Pas d'indicateur {function} pour {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur indicateur {function} pour {symbol}: {e}")
            return None
    
    def get_news_sentiment(self, tickers: str = None, topics: str = None) -> Optional[List[Dict]]:
        """Récupère les news et sentiment du marché"""
        self._rate_limit_check()
        
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'apikey': self.api_key
            }
            
            if tickers:
                params['tickers'] = tickers
            if topics:
                params['topics'] = topics
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'feed' in data:
                    news_items = []
                    for item in data['feed'][:10]:  # Limiter à 10 news
                        news_items.append({
                            'title': item.get('title', ''),
                            'summary': item.get('summary', ''),
                            'sentiment_score': float(item.get('overall_sentiment_score', 0)),
                            'sentiment_label': item.get('overall_sentiment_label', ''),
                            'ticker_sentiment': item.get('ticker_sentiment', []),
                            'time_published': item.get('time_published', ''),
                            'url': item.get('url', ''),
                            'timestamp': datetime.now()
                        })
                    
                    return news_items
                
                logger.warning("⚠️ Pas de news disponibles")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur news sentiment: {e}")
            return None
    
    def get_economic_calendar(self) -> Optional[List[Dict]]:
        """Récupère le calendrier économique"""
        self._rate_limit_check()
        
        try:
            params = {
                'function': 'ECONOMIC_CALENDAR',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'economic_calendar' in data:
                    events = []
                    for event in data['economic_calendar'][:20]:  # Limiter à 20 événements
                        events.append({
                            'event': event.get('event', ''),
                            'currency': event.get('currency', ''),
                            'importance': event.get('importance', ''),
                            'previous': event.get('previous', ''),
                            'forecast': event.get('forecast', ''),
                            'actual': event.get('actual', ''),
                            'timestamp': datetime.now()
                        })
                    
                    return events
                
                logger.warning("⚠️ Pas de calendrier économique disponible")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur calendrier économique: {e}")
            return None

class FederalReserveIntegration:
    """Intégration Federal Reserve - Données macro-économiques"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.last_call_time = 0
        self.call_count = 0
        self.rate_limit_reset = time.time() + 60
        
    def _rate_limit_check(self):
        """Vérification du rate limiting Fed"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time > self.rate_limit_reset:
            self.call_count = 0
            self.rate_limit_reset = current_time + 60
        
        if self.call_count >= 120:  # Fed limit
            wait_time = self.rate_limit_reset - current_time
            if wait_time > 0:
                logger.warning(f"⚠️ Rate limit Fed atteint. Attente {wait_time:.1f}s")
                time.sleep(wait_time)
                self.call_count = 0
                self.rate_limit_reset = time.time() + 60
        
        self.call_count += 1
    
    def get_interest_rates(self, series_id: str = 'FEDFUNDS') -> Optional[Dict]:
        """Récupère les taux d'intérêt (Fed Funds Rate par défaut)"""
        self._rate_limit_check()
        
        try:
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': 10
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'observations' in data:
                    observations = data['observations']
                    if observations:
                        latest = observations[0]
                        previous = observations[1] if len(observations) > 1 else observations[0]
                        
                        return {
                            'series_id': series_id,
                            'current_rate': float(latest.get('value', 0)),
                            'previous_rate': float(previous.get('value', 0)),
                            'change': float(latest.get('value', 0)) - float(previous.get('value', 0)),
                            'date': latest.get('date', ''),
                            'timestamp': datetime.now()
                        }
                
                logger.warning(f"⚠️ Pas de données pour {series_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur taux d'intérêt {series_id}: {e}")
            return None
    
    def get_unemployment_data(self) -> Optional[Dict]:
        """Récupère les données de chômage"""
        return self.get_interest_rates('UNRATE')
    
    def get_inflation_data(self) -> Optional[Dict]:
        """Récupère les données d'inflation (CPI)"""
        return self.get_interest_rates('CPIAUCSL')
    
    def get_gdp_data(self) -> Optional[Dict]:
        """Récupère les données de PIB"""
        return self.get_interest_rates('GDP')
    
    def get_employment_data(self) -> Optional[Dict]:
        """Récupère les données d'emploi (Non-Farm Payrolls)"""
        return self.get_interest_rates('PAYEMS')

class AdvancedDataAnalyzer:
    """Analyseur de données avancées combinant les APIs"""
    
    def __init__(self, alpha_vantage_key: str, fed_key: str):
        self.alpha_vantage = AlphaVantageIntegration(alpha_vantage_key)
        self.federal_reserve = FederalReserveIntegration(fed_key)
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    def _get_cached_data(self, key: str):
        """Récupère les données en cache"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.cache_duration:
                return data
        return None
    
    def _set_cached_data(self, key: str, data):
        """Met en cache les données"""
        self.cache[key] = (data, datetime.now())
    
    def get_comprehensive_market_analysis(self, symbol: str) -> Dict:
        """Analyse complète du marché combinant toutes les sources"""
        
        # Vérifier le cache
        cache_key = f"market_analysis_{symbol}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'technical_indicators': {},
            'news_sentiment': {},
            'macro_economic': {},
            'composite_score': 0.0,
            'recommendation': 'HOLD'
        }
        
        try:
            # 1. Données techniques Alpha Vantage
            logger.info(f"📊 Récupération données techniques pour {symbol}")
            
            # RSI
            rsi_data = self.alpha_vantage.get_technical_indicators(symbol, 'RSI')
            if rsi_data:
                analysis['technical_indicators']['rsi'] = rsi_data
            
            # MACD
            macd_data = self.alpha_vantage.get_technical_indicators(symbol, 'MACD')
            if macd_data:
                analysis['technical_indicators']['macd'] = macd_data
            
            # Bollinger Bands
            bb_data = self.alpha_vantage.get_technical_indicators(symbol, 'BBANDS')
            if bb_data:
                analysis['technical_indicators']['bollinger'] = bb_data
            
            # 2. News et sentiment
            logger.info(f"📰 Récupération news sentiment pour {symbol}")
            news_data = self.alpha_vantage.get_news_sentiment(tickers=symbol)
            if news_data:
                analysis['news_sentiment'] = {
                    'count': len(news_data),
                    'avg_sentiment': np.mean([item['sentiment_score'] for item in news_data]),
                    'recent_news': news_data[:3]  # 3 news récentes
                }
            
            # 3. Données macro-économiques
            logger.info("🏛️ Récupération données macro-économiques")
            
            # Taux d'intérêt
            fed_funds = self.federal_reserve.get_interest_rates()
            if fed_funds:
                analysis['macro_economic']['fed_funds'] = fed_funds
            
            # Chômage
            unemployment = self.federal_reserve.get_unemployment_data()
            if unemployment:
                analysis['macro_economic']['unemployment'] = unemployment
            
            # Inflation
            inflation = self.federal_reserve.get_inflation_data()
            if inflation:
                analysis['macro_economic']['inflation'] = inflation
            
            # 4. Calcul du score composite
            analysis['composite_score'] = self._calculate_composite_score(analysis)
            analysis['recommendation'] = self._generate_recommendation(analysis['composite_score'])
            
            # Mettre en cache
            self._set_cached_data(cache_key, analysis)
            
            logger.info(f"✅ Analyse complète générée pour {symbol}: Score {analysis['composite_score']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse complète {symbol}: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _calculate_composite_score(self, analysis: Dict) -> float:
        """Calcule le score composite basé sur tous les indicateurs"""
        score = 0.5  # Score neutre de départ
        weights = {
            'technical': 0.4,
            'sentiment': 0.3,
            'macro': 0.3
        }
        
        # Score technique (40%)
        technical_score = 0.0
        if 'rsi' in analysis['technical_indicators']:
            rsi = analysis['technical_indicators']['rsi']['current_value']
            if rsi < 30:
                technical_score += 0.3  # Survente
            elif rsi > 70:
                technical_score -= 0.3  # Surachat
            elif rsi < 40:
                technical_score += 0.15
            elif rsi > 60:
                technical_score -= 0.15
        
        if 'macd' in analysis['technical_indicators']:
            macd_trend = analysis['technical_indicators']['macd']['trend']
            if macd_trend == 'UP':
                technical_score += 0.2
            else:
                technical_score -= 0.2
        
        # Score sentiment (30%)
        sentiment_score = 0.0
        if analysis['news_sentiment']:
            avg_sentiment = analysis['news_sentiment']['avg_sentiment']
            if avg_sentiment > 0.6:
                sentiment_score += 0.3  # Sentiment très positif
            elif avg_sentiment > 0.4:
                sentiment_score += 0.15  # Sentiment positif
            elif avg_sentiment < 0.2:
                sentiment_score -= 0.3  # Sentiment très négatif
            elif avg_sentiment < 0.4:
                sentiment_score -= 0.15  # Sentiment négatif
        
        # Score macro (30%)
        macro_score = 0.0
        if 'fed_funds' in analysis['macro_economic']:
            fed_change = analysis['macro_economic']['fed_funds']['change']
            if fed_change < 0:  # Baisse des taux = positif pour marchés
                macro_score += 0.2
            elif fed_change > 0:  # Hausse des taux = négatif pour marchés
                macro_score -= 0.2
        
        if 'unemployment' in analysis['macro_economic']:
            unemployment_change = analysis['macro_economic']['unemployment']['change']
            if unemployment_change < 0:  # Baisse chômage = positif
                macro_score += 0.1
            elif unemployment_change > 0:  # Hausse chômage = négatif
                macro_score -= 0.1
        
        # Score final pondéré
        final_score = 0.5 + (
            technical_score * weights['technical'] +
            sentiment_score * weights['sentiment'] +
            macro_score * weights['macro']
        )
        
        return max(0.0, min(1.0, final_score))  # Plafonner entre 0 et 1
    
    def _generate_recommendation(self, score: float) -> str:
        """Génère une recommandation basée sur le score"""
        if score >= 0.65:
            return 'BUY'
        elif score <= 0.35:
            return 'SELL'
        else:
            return 'HOLD'
    
    def get_market_sentiment_summary(self) -> Dict:
        """Résumé du sentiment global du marché"""
        
        cache_key = "market_sentiment_summary"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # News sentiment global
            global_news = self.alpha_vantage.get_news_sentiment()
            
            # Données macro récentes
            fed_funds = self.federal_reserve.get_interest_rates()
            unemployment = self.federal_reserve.get_unemployment_data()
            
            summary = {
                'timestamp': datetime.now(),
                'global_sentiment': {
                    'news_count': len(global_news) if global_news else 0,
                    'avg_sentiment': np.mean([item['sentiment_score'] for item in global_news]) if global_news else 0.5
                },
                'macro_indicators': {
                    'fed_funds_rate': fed_funds['current_rate'] if fed_funds else None,
                    'unemployment_rate': unemployment['current_rate'] if unemployment else None
                },
                'market_condition': 'NEUTRAL'
            }
            
            # Déterminer la condition du marché
            if summary['global_sentiment']['avg_sentiment'] > 0.6:
                summary['market_condition'] = 'BULLISH'
            elif summary['global_sentiment']['avg_sentiment'] < 0.4:
                summary['market_condition'] = 'BEARISH'
            
            # Mettre en cache
            self._set_cached_data(cache_key, summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Erreur résumé sentiment: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}

def main():
    """Test des intégrations d'APIs"""
    
    print("🚀" + "="*80 + "🚀")
    print("   🔥 ADVANCED DATA INTEGRATION - APIs CRITIQUES")
    print("="*84)
    print("   ✅ Alpha Vantage API")
    print("   ✅ Federal Reserve API")
    print("   🎯 Objectif: +15-25% précision trading")
    print("🚀" + "="*80 + "🚀")
    
    # Configuration
    config = APIConfig()
    
    # Initialisation
    analyzer = AdvancedDataAnalyzer(config.alpha_vantage_key, config.federal_reserve_key)
    
    # Test des APIs
    print("\n🧪 TEST DES INTÉGRATIONS:")
    
    # 1. Test Alpha Vantage
    print("\n📊 Test Alpha Vantage - AAPL:")
    try:
        quote = analyzer.alpha_vantage.get_real_time_quote("AAPL")
        if quote:
            print(f"   Prix: ${quote['price']:.2f}")
            print(f"   Changement: {quote['change_percent']:+.2f}%")
            print(f"   Volume: {quote['volume']:,}")
        else:
            print("   ❌ Erreur récupération quote")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # 2. Test Federal Reserve
    print("\n🏛️ Test Federal Reserve - Fed Funds Rate:")
    try:
        fed_data = analyzer.federal_reserve.get_interest_rates()
        if fed_data:
            print(f"   Taux actuel: {fed_data['current_rate']:.2f}%")
            print(f"   Changement: {fed_data['change']:+.2f}%")
            print(f"   Date: {fed_data['date']}")
        else:
            print("   ❌ Erreur récupération taux Fed")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # 3. Test analyse complète
    print("\n🔍 Test Analyse Complète - AAPL:")
    try:
        analysis = analyzer.get_comprehensive_market_analysis("AAPL")
        if 'error' not in analysis:
            print(f"   Score composite: {analysis['composite_score']:.2f}")
            print(f"   Recommandation: {analysis['recommendation']}")
            print(f"   Indicateurs techniques: {len(analysis['technical_indicators'])}")
            print(f"   News sentiment: {len(analysis['news_sentiment'])}")
            print(f"   Données macro: {len(analysis['macro_economic'])}")
        else:
            print(f"   ❌ Erreur: {analysis['error']}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    print("\n✅ Tests terminés!")

if __name__ == "__main__":
    main()
