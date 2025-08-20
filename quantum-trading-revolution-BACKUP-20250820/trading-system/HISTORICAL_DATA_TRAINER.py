"""
📊 HISTORICAL DATA TRAINER - ENTRAÎNEMENT INTELLIGENT AGENT

MISSION: Entraîner l'agent sur 5-10 ans de données historiques réelles
SOURCES: Bitcoin (2015-2025), Tesla (2020-2025), Apple, Ethereum, etc.
PATTERNS: Cycles bull/bear, corrélations macro, sentiment social
RÉSULTAT: Agent intelligent qui comprend les vrais patterns de marché

💎 ENFIN UN AGENT QUI APPREND DU MARCHÉ RÉEL !
"""

import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging
from typing import Dict, List, Tuple
import time
import pickle

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HISTORICAL_TRAINER")

class HistoricalDataTrainer:
    """Système d'entraînement sur données historiques"""
    
    def __init__(self):
        self.data_cache = {}
        self.market_regimes = {}
        self.correlation_patterns = {}
        self.sentiment_patterns = {}
        
        # Chemins de sauvegarde
        self.cache_dir = "data/historical_cache"
        self.models_dir = "models/trained"
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info("📊 Historical Data Trainer initialized")
    
    def collect_crypto_history(self, symbols: List[str] = ["BTC", "ETH", "BNB"], years: int = 10) -> Dict:
        """Collecte historique crypto 5-10 ans"""
        
        logger.info(f"📈 Collecting {years} years of crypto data...")
        crypto_data = {}
        
        # CoinGecko mapping
        coin_mapping = {
            "BTC": "bitcoin",
            "ETH": "ethereum", 
            "BNB": "binancecoin",
            "ADA": "cardano",
            "DOT": "polkadot",
            "LINK": "chainlink"
        }
        
        for symbol in symbols:
            if symbol not in coin_mapping:
                continue
                
            cache_file = f"{self.cache_dir}/{symbol}_history_{years}y.json"
            
            # Check cache first
            if os.path.exists(cache_file):
                logger.info(f"   📁 Loading {symbol} from cache")
                with open(cache_file, 'r') as f:
                    crypto_data[symbol] = json.load(f)
                continue
            
            try:
                # CoinGecko API (gratuit)
                coin_id = coin_mapping[symbol]
                days = years * 365
                
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': days,
                    'interval': 'daily'
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Structurer les données
                    structured_data = []
                    for i, (timestamp, price) in enumerate(data['prices']):
                        if i < len(data['total_volumes']) and i < len(data['market_caps']):
                            structured_data.append({
                                'date': datetime.fromtimestamp(timestamp/1000).isoformat(),
                                'price': price,
                                'volume': data['total_volumes'][i][1],
                                'market_cap': data['market_caps'][i][1]
                            })
                    
                    crypto_data[symbol] = structured_data
                    
                    # Cache pour éviter re-téléchargement
                    with open(cache_file, 'w') as f:
                        json.dump(structured_data, f, indent=2)
                    
                    logger.info(f"   ✅ {symbol}: {len(structured_data)} days collected")
                    
                else:
                    logger.error(f"   ❌ {symbol}: API error {response.status_code}")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"   ❌ {symbol}: {e}")
        
        return crypto_data
    
    def collect_stocks_history(self, symbols: List[str] = ["TSLA", "AAPL", "MSFT", "GOOGL"], years: int = 5) -> Dict:
        """Collecte historique actions 5 ans"""
        
        logger.info(f"📊 Collecting {years} years of stock data...")
        stocks_data = {}
        
        for symbol in symbols:
            cache_file = f"{self.cache_dir}/{symbol}_history_{years}y.json"
            
            # Check cache
            if os.path.exists(cache_file):
                logger.info(f"   📁 Loading {symbol} from cache")
                with open(cache_file, 'r') as f:
                    stocks_data[symbol] = json.load(f)
                continue
            
            try:
                # Yahoo Finance
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{years}y", interval="1d")
                
                if not hist.empty:
                    # Structurer les données
                    structured_data = []
                    for date, row in hist.iterrows():
                        structured_data.append({
                            'date': date.isoformat(),
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close']),
                            'volume': int(row['Volume']),
                            'price': float(row['Close'])  # Pour compatibilité
                        })
                    
                    stocks_data[symbol] = structured_data
                    
                    # Cache
                    with open(cache_file, 'w') as f:
                        json.dump(structured_data, f, indent=2)
                    
                    logger.info(f"   ✅ {symbol}: {len(structured_data)} days collected")
                
            except Exception as e:
                logger.error(f"   ❌ {symbol}: {e}")
        
        return stocks_data
    
    def collect_macro_data(self) -> Dict:
        """Collecte données macroéconomiques"""
        
        logger.info("🌍 Collecting macro economic data...")
        macro_data = {}
        
        cache_file = f"{self.cache_dir}/macro_data_5y.json"
        
        if os.path.exists(cache_file):
            logger.info("   📁 Loading macro data from cache")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        try:
            # VIX (Fear Index)
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="5y", interval="1d")
            
            if not vix_hist.empty:
                macro_data['vix'] = []
                for date, row in vix_hist.iterrows():
                    macro_data['vix'].append({
                        'date': date.isoformat(),
                        'value': float(row['Close'])
                    })
            
            # Dollar Index (DXY)
            dxy = yf.Ticker("DX-Y.NYB")
            dxy_hist = dxy.history(period="5y", interval="1d")
            
            if not dxy_hist.empty:
                macro_data['dxy'] = []
                for date, row in dxy_hist.iterrows():
                    macro_data['dxy'].append({
                        'date': date.isoformat(),
                        'value': float(row['Close'])
                    })
            
            # S&P 500
            sp500 = yf.Ticker("^GSPC")
            sp500_hist = sp500.history(period="5y", interval="1d")
            
            if not sp500_hist.empty:
                macro_data['sp500'] = []
                for date, row in sp500_hist.iterrows():
                    macro_data['sp500'].append({
                        'date': date.isoformat(),
                        'value': float(row['Close'])
                    })
            
            # Cache
            with open(cache_file, 'w') as f:
                json.dump(macro_data, f, indent=2)
            
            logger.info("   ✅ Macro data collected")
            
        except Exception as e:
            logger.error(f"   ❌ Macro data: {e}")
        
        return macro_data
    
    def identify_market_regimes(self, data: Dict) -> Dict:
        """Identifie les régimes de marché historiques"""
        
        logger.info("🔍 Identifying historical market regimes...")
        regimes = {}
        
        # Analyse Bitcoin pour régimes crypto
        if 'BTC' in data:
            btc_data = pd.DataFrame(data['BTC'])
            btc_data['date'] = pd.to_datetime(btc_data['date'])
            btc_data.set_index('date', inplace=True)
            
            # Calcul des rendements
            btc_data['returns'] = btc_data['price'].pct_change()
            btc_data['ma_20'] = btc_data['price'].rolling(20).mean()
            btc_data['ma_50'] = btc_data['price'].rolling(50).mean()
            
            # Identification des régimes
            regimes['crypto'] = []
            
            for i in range(50, len(btc_data)):
                current_price = btc_data['price'].iloc[i]
                ma_20 = btc_data['ma_20'].iloc[i]
                ma_50 = btc_data['ma_50'].iloc[i]
                
                if current_price > ma_20 > ma_50:
                    regime = "bull_market"
                elif current_price < ma_20 < ma_50:
                    regime = "bear_market"
                else:
                    regime = "sideways"
                
                regimes['crypto'].append({
                    'date': btc_data.index[i].isoformat(),
                    'regime': regime,
                    'price': current_price,
                    'ma_20': ma_20,
                    'ma_50': ma_50
                })
        
        # Analyse Tesla pour régimes tech stocks
        if 'TSLA' in data:
            tsla_data = pd.DataFrame(data['TSLA'])
            tsla_data['date'] = pd.to_datetime(tsla_data['date'])
            tsla_data.set_index('date', inplace=True)
            
            # Volatilité roulante
            tsla_data['returns'] = tsla_data['close'].pct_change()
            tsla_data['volatility'] = tsla_data['returns'].rolling(20).std() * np.sqrt(252)
            
            regimes['tech_stocks'] = []
            
            for i in range(20, len(tsla_data)):
                vol = tsla_data['volatility'].iloc[i]
                
                if vol > 0.8:
                    regime = "high_volatility"
                elif vol > 0.4:
                    regime = "medium_volatility"
                else:
                    regime = "low_volatility"
                
                regimes['tech_stocks'].append({
                    'date': tsla_data.index[i].isoformat(),
                    'regime': regime,
                    'volatility': vol,
                    'price': tsla_data['close'].iloc[i]
                })
        
        self.market_regimes = regimes
        
        # Sauvegarder
        with open(f"{self.models_dir}/market_regimes.json", 'w') as f:
            json.dump(regimes, f, indent=2)
        
        logger.info(f"   ✅ Market regimes identified: {len(regimes)} asset classes")
        
        return regimes
    
    def analyze_correlations(self, crypto_data: Dict, stocks_data: Dict, macro_data: Dict) -> Dict:
        """Analyse corrélations historiques"""
        
        logger.info("🔗 Analyzing historical correlations...")
        correlations = {}
        
        try:
            # Créer DataFrames alignés
            all_prices = {}
            
            # Crypto prices
            for symbol, data in crypto_data.items():
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                all_prices[f"{symbol}_crypto"] = df['price']
            
            # Stock prices
            for symbol, data in stocks_data.items():
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                all_prices[f"{symbol}_stock"] = df['close']
            
            # Macro data
            for indicator, data in macro_data.items():
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                all_prices[f"{indicator}_macro"] = df['value']
            
            # Créer DataFrame combiné
            combined_df = pd.DataFrame(all_prices)
            combined_df = combined_df.dropna()
            
            # Calcul corrélations
            corr_matrix = combined_df.corr()
            
            # Extraire corrélations intéressantes
            correlations['crypto_stocks'] = {}
            correlations['crypto_macro'] = {}
            correlations['stocks_macro'] = {}
            
            # BTC vs tout
            if 'BTC_crypto' in corr_matrix.columns:
                btc_corrs = corr_matrix['BTC_crypto'].to_dict()
                correlations['btc_correlations'] = {
                    k: v for k, v in btc_corrs.items() 
                    if k != 'BTC_crypto' and abs(v) > 0.3
                }
            
            # Tesla vs tout
            if 'TSLA_stock' in corr_matrix.columns:
                tsla_corrs = corr_matrix['TSLA_stock'].to_dict()
                correlations['tsla_correlations'] = {
                    k: v for k, v in tsla_corrs.items() 
                    if k != 'TSLA_stock' and abs(v) > 0.3
                }
            
            self.correlation_patterns = correlations
            
            # Sauvegarder
            with open(f"{self.models_dir}/correlations.json", 'w') as f:
                json.dump(correlations, f, indent=2, default=str)
            
            logger.info(f"   ✅ Correlations analyzed: {len(correlations)} patterns")
            
        except Exception as e:
            logger.error(f"   ❌ Correlation analysis: {e}")
        
        return correlations
    
    def extract_trading_patterns(self, crypto_data: Dict, stocks_data: Dict) -> Dict:
        """Extrait patterns de trading gagnants"""
        
        logger.info("🎯 Extracting winning trading patterns...")
        patterns = {}
        
        # Patterns crypto (basés sur Bitcoin)
        if 'BTC' in crypto_data:
            btc_df = pd.DataFrame(crypto_data['BTC'])
            btc_df['date'] = pd.to_datetime(btc_df['date'])
            btc_df.set_index('date', inplace=True)
            
            # Calculs techniques
            btc_df['returns'] = btc_df['price'].pct_change()
            btc_df['ma_7'] = btc_df['price'].rolling(7).mean()
            btc_df['ma_21'] = btc_df['price'].rolling(21).mean()
            btc_df['rsi'] = self._calculate_rsi(btc_df['price'])
            
            # Pattern: Golden Cross
            golden_crosses = []
            for i in range(21, len(btc_df)):
                if (btc_df['ma_7'].iloc[i] > btc_df['ma_21'].iloc[i] and 
                    btc_df['ma_7'].iloc[i-1] <= btc_df['ma_21'].iloc[i-1]):
                    
                    # Vérifier performance 30 jours après
                    if i + 30 < len(btc_df):
                        future_return = (btc_df['price'].iloc[i+30] - btc_df['price'].iloc[i]) / btc_df['price'].iloc[i]
                        golden_crosses.append({
                            'date': btc_df.index[i].isoformat(),
                            'entry_price': btc_df['price'].iloc[i],
                            'return_30d': future_return
                        })
            
            # Statistiques Golden Cross
            if golden_crosses:
                gc_returns = [gc['return_30d'] for gc in golden_crosses]
                patterns['golden_cross_btc'] = {
                    'total_signals': len(golden_crosses),
                    'avg_return': np.mean(gc_returns),
                    'win_rate': len([r for r in gc_returns if r > 0]) / len(gc_returns),
                    'best_return': max(gc_returns),
                    'worst_return': min(gc_returns)
                }
        
        # Patterns Tesla (basés sur volatilité)
        if 'TSLA' in stocks_data:
            tsla_df = pd.DataFrame(stocks_data['TSLA'])
            tsla_df['date'] = pd.to_datetime(tsla_df['date'])
            tsla_df.set_index('date', inplace=True)
            
            # Calculs
            tsla_df['returns'] = tsla_df['close'].pct_change()
            tsla_df['volatility'] = tsla_df['returns'].rolling(20).std() * np.sqrt(252)
            
            # Pattern: High volatility mean reversion
            vol_trades = []
            for i in range(20, len(tsla_df) - 10):
                if tsla_df['volatility'].iloc[i] > 1.0:  # High vol
                    # Entry après 3 jours de baisse
                    if all(tsla_df['returns'].iloc[i-j] < 0 for j in range(3)):
                        entry_price = tsla_df['close'].iloc[i]
                        exit_price = tsla_df['close'].iloc[i+10]  # Hold 10 days
                        trade_return = (exit_price - entry_price) / entry_price
                        
                        vol_trades.append({
                            'date': tsla_df.index[i].isoformat(),
                            'entry_price': entry_price,
                            'return_10d': trade_return,
                            'volatility': tsla_df['volatility'].iloc[i]
                        })
            
            if vol_trades:
                vol_returns = [vt['return_10d'] for vt in vol_trades]
                patterns['volatility_mean_reversion_tsla'] = {
                    'total_signals': len(vol_trades),
                    'avg_return': np.mean(vol_returns),
                    'win_rate': len([r for r in vol_returns if r > 0]) / len(vol_returns),
                    'best_return': max(vol_returns),
                    'worst_return': min(vol_returns)
                }
        
        # Sauvegarder patterns
        with open(f"{self.models_dir}/trading_patterns.json", 'w') as f:
            json.dump(patterns, f, indent=2, default=str)
        
        logger.info(f"   ✅ Trading patterns extracted: {len(patterns)} strategies")
        
        return patterns
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calcule RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def train_agent_with_history(self, agent_class, config_class) -> Dict:
        """Entraîne l'agent avec toutes les données historiques"""
        
        logger.info("🎓 Training agent with historical data...")
        
        # 1. Collecte toutes les données
        crypto_data = self.collect_crypto_history(["BTC", "ETH", "BNB"], years=5)
        stocks_data = self.collect_stocks_history(["TSLA", "AAPL", "MSFT", "GOOGL"], years=5)
        macro_data = self.collect_macro_data()
        
        # 2. Analyse des patterns
        regimes = self.identify_market_regimes({**crypto_data, **stocks_data})
        correlations = self.analyze_correlations(crypto_data, stocks_data, macro_data)
        patterns = self.extract_trading_patterns(crypto_data, stocks_data)
        
        # 3. Configuration agent optimisée
        trained_config = config_class()
        
        # Ajuster config basé sur patterns historiques
        if 'golden_cross_btc' in patterns:
            gc_pattern = patterns['golden_cross_btc']
            if gc_pattern['win_rate'] > 0.6:
                trained_config.sentiment_weight_minimum = 0.35  # Plus de sentiment si ça marche
        
        if 'volatility_mean_reversion_tsla' in patterns:
            vol_pattern = patterns['volatility_mean_reversion_tsla']
            if vol_pattern['win_rate'] > 0.55:
                trained_config.mps_weight_minimum = 0.40  # Plus de MPS pour mean reversion
        
        # 4. Créer agent avec config optimisée
        trained_agent = agent_class(trained_config)
        
        # 5. Injecter connaissances historiques
        trained_agent.historical_patterns = patterns
        trained_agent.market_regimes = regimes
        trained_agent.correlation_patterns = correlations
        
        # 6. Sauvegarder modèle entraîné
        training_results = {
            'training_date': datetime.now().isoformat(),
            'data_periods': {
                'crypto_years': 5,
                'stocks_years': 5,
                'macro_years': 5
            },
            'patterns_found': len(patterns),
            'regimes_identified': len(regimes),
            'correlations_analyzed': len(correlations),
            'config_optimizations': {
                'sentiment_weight': trained_config.sentiment_weight_minimum,
                'mps_weight': trained_config.mps_weight_minimum,
                'quantum_weight': trained_config.quantum_weight_minimum
            }
        }
        
        with open(f"{self.models_dir}/training_results.json", 'w') as f:
            json.dump(training_results, f, indent=2)
        
        # Sauvegarder agent (pickle)
        with open(f"{self.models_dir}/trained_agent.pkl", 'wb') as f:
            pickle.dump(trained_agent, f)
        
        logger.info("🎯 Agent training completed!")
        logger.info(f"   📊 Patterns found: {len(patterns)}")
        logger.info(f"   🔍 Regimes identified: {len(regimes)}")  
        logger.info(f"   🔗 Correlations analyzed: {len(correlations)}")
        
        return training_results
    
    def load_trained_agent(self):
        """Charge agent pré-entraîné"""
        
        try:
            with open(f"{self.models_dir}/trained_agent.pkl", 'rb') as f:
                agent = pickle.load(f)
            
            logger.info("📚 Trained agent loaded successfully")
            return agent
            
        except FileNotFoundError:
            logger.error("❌ No trained agent found. Run training first.")
            return None
    
    def generate_training_report(self) -> str:
        """Génère rapport d'entraînement"""
        
        report = "# 📊 HISTORICAL TRAINING REPORT\n\n"
        
        # Charger tous les résultats
        try:
            with open(f"{self.models_dir}/training_results.json", 'r') as f:
                results = json.load(f)
            
            with open(f"{self.models_dir}/trading_patterns.json", 'r') as f:
                patterns = json.load(f)
            
            report += f"## 🎯 TRAINING SUMMARY\n"
            report += f"- **Training Date:** {results['training_date']}\n"
            report += f"- **Data Period:** {results['data_periods']['crypto_years']} years crypto, {results['data_periods']['stocks_years']} years stocks\n"
            report += f"- **Patterns Found:** {results['patterns_found']}\n"
            report += f"- **Regimes Identified:** {results['regimes_identified']}\n\n"
            
            report += f"## 📈 WINNING PATTERNS\n"
            for pattern_name, pattern_data in patterns.items():
                if isinstance(pattern_data, dict) and 'win_rate' in pattern_data:
                    report += f"### {pattern_name.upper()}\n"
                    report += f"- **Win Rate:** {pattern_data['win_rate']:.1%}\n"
                    report += f"- **Average Return:** {pattern_data['avg_return']:.1%}\n"
                    report += f"- **Total Signals:** {pattern_data['total_signals']}\n"
                    report += f"- **Best Trade:** {pattern_data['best_return']:.1%}\n"
                    report += f"- **Worst Trade:** {pattern_data['worst_return']:.1%}\n\n"
            
            report += f"## ⚙️ OPTIMIZED CONFIG\n"
            config_opt = results['config_optimizations']
            report += f"- **Sentiment Weight:** {config_opt['sentiment_weight']:.0%}\n"
            report += f"- **MPS Weight:** {config_opt['mps_weight']:.0%}\n"
            report += f"- **Quantum Weight:** {config_opt['quantum_weight']:.0%}\n\n"
            
        except Exception as e:
            report += f"❌ Error loading training results: {e}\n"
        
        return report

def main():
    """Entraînement complet de l'agent"""
    
    print("📊" + "="*70 + "📊")
    print("   🎓 HISTORICAL DATA TRAINER - AGENT INTELLIGENT")
    print("   💎 Entraînement sur 5-10 ans de données réelles")
    print("   📈 Bitcoin • Ethereum • Tesla • Apple • Macro")
    print("="*74)
    
    trainer = HistoricalDataTrainer()
    
    print(f"\n🔄 Étapes d'entraînement :")
    print(f"   1. 📥 Collecte données historiques (BTC, ETH, TSLA, AAPL)")
    print(f"   2. 🔍 Identification régimes de marché")
    print(f"   3. 🔗 Analyse corrélations")
    print(f"   4. 🎯 Extraction patterns gagnants")
    print(f"   5. ⚙️ Configuration agent optimisée")
    print(f"   6. 💾 Sauvegarde modèle entraîné")
    
    choice = input(f"\n🚀 Lancer l'entraînement complet ? [y/N]: ").lower()
    
    if choice == 'y':
        try:
            # Import de l'agent optimisé
            from AGENT_OPTIMIZED_MAXIMUM_POWER import OptimizedTradingAgent, OptimizedConfig
            
            # Entraînement
            results = trainer.train_agent_with_history(OptimizedTradingAgent, OptimizedConfig)
            
            print(f"\n🎉 ENTRAÎNEMENT TERMINÉ !")
            print(f"   📊 Patterns trouvés: {results['patterns_found']}")
            print(f"   🔍 Régimes identifiés: {results['regimes_identified']}")
            print(f"   🔗 Corrélations analysées: {results['correlations_analyzed']}")
            
            # Générer rapport
            report = trainer.generate_training_report()
            
            with open("TRAINING_REPORT.md", 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"   📝 Rapport sauvé: TRAINING_REPORT.md")
            print(f"   💾 Agent entraîné: models/trained/trained_agent.pkl")
            
        except Exception as e:
            print(f"❌ Erreur entraînement: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Entraînement annulé")

if __name__ == "__main__":
    main()
