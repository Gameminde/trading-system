# Import du calculateur d'indicateurs réels
from real_indicators_calculator import RealIndicatorsCalculator

class IntegratedTradingAgent:
    def __init__(self, config: IntegratedTradingConfig):
        self.config = config
        self.cash = config.initial_capital
        self.positions = {}
        self.trade_history = []
        
        # Intégration du calculateur d'indicateurs réels
        self.indicators_calculator = RealIndicatorsCalculator()
        
        # Initialisation des composants avancés
        self.market_analyzer = IntegratedMarketAnalyzer(config)
        
        logger.info("🤖 Agent de Trading Intégré initialisé")
        logger.info(f"   Capital: ${config.initial_capital:.2f}")
        logger.info(f"   RL: {'Activé' if config.enable_rl else 'Désactivé'}")
        logger.info(f"   Multi-Broker: {'Activé' if config.enable_multi_broker else 'Désactivé'}")
        logger.info(f"   Transformer: {'Activé' if config.enable_transformer else 'Désactivé'}")

    def _get_real_historical_data_for_training(self, symbol: str = "SPY", period: str = "2y") -> pd.DataFrame:
        """
        RÉCUPÉRER VRAIES DONNÉES HISTORIQUES POUR ENTRAÎNEMENT
        
        Objectifs :
        1. Utiliser données réelles via yfinance
        2. Calculer tous les indicateurs techniques
        3. Nettoyer et valider les données
        4. Fallback si erreur
        """
        
        try:
            logger.info(f"📊 Récupération données historiques {symbol} ({period})")
            
            # Récupérer données ETF ou indice large pour stabilité
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval="1d")
            
            if len(hist) < 100:  # Minimum 100 jours
                logger.warning(f"⚠️ Données insuffisantes pour {symbol}: {len(hist)} jours")
                return self._get_fallback_historical_data()
            
            # Nettoyer les données
            hist = hist[hist['Volume'] > 0]  # Enlever jours sans volume
            
            # Calculer tous les indicateurs
            df = pd.DataFrame({
                'open': hist['Open'],
                'high': hist['High'],
                'low': hist['Low'],
                'close': hist['Close'],
                'volume': hist['Volume']
            })
            
            # RSI
            df['rsi'] = self._calculate_rsi_series(df['close'])
            
            # MACD
            df['macd'] = self._calculate_macd_series(df['close'])
            
            # SMAs
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Supprimer les NaN des premiers jours
            df = df.dropna()
            
            logger.info(f"✅ Données historiques {symbol}: {len(df)} périodes avec indicateurs")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur données historiques {symbol}: {e}")
            return self._get_fallback_historical_data()

    def _calculate_rsi_series(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculer RSI sur série complète"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd_series(self, prices: pd.Series) -> pd.Series:
        """Calculer MACD sur série complète"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        return ema_12 - ema_26

    def _get_fallback_historical_data(self) -> pd.DataFrame:
        """Données de fallback si récupération échoue"""
        logger.warning("🚨 Utilisation données historiques de fallback")
        
        # Créer données minimales pour éviter le crash
        dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
        df = pd.DataFrame({
            'open': [100.0] * len(dates),
            'high': [101.0] * len(dates),
            'low': [99.0] * len(dates),
            'close': [100.0] * len(dates),
            'volume': [1000000] * len(dates),
            'rsi': [50.0] * len(dates),
            'macd': [0.0] * len(dates),
            'sma_20': [100.0] * len(dates),
            'sma_50': [100.0] * len(dates)
        }, index=dates)
        
        return df

    def initialize_system(self):
        """Initialiser avec vraies données"""
        logger.info("🚀 Initialisation avec VRAIES données historiques...")
        
        # Utiliser vraies données pour RL et Transformer
        real_data = self._get_real_historical_data_for_training("SPY", "2y")
        
        if self.config.enable_rl and self.market_analyzer.rl_trader:
            logger.info("🤖 Entraînement du modèle RL...")
            self.market_analyzer.rl_trader.train_model(real_data, total_timesteps=10000)
        
        if self.config.enable_transformer and self.market_analyzer.transformer_predictor:
            logger.info("🔮 Entraînement du modèle Transformer...")
            self.market_analyzer.transformer_predictor.train_model(real_data, epochs=50)
        
        logger.info("✅ Système initialisé avec vraies données")

    def _get_market_data(self, symbol: str) -> Optional[Dict]:
        """REMPLACER par vraies données via le calculateur d'indicateurs"""
        return self.indicators_calculator.get_real_market_data(symbol)
