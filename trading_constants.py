"""
Module de constantes nommées pour le trading algorithmique
Remplace tous les magic numbers par des constantes explicites
"""

class TradingConstants:
    """Constantes centralisées pour le trading algorithmique"""
    
    # ============================================================================
    # RSI THRESHOLDS (Relative Strength Index)
    # ============================================================================
    RSI_OVERSOLD = 30          # Niveau de survente
    RSI_OVERBOUGHT = 70        # Niveau de surachat
    RSI_NEUTRAL = 50           # Niveau neutre
    RSI_SIGNIFICANT_CHANGE = 5 # Changement significatif RSI
    
    # ============================================================================
    # RISK LEVELS (Niveaux de risque)
    # ============================================================================
    RISK_CRITICAL_THRESHOLD = 0.8    # Seuil critique (80%)
    RISK_HIGH_THRESHOLD = 0.6        # Seuil élevé (60%)
    RISK_MODERATE_THRESHOLD = 0.4    # Seuil modéré (40%)
    RISK_LOW_THRESHOLD = 0.2         # Seuil faible (20%)
    RISK_MINIMAL_THRESHOLD = 0.1     # Seuil minimal (10%)
    
    # ============================================================================
    # TRADING CONFIDENCE (Confiance des décisions)
    # ============================================================================
    MIN_CONFIDENCE_THRESHOLD = 0.08      # Confiance minimum (8%)
    HIGH_CONFIDENCE_THRESHOLD = 0.8      # Confiance élevée (80%)
    EXTREME_CONFIDENCE_THRESHOLD = 0.95  # Confiance extrême (95%)
    
    # ============================================================================
    # PORTFOLIO LIMITS (Limites du portfolio)
    # ============================================================================
    MAX_POSITION_CONCENTRATION = 0.5      # 50% max dans une position
    MAX_DAILY_LOSS_RATIO = 0.05          # 5% perte quotidienne max
    MAX_DRAWDOWN_LIMIT = 0.15            # 15% drawdown max
    MAX_POSITIONS_COUNT = 4              # Nombre max de positions
    POSITION_SIZE_BASE = 0.25            # Taille de base des positions (25%)
    
    # ============================================================================
    # QUANTUM SCORING (Scores quantiques)
    # ============================================================================
    QUANTUM_HIGH_SCORE = 0.6             # Score quantique élevé
    QUANTUM_LOW_SCORE = 0.4              # Score quantique faible
    QUANTUM_NEUTRAL_SCORE = 0.5          # Score quantique neutre
    QUANTUM_BOOST_FACTOR = 0.8           # Facteur de boost quantique
    
    # ============================================================================
    # PRICE CHANGE THRESHOLDS (Seuils de changement de prix)
    # ============================================================================
    SIGNIFICANT_PRICE_CHANGE = 0.01      # 1% changement significatif
    MAJOR_PRICE_CHANGE = 0.02            # 2% changement majeur
    EXTREME_PRICE_CHANGE = 0.05          # 5% changement extrême
    MIN_PRICE_CHANGE = 0.001             # 0.1% changement minimum
    
    # ============================================================================
    # PERFORMANCE MONITORING (Monitoring des performances)
    # ============================================================================
    HIGH_LATENCY_THRESHOLD_MS = 100      # 100ms latence élevée
    LOW_ACCURACY_THRESHOLD = 0.6         # 60% accuracy minimum
    ERROR_RATE_THRESHOLD = 0.05          # 5% taux d'erreur max
    THROUGHPUT_MIN_OPS_PER_SEC = 10      # 10 opérations/sec minimum
    
    # ============================================================================
    # MEMORY DECODER (Décodeur de mémoire)
    # ============================================================================
    MEMORY_SIZE_DEFAULT = 100000         # Taille mémoire par défaut
    K_NEIGHBORS_DEFAULT = 32             # Nombre de voisins k-NN
    MEMORY_WEIGHT_DEFAULT = 0.6          # Poids mémoire par défaut
    MEMORY_UPDATE_FREQUENCY = 100        # Fréquence mise à jour mémoire
    
    # ============================================================================
    # TRANSACTION COSTS (Coûts de transaction)
    # ============================================================================
    TRANSACTION_COST_THRESHOLD = 0.004   # 0.4% coût transaction
    SLIPPAGE_THRESHOLD = 0.001           # 0.1% slippage
    MIN_TRANSACTION_AMOUNT = 10.0        # Montant minimum transaction ($10)
    
    # ============================================================================
    # STOP LOSS & TAKE PROFIT (Arrêt de perte et prise de profit)
    # ============================================================================
    STOP_LOSS_DEFAULT = 0.03             # 3% stop loss par défaut
    TAKE_PROFIT_DEFAULT = 0.06           # 6% take profit par défaut
    TRAILING_STOP_ACTIVATION = 0.02      # 2% activation trailing stop
    
    # ============================================================================
    # VOLATILITY THRESHOLDS (Seuils de volatilité)
    # ============================================================================
    LOW_VOLATILITY_THRESHOLD = 0.02      # 2% volatilité faible
    HIGH_VOLATILITY_THRESHOLD = 0.05     # 5% volatilité élevée
    EXTREME_VOLATILITY_THRESHOLD = 0.10  # 10% volatilité extrême
    
    # ============================================================================
    # TIME CONSTANTS (Constantes temporelles)
    # ============================================================================
    CACHE_DURATION_DEFAULT = 300         # 5 minutes cache par défaut
    MAX_RUNTIME_DEFAULT = 45             # 45 secondes runtime max
    MAX_STEPS_DEFAULT = 300              # 300 étapes max par run
    PROGRESS_LOG_INTERVAL = 100          # Log progression tous les 100 steps
    
    # ============================================================================
    # FUSION WEIGHTS (Poids de fusion)
    # ============================================================================
    FUSION_WEIGHT_QUANTUM = 0.25         # Poids fusion quantique
    FUSION_WEIGHT_MEMORY = 0.25          # Poids fusion mémoire
    FUSION_WEIGHT_TRADING = 0.25         # Poids fusion trading
    FUSION_WEIGHT_PREDICTOR = 0.15       # Poids fusion prédicteur
    FUSION_WEIGHT_RISK = 0.10            # Poids fusion risque
    
    # ============================================================================
    # MARKET REGIME THRESHOLDS (Seuils de régime de marché)
    # ============================================================================
    BULL_MARKET_THRESHOLD = 0.6          # Seuil marché haussier
    BEAR_MARKET_THRESHOLD = 0.4          # Seuil marché baissier
    SIDEWAYS_MARKET_MIN = 0.4            # Minimum marché latéral
    SIDEWAYS_MARKET_MAX = 0.6            # Maximum marché latéral
    
    # ============================================================================
    # VALIDATION LIMITS (Limites de validation)
    # ============================================================================
    MIN_QUANTITY_VALIDATION = 0.001      # Quantité minimum validation
    MAX_QUANTITY_VALIDATION = 1000000    # Quantité maximum validation
    MIN_PRICE_VALIDATION = 0.01          # Prix minimum validation
    MAX_PRICE_VALIDATION = 1000000       # Prix maximum validation
    MAX_SYMBOL_LENGTH = 10               # Longueur maximum symbole
    MAX_CAPITAL_VALIDATION = 1000000000  # Capital maximum validation (1 milliard)
    
    # ============================================================================
    # ALERT THRESHOLDS (Seuils d'alerte)
    # ============================================================================
    HIGH_DRAWDOWN_ALERT = 0.10          # 10% drawdown = alerte
    CRITICAL_DRAWDOWN_ALERT = 0.15      # 15% drawdown = alerte critique
    LOW_CAPITAL_ALERT = 0.5              # 50% capital restant = alerte
    HIGH_ERROR_RATE_ALERT = 0.10         # 10% taux d'erreur = alerte
    
    # ============================================================================
    # NEURAL NETWORK PARAMETERS (Paramètres réseaux de neurones)
    # ============================================================================
    PREDICTOR_SEQUENCE_LENGTH = 30       # Longueur séquence prédicteur
    PREDICTOR_TRAIN_EPOCHS = 5           # Époques d'entraînement prédicteur
    PREDICTOR_LEARNING_RATE = 0.001      # Taux d'apprentissage prédicteur
    PREDICTOR_BATCH_SIZE = 32            # Taille batch prédicteur
    
    # ============================================================================
    # CACHE AND PERFORMANCE (Cache et performance)
    # ============================================================================
    INDICATORS_CACHE_SECONDS = 300       # Cache indicateurs (5 minutes)
    MEMORY_CACHE_SIZE = 1000             # Taille cache mémoire
    PERFORMANCE_METRICS_WINDOW = 100     # Fenêtre métriques performance
    ALERT_HISTORY_SIZE = 100             # Taille historique alertes
    
    # ============================================================================
    # ADDITIONAL CONSTANTS (Constantes additionnelles)
    # ============================================================================
    INITIAL_CAPITAL_DEFAULT = 1000.0     # Capital initial par défaut
    PERFORMANCE_CHECK_INTERVAL = 100     # Intervalle vérification performance
    QUANTUM_ITERATIONS_DEFAULT = 100     # Itérations quantiques par défaut
    MAX_RUNTIME_DEFAULT = 45             # Runtime maximum par défaut
    MAX_STEPS_PER_RUN_DEFAULT = 300      # Étapes maximum par run


# Constantes pour timeframes
class TimeFrames:
    """Constantes pour timeframes de trading"""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    
    VALID_TIME_FRAMES = {
        ONE_MINUTE, FIVE_MINUTES, FIFTEEN_MINUTES, THIRTY_MINUTES,
        ONE_HOUR, FOUR_HOURS, ONE_DAY, ONE_WEEK
    }


# Constantes pour actions de trading
class TradingActions:
    """Constantes pour actions de trading"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"
    
    VALID_ACTIONS = {BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL}


# Constantes pour régimes de marché
class MarketRegimes:
    """Constantes pour régimes de marché"""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    CRISIS = "CRISIS"
    
    VALID_REGIMES = {BULL, BEAR, SIDEWAYS, HIGH_VOLATILITY, CRISIS}


# Test des constantes
if __name__ == "__main__":
    print("🔧 Test des constantes de trading...")
    
    print(f"RSI Oversold: {TradingConstants.RSI_OVERSOLD}")
    print(f"Risk Critical: {TradingConstants.RISK_CRITICAL_THRESHOLD}")
    print(f"Max Position Concentration: {TradingConstants.MAX_POSITION_CONCENTRATION}")
    print(f"Trading Fee Rate: {TradingConstants.TRANSACTION_COST_THRESHOLD}")
    print(f"High Volatility Threshold: {TradingConstants.HIGH_VOLATILITY_THRESHOLD}")
    
    print(f"Valid Actions: {TradingActions.VALID_ACTIONS}")
    print(f"Valid Timeframes: {TimeFrames.VALID_TIME_FRAMES}")
    print(f"Valid Market Regimes: {MarketRegimes.VALID_REGIMES}")
    
    print("✅ Constantes de trading chargées avec succès!")
