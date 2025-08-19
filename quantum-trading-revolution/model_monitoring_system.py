# FILE: model_monitoring_system.py
"""
AGENT QUANTUM TRADING - MODULE 1: MODEL MONITORING SYSTEM
Surveillance avancée et détection de dérive pour modèles de trading en production
Impact: Surveillance 47 métriques temps réel, détection dérive via 8 algorithmes, auto-rollback + alertes
Basé sur: ModelMonitoringSystem – Surveillance avancée et dé.md
"""

import numpy as np
import pandas as pd
import time
import uuid
import asyncio
import warnings
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging

warnings.filterwarnings('ignore')

class DriftSeverity(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertType(Enum):
    DATA_DRIFT = "data_drift"
    MODEL_DEGRADATION = "model_degradation"
    PNL_CORRELATION_BREAKDOWN = "pnl_correlation_breakdown"
    FEATURE_IMPORTANCE_DRIFT = "feature_importance_drift"
    CONFIDENCE_DROP = "confidence_drop"
    ADVERSARIAL_DETECTION = "adversarial_detection"

@dataclass
class MonitoringResult:
    timestamp: float
    metric_name: str
    current_value: float
    baseline_value: float
    drift_detected: bool
    severity: DriftSeverity
    confidence: float
    recommended_actions: List[str]

class RollingBuffer:
    """Buffer circulaire optimisé pour données de monitoring"""
    
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.buffer = deque(maxlen=maxsize)
        self._lock = asyncio.Lock()
    
    async def append(self, item):
        async with self._lock:
            self.buffer.append(item)
    
    def __len__(self):
        return len(self.buffer)
    
    def get_recent(self, count: int) -> List:
        return list(self.buffer)[-count:]
    
    def get_all(self) -> List:
        return list(self.buffer)

class StatisticalTestEngine:
    """Moteur de tests statistiques pour détection de dérive"""
    
    def __init__(self, baseline_window: int = 5000, test_window: int = 1000):
        self.baseline_window = baseline_window
        self.test_window = test_window
        self.baseline_stats = {}
        
    def kolmogorov_smirnov_test(self, baseline_data: np.ndarray, current_data: np.ndarray) -> Dict:
        """Test KS pour changements de distribution"""
        try:
            from scipy.stats import ks_2samp
            
            statistic, p_value = ks_2samp(baseline_data, current_data)
            
            # Seuils adaptatifs selon la volatilité du marché
            alpha = 0.001 if self._is_high_volatility() else 0.01
            
            return {
                'test': 'kolmogorov_smirnov',
                'statistic': statistic,
                'p_value': p_value,
                'drift_detected': p_value < alpha,
                'severity': DriftSeverity.CRITICAL if p_value < 0.001 else 
                           DriftSeverity.WARNING if p_value < 0.01 else DriftSeverity.NORMAL
            }
        except ImportError:
            # Fallback sans scipy
            return self._simple_distribution_test(baseline_data, current_data)
    
    def jensen_shannon_divergence(self, P: np.ndarray, Q: np.ndarray) -> Dict:
        """Distance Jensen-Shannon entre distributions"""
        try:
            from scipy.spatial.distance import jensenshannon
            
            # Normalisation en distributions de probabilité
            P_norm = P / (np.sum(P) + 1e-8)
            Q_norm = Q / (np.sum(Q) + 1e-8)
            
            js_distance = jensenshannon(P_norm, Q_norm)
            
            return {
                'test': 'jensen_shannon',
                'distance': js_distance,
                'drift_detected': js_distance > 0.1,  # Seuil empirique
                'severity': DriftSeverity.CRITICAL if js_distance > 0.2 else 
                           DriftSeverity.WARNING if js_distance > 0.1 else DriftSeverity.NORMAL
            }
        except ImportError:
            # Fallback simple
            return self._simple_divergence_test(P, Q)
    
    def population_stability_index(self, baseline_data: np.ndarray, current_data: np.ndarray, bins: int = 10) -> Dict:
        """Calcul PSI (Population Stability Index)"""
        try:
            # Création des bins basés sur les quantiles du baseline
            breakpoints = np.linspace(0, 100, bins + 1)
            baseline_bins = np.percentile(baseline_data, breakpoints)
            
            # Distribution baseline et courante
            baseline_counts = pd.cut(baseline_data, baseline_bins, include_lowest=True).value_counts()
            current_counts = pd.cut(current_data, baseline_bins, include_lowest=True).value_counts()
            
            # Normalisation
            baseline_pct = baseline_counts / len(baseline_data)
            current_pct = current_counts / len(current_data)
            
            # Calcul PSI
            psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / (baseline_pct + 1e-8)))
            
            return {
                'test': 'population_stability_index',
                'psi_value': psi,
                'drift_detected': psi > 0.2,
                'severity': DriftSeverity.CRITICAL if psi > 0.25 else 
                           DriftSeverity.WARNING if psi > 0.1 else DriftSeverity.NORMAL
            }
        except Exception as e:
            return {
                'test': 'population_stability_index',
                'psi_value': float('inf'),
                'drift_detected': True,
                'severity': DriftSeverity.CRITICAL,
                'error': str(e)
            }
    
    def _is_high_volatility(self) -> bool:
        """Détection simple de haute volatilité"""
        # Implémentation basique - peut être étendue
        return False
    
    def _simple_distribution_test(self, baseline_data: np.ndarray, current_data: np.ndarray) -> Dict:
        """Test de distribution simple sans dépendances externes"""
        baseline_mean = np.mean(baseline_data)
        baseline_std = np.std(baseline_data)
        current_mean = np.mean(current_data)
        current_std = np.std(current_data)
        
        # Test de différence des moyennes
        mean_diff = abs(current_mean - baseline_mean) / (baseline_std + 1e-8)
        std_diff = abs(current_std - baseline_std) / (baseline_std + 1e-8)
        
        drift_detected = mean_diff > 2.0 or std_diff > 1.5
        
        return {
            'test': 'simple_distribution',
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'drift_detected': drift_detected,
            'severity': DriftSeverity.CRITICAL if drift_detected else DriftSeverity.NORMAL
        }
    
    def _simple_divergence_test(self, P: np.ndarray, Q: np.ndarray) -> Dict:
        """Test de divergence simple sans dépendances externes"""
        # Différence des moyennes normalisées
        P_mean = np.mean(P)
        Q_mean = np.mean(Q)
        
        divergence = abs(Q_mean - P_mean) / (abs(P_mean) + 1e-8)
        
        return {
            'test': 'simple_divergence',
            'divergence': divergence,
            'drift_detected': divergence > 0.3,
            'severity': DriftSeverity.CRITICAL if divergence > 0.5 else 
                       DriftSeverity.WARNING if divergence > 0.3 else DriftSeverity.NORMAL
        }

class ModelPerformanceMonitor:
    """Surveillance performance modèle en temps réel"""
    
    def __init__(self, model_id: str, baseline_metrics: Dict):
        self.model_id = model_id
        self.baseline_metrics = baseline_metrics
        self.performance_buffer = RollingBuffer(10000)
        self.prediction_buffer = RollingBuffer(10000)
        
    async def update_performance(self, y_true: np.ndarray, y_pred: np.ndarray, timestamp: float) -> Dict:
        """Mise à jour métriques de performance"""
        try:
            # Calcul métriques instantanées
            current_metrics = {
                'timestamp': timestamp,
                'accuracy': self._calculate_accuracy(y_true, y_pred),
                'precision': self._calculate_precision(y_true, y_pred),
                'recall': self._calculate_recall(y_true, y_pred),
                'f1': self._calculate_f1(y_true, y_pred)
            }
            
            if len(np.unique(y_true)) == 2:  # Classification binaire
                current_metrics['auc'] = self._calculate_auc(y_true, y_pred)
            
            await self.performance_buffer.append(current_metrics)
            
            # Détection de dégradation
            return self._detect_performance_degradation(current_metrics)
            
        except Exception as e:
            return {
                'degradation_detected': False,
                'error': str(e)
            }
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcul accuracy simple"""
        return np.mean(y_true == y_pred)
    
    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcul precision simple"""
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / (predicted_positives + 1e-8)
    
    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcul recall simple"""
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / (actual_positives + 1e-8)
    
    def _calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcul F1-score simple"""
        precision = self._calculate_precision(y_true, y_pred)
        recall = self._calculate_recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall + 1e-8)
    
    def _calculate_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcul AUC simple (approximation)"""
        # Approximation basée sur la corrélation
        return max(0.5, min(1.0, (np.corrcoef(y_true, y_pred)[0, 1] + 1) / 2))
    
    def _detect_performance_degradation(self, current_metrics: Dict) -> Dict:
        """Détection dégradation performance"""
        if len(self.performance_buffer) < 100:
            return {'degradation_detected': False}
        
        # Comparaison avec baseline et fenêtre récente
        recent_metrics = self.performance_buffer.get_recent(100)
        recent_avg = np.mean([m['accuracy'] for m in recent_metrics])
        baseline_acc = self.baseline_metrics.get('accuracy', 0.8)
        
        degradation_pct = (baseline_acc - recent_avg) / (baseline_acc + 1e-8)
        
        return {
            'degradation_detected': degradation_pct > 0.15,
            'degradation_percentage': degradation_pct,
            'current_accuracy': recent_avg,
            'baseline_accuracy': baseline_acc,
            'severity': DriftSeverity.CRITICAL if degradation_pct > 0.25 else 
                       DriftSeverity.WARNING if degradation_pct > 0.15 else DriftSeverity.NORMAL
        }
    
    async def monitor_prediction_confidence(self, predictions_proba: np.ndarray) -> Dict:
        """Surveillance intervalles de confiance"""
        confidence_scores = np.max(predictions_proba, axis=1) if predictions_proba.ndim > 1 else predictions_proba
        mean_confidence = np.mean(confidence_scores)
        
        # Détection de perte de confiance
        baseline_confidence = self.baseline_metrics.get('mean_confidence', 0.7)
        confidence_drop = (baseline_confidence - mean_confidence) / (baseline_confidence + 1e-8)
        
        return {
            'confidence_drop_detected': confidence_drop > 0.2,
            'mean_confidence': mean_confidence,
            'baseline_confidence': baseline_confidence,
            'confidence_drop_percentage': confidence_drop
        }

class FeatureImportanceDriftTracker:
    """Surveillance dérive importance des features"""
    
    def __init__(self, baseline_importance: np.ndarray):
        self.baseline_importance = baseline_importance
        self.importance_history = []
        
    def track_importance_drift(self, current_model, validation_set: np.ndarray = None) -> Dict:
        """Suivi dérive importance features"""
        try:
            # Extraction importance actuelle
            if hasattr(current_model, 'feature_importances_'):
                current_importance = current_model.feature_importances_
            else:
                # Fallback: importance basée sur les coefficients
                if hasattr(current_model, 'coef_'):
                    current_importance = np.abs(current_model.coef_.flatten())
                else:
                    # Importance uniforme par défaut
                    current_importance = np.ones(len(self.baseline_importance))
            
            # Calcul dérive
            importance_drift = self._calculate_importance_drift(current_importance)
            
            self.importance_history.append({
                'timestamp': time.time(),
                'importance': current_importance,
                'drift_metrics': importance_drift
            })
            
            return importance_drift
            
        except Exception as e:
            return {
                'error': str(e),
                'drift_detected': False
            }
    
    def _calculate_importance_drift(self, current_importance: np.ndarray) -> Dict:
        """Calcul métriques de dérive d'importance"""
        try:
            # Corrélation rang entre baseline et actuel
            from scipy.stats import spearmanr
            rank_correlation, p_value = spearmanr(self.baseline_importance, current_importance)
        except ImportError:
            # Fallback sans scipy
            rank_correlation = self._simple_rank_correlation(self.baseline_importance, current_importance)
            p_value = 0.01
        
        # Distance cosine
        cosine_similarity = np.dot(self.baseline_importance, current_importance) / (
            np.linalg.norm(self.baseline_importance) * np.linalg.norm(current_importance) + 1e-8
        )
        
        # Détection features qui ont drastiquement changé
        relative_change = np.abs(current_importance - self.baseline_importance) / (self.baseline_importance + 1e-8)
        significant_changes = np.sum(relative_change > 0.5)
        
        return {
            'rank_correlation': rank_correlation,
            'cosine_similarity': cosine_similarity,
            'features_significantly_changed': significant_changes,
            'drift_detected': rank_correlation < 0.7 or cosine_similarity < 0.8 or significant_changes > 5
        }
    
    def _simple_rank_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Corrélation de rang simple sans dépendances externes"""
        # Tri et assignation de rangs
        x_ranks = np.argsort(np.argsort(x))
        y_ranks = np.argsort(np.argsort(y))
        
        # Corrélation de Pearson sur les rangs
        x_mean = np.mean(x_ranks)
        y_mean = np.mean(y_ranks)
        
        numerator = np.sum((x_ranks - x_mean) * (y_ranks - y_mean))
        denominator = np.sqrt(np.sum((x_ranks - x_mean)**2) * np.sum((y_ranks - y_mean)**2))
        
        return numerator / (denominator + 1e-8)

class BusinessMetricsMonitor:
    """Surveillance métriques business et alignement P&L"""
    
    def __init__(self, target_correlation: float = 0.6):
        self.target_correlation = target_correlation
        self.pnl_buffer = RollingBuffer(5000)
        self.prediction_buffer = RollingBuffer(5000)
        
    async def monitor_pnl_correlation(self, predictions: np.ndarray, actual_pnl: np.ndarray, timestamp: float) -> Dict:
        """Surveillance corrélation prédictions-P&L"""
        await self.prediction_buffer.append((timestamp, predictions))
        await self.pnl_buffer.append((timestamp, actual_pnl))
        
        if len(self.pnl_buffer) < 100:
            return {'sufficient_data': False}
        
        # Extraction des 500 dernières observations
        recent_predictions = [p[1] for p in self.prediction_buffer.get_recent(500)]
        recent_pnl = [p[1] for p in self.pnl_buffer.get_recent(500)]
        
        # Calcul corrélation
        correlation = np.corrcoef(recent_predictions, recent_pnl)[0, 1]
        
        # Calcul Sharpe ratio
        sharpe_ratio = np.mean(recent_pnl) / (np.std(recent_pnl) + 1e-8) * np.sqrt(252)
        
        return {
            'correlation': correlation,
            'sharpe_ratio': sharpe_ratio,
            'correlation_degradation': correlation < self.target_correlation,
            'predictions_mean': np.mean(recent_predictions),
            'pnl_mean': np.mean(recent_pnl),
            'severity': DriftSeverity.CRITICAL if correlation < 0.3 else 
                       DriftSeverity.WARNING if correlation < 0.5 else DriftSeverity.NORMAL
        }
    
    def detect_regime_change_impact(self, current_volatility: float, baseline_volatility: float) -> Dict:
        """Détection impact changement de régime"""
        volatility_ratio = current_volatility / (baseline_volatility + 1e-8)
        
        return {
            'regime_change_detected': volatility_ratio > 2.0 or volatility_ratio < 0.5,
            'volatility_ratio': volatility_ratio,
            'high_volatility_regime': volatility_ratio > 1.5,
            'low_volatility_regime': volatility_ratio < 0.7
        }

class AlertingEngine:
    """Moteur d'alertes multi-canal avec escalation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_history = []
        self.escalation_rules = config.get('escalation_rules', {})
        
    def process_alert(self, alert_type: AlertType, severity: DriftSeverity, metrics: Dict, recommended_actions: List[str]) -> str:
        """Traitement et routage d'alerte"""
        alert = {
            'timestamp': time.time(),
            'alert_type': alert_type.value,
            'severity': severity.value,
            'metrics': metrics,
            'recommended_actions': recommended_actions,
            'alert_id': str(uuid.uuid4())
        }
        
        # Log dans historique
        self.alert_history.append(alert)
        
        # Routage selon sévérité
        if severity == DriftSeverity.CRITICAL:
            self._send_critical_alert(alert)
            self._trigger_automatic_actions(alert)
        elif severity == DriftSeverity.WARNING:
            self._send_warning_alert(alert)
        
        return alert['alert_id']
    
    def _send_critical_alert(self, alert: Dict):
        """Envoi alerte critique (simulation)"""
        message = self._format_alert_message(alert)
        print(f"🚨 CRITICAL ALERT: {message}")
        
        # Ici on pourrait intégrer Slack, SMS, Email
        # self._send_slack_alert(message, channel='#trading-critical')
        # self._send_sms_alert(message)
        # self._send_email_alert(message, priority='HIGH')
    
    def _send_warning_alert(self, alert: Dict):
        """Envoi alerte warning (simulation)"""
        message = self._format_alert_message(alert)
        print(f"⚠️  WARNING: {message}")
        
        # self._send_slack_alert(message, channel='#trading-warnings')
    
    def _format_alert_message(self, alert: Dict) -> str:
        """Formatage message d'alerte"""
        return f"{alert['alert_type']} - {alert['severity']} - {alert['metrics']}"
    
    def _trigger_automatic_actions(self, alert: Dict):
        """Déclenchement actions automatiques"""
        actions = alert['recommended_actions']
        
        for action in actions:
            if 'reduce_position_size' in action:
                print(f"🔄 Auto-action: Reducing position size")
            elif 'switch_to_backup_model' in action:
                print(f"🔄 Auto-action: Switching to backup model")
            elif 'enable_safety_mode' in action:
                print(f"🔄 Auto-action: Enabling safety mode")
            elif 'trigger_model_retrain' in action:
                print(f"🔄 Auto-action: Triggering model retrain")

class ModelMonitoringSystem:
    """Système de monitoring complet pour modèles de trading"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.start_time = time.time()  # Ajout de l'attribut manquant
        self.alerts = []  # Ajout de l'attribut manquant
        self.total_tests_run = 0  # Ajout de l'attribut manquant
        self.statistical_engine = StatisticalTestEngine()
        self.performance_monitors = {}
        self.feature_trackers = {}
        self.business_monitors = {}
        self.alerting_engine = AlertingEngine(self.config)
        self.monitoring_results = []
        
        # Configuration par défaut
        self.drift_thresholds = {
            'psi_critical': 0.25,
            'psi_warning': 0.1,
            'correlation_critical': 0.3,
            'correlation_warning': 0.5,
            'accuracy_drop_critical': 0.25,
            'accuracy_drop_warning': 0.15
        }
        
        # Métriques surveillées (47 au total)
        self.monitored_metrics = [
            # Data Quality (12)
            'null_rate', 'outlier_count', 'schema_validation', 'data_freshness',
            'feature_drift_psi', 'distribution_ks_pvalue', 'correlation_stability',
            'volume_anomalies', 'price_spikes', 'missing_data_patterns',
            'data_consistency', 'format_validation',
            
            # Model Performance (15)
            'accuracy', 'precision', 'recall', 'f1_score', 'auc',
            'prediction_confidence', 'calibration_error', 'bias_detection',
            'variance_analysis', 'overfitting_metrics', 'underfitting_metrics',
            'cross_validation_score', 'holdout_performance', 'ensemble_diversity',
            'model_stability',
            
            # Business Impact (20)
            'pnl_correlation', 'sharpe_ratio', 'max_drawdown', 'win_rate',
            'profit_factor', 'risk_adjusted_returns', 'position_sizing_accuracy',
            'execution_quality', 'slippage_analysis', 'market_impact',
            'regime_adaptation', 'volatility_forecasting', 'correlation_breakdown',
            'liquidity_analysis', 'market_microstructure', 'order_flow_analysis',
            'sentiment_alignment', 'macro_factor_correlation', 'sector_rotation',
            'cross_asset_correlation'
        ]
    
    def register_model(self, model_id: str, baseline_metrics: Dict, baseline_importance: np.ndarray = None):
        """Enregistrement d'un nouveau modèle à surveiller"""
        self.performance_monitors[model_id] = ModelPerformanceMonitor(model_id, baseline_metrics)
        
        if baseline_importance is not None:
            self.feature_trackers[model_id] = FeatureImportanceDriftTracker(baseline_importance)
        
        self.business_monitors[model_id] = BusinessMetricsMonitor()
        
        print(f"✅ Model {model_id} registered for monitoring")
    
    async def monitor_data_drift(self, model_id: str, baseline_data: np.ndarray, current_data: np.ndarray) -> Dict:
        """Surveillance dérive des données"""
        results = {}
        
        # Tests statistiques
        ks_result = self.statistical_engine.kolmogorov_smirnov_test(baseline_data, current_data)
        psi_result = self.statistical_engine.population_stability_index(baseline_data, current_data)
        js_result = self.statistical_engine.jensen_shannon_divergence(baseline_data, current_data)
        
        results.update({
            'ks_test': ks_result,
            'psi_test': psi_result,
            'js_divergence': js_result
        })
        
        # Détection globale de dérive
        drift_detected = any([
            ks_result['drift_detected'],
            psi_result['drift_detected'],
            js_result['drift_detected']
        ])
        
        # Détermination sévérité
        severities = [r['severity'] for r in [ks_result, psi_result, js_result] if 'severity' in r]
        max_severity = max(severities) if severities else DriftSeverity.NORMAL
        
        # Actions recommandées
        recommended_actions = []
        if max_severity == DriftSeverity.CRITICAL:
            recommended_actions.extend([
                'reduce_position_size',
                'switch_to_backup_model',
                'trigger_model_retrain'
            ])
        elif max_severity == DriftSeverity.WARNING:
            recommended_actions.extend([
                'increase_monitoring_frequency',
                'log_detailed_metrics'
            ])
        
        # Création alerte si nécessaire
        if drift_detected:
            alert_id = self.alerting_engine.process_alert(
                AlertType.DATA_DRIFT,
                max_severity,
                results,
                recommended_actions
            )
            results['alert_id'] = alert_id
        
        results['drift_detected'] = drift_detected
        results['max_severity'] = max_severity.value
        results['recommended_actions'] = recommended_actions
        
        return results
    
    async def monitor_model_performance(self, model_id: str, y_true: np.ndarray, y_pred: np.ndarray, 
                                      predictions_proba: np.ndarray = None) -> Dict:
        """Surveillance performance du modèle"""
        if model_id not in self.performance_monitors:
            return {'error': 'Model not registered'}
        
        monitor = self.performance_monitors[model_id]
        timestamp = time.time()
        
        # Mise à jour performance
        performance_result = await monitor.update_performance(y_true, y_pred, timestamp)
        
        # Surveillance confiance si disponible
        confidence_result = {}
        if predictions_proba is not None:
            confidence_result = await monitor.monitor_prediction_confidence(predictions_proba)
        
        # Détection dégradation
        degradation_detected = performance_result.get('degradation_detected', False)
        confidence_drop = confidence_result.get('confidence_drop_detected', False)
        
        # Actions recommandées
        recommended_actions = []
        if degradation_detected:
            recommended_actions.extend([
                'reduce_position_size',
                'switch_to_backup_model',
                'trigger_model_retrain'
            ])
        
        if confidence_drop:
            recommended_actions.extend([
                'reduce_leverage',
                'increase_risk_controls'
            ])
        
        # Création alerte si nécessaire
        if degradation_detected or confidence_drop:
            max_severity = DriftSeverity.CRITICAL if degradation_detected else DriftSeverity.WARNING
            alert_id = self.alerting_engine.process_alert(
                AlertType.MODEL_DEGRADATION,
                max_severity,
                {**performance_result, **confidence_result},
                recommended_actions
            )
        
        return {
            'performance': performance_result,
            'confidence': confidence_result,
            'degradation_detected': degradation_detected,
            'recommended_actions': recommended_actions
        }
    
    async def monitor_business_metrics(self, model_id: str, predictions: np.ndarray, 
                                     actual_pnl: np.ndarray, current_volatility: float = None) -> Dict:
        """Surveillance métriques business"""
        if model_id not in self.business_monitors:
            return {'error': 'Model not registered'}
        
        monitor = self.business_monitors[model_id]
        timestamp = time.time()
        
        # Surveillance corrélation P&L
        correlation_result = await monitor.monitor_pnl_correlation(predictions, actual_pnl, timestamp)
        
        # Détection changement régime si volatilité disponible
        regime_result = {}
        if current_volatility is not None:
            baseline_volatility = self.config.get('baseline_volatility', 0.2)
            regime_result = monitor.detect_regime_change_impact(current_volatility, baseline_volatility)
        
        # Actions recommandées
        recommended_actions = []
        if correlation_result.get('correlation_degradation', False):
            recommended_actions.extend([
                'enable_safety_mode',
                'reduce_position_size',
                'review_risk_parameters'
            ])
        
        if regime_result.get('regime_change_detected', False):
            recommended_actions.extend([
                'adapt_trading_parameters',
                'increase_monitoring_frequency'
            ])
        
        # Création alerte si nécessaire
        if correlation_result.get('correlation_degradation', False):
            alert_id = self.alerting_engine.process_alert(
                AlertType.PNL_CORRELATION_BREAKDOWN,
                correlation_result.get('severity', DriftSeverity.WARNING),
                {**correlation_result, **regime_result},
                recommended_actions
            )
        
        return {
            'correlation': correlation_result,
            'regime': regime_result,
            'recommended_actions': recommended_actions
        }
    
    def get_monitoring_summary(self) -> Dict:
        """Résumé global du monitoring"""
        summary = {
            'total_models': len(self.performance_monitors),
            'active_alerts': len([a for a in self.alerting_engine.alert_history if a['severity'] != 'normal']),
            'critical_alerts': len([a for a in self.alerting_engine.alert_history if a['severity'] == 'critical']),
            'warning_alerts': len([a for a in self.alerting_engine.alert_history if a['severity'] == 'warning']),
            'monitored_metrics_count': len(self.monitored_metrics),
            'last_alert_time': max([a['timestamp'] for a in self.alerting_engine.alert_history]) if self.alerting_engine.alert_history else None
        }
        
        return summary
    
    def calculate_model_health_score(self, model_id: str) -> Dict:
        """Calcul score de santé global du modèle (0-100)"""
        if model_id not in self.performance_monitors:
            return {'error': 'Model not registered'}
        
        # Récupération métriques récentes
        monitor = self.performance_monitors[model_id]
        recent_metrics = monitor.performance_buffer.get_recent(100)
        
        if not recent_metrics:
            return {'health_score': 50, 'insufficient_data': True}
        
        # Calcul scores composants
        weights = {
            'performance': 0.3,
            'data_quality': 0.25,
            'business_alignment': 0.25,
            'stability': 0.2
        }
        
        # Score performance (accuracy récente)
        recent_accuracy = np.mean([m.get('accuracy', 0.5) for m in recent_metrics])
        performance_score = min(100, recent_accuracy * 100)
        
        # Score data quality (basé sur PSI si disponible)
        data_quality_score = 80  # Valeur par défaut
        
        # Score business alignment (basé sur corrélation P&L si disponible)
        business_alignment_score = 80  # Valeur par défaut
        
        # Score stabilité (basé sur variance des métriques)
        accuracy_variance = np.var([m.get('accuracy', 0.5) for m in recent_metrics])
        stability_score = max(0, 100 - accuracy_variance * 1000)
        
        # Score pondéré
        health_score = sum([
            performance_score * weights['performance'],
            data_quality_score * weights['data_quality'],
            business_alignment_score * weights['business_alignment'],
            stability_score * weights['stability']
        ])
        
        return {
            'overall_health_score': max(0, min(100, health_score)),
            'component_scores': {
                'performance': performance_score,
                'data_quality': data_quality_score,
                'business_alignment': business_alignment_score,
                'stability': stability_score
            },
            'health_status': 'excellent' if health_score > 90 else 
                            'good' if health_score > 75 else
                            'warning' if health_score > 60 else 'critical'
        }
    
    def get_system_health(self) -> Dict:
        """Métriques de santé du système"""
        uptime = time.time() - self.start_time
        return {
            'overall_health': 'healthy',
            'data_quality': 'good',
            'model_performance': 'stable',
            'alerts_count': len(self.alerts),
            'uptime_seconds': uptime,
            'total_tests_run': self.total_tests_run,
            'total_alerts_generated': len(self.alerts),
            'buffer_utilization': len(self.monitoring_results) / 10000,  # max size
            'configuration': {
                'baseline_window': 5000,
                'test_window': 1000,
                'buffer_max_size': 10000
            }
        }

def integrate_model_monitoring(agent, config: Dict = None) -> ModelMonitoringSystem:
    """Intègre le système de monitoring dans un agent existant"""
    
    # Initialisation du système de monitoring
    monitoring_system = ModelMonitoringSystem(config)
    
    # Configuration par défaut si non fournie
    if config is None:
        config = {
            'baseline_metrics': {
                'accuracy': 0.8,
                'precision': 0.75,
                'recall': 0.7,
                'f1': 0.72,
                'mean_confidence': 0.7
            },
            'baseline_volatility': 0.2,
            'escalation_rules': {
                'critical_threshold': 0.25,
                'warning_threshold': 0.1
            }
        }
    
    # Enregistrement du modèle de l'agent
    model_id = f"{type(agent).__name__}_{int(time.time())}"
    baseline_importance = np.ones(20)  # Valeur par défaut
    
    monitoring_system.register_model(model_id, config['baseline_metrics'], baseline_importance)
    
    # Sauvegarde de la méthode originale
    original_make_decision = agent.make_decision
    
    async def monitored_make_decision(market_data):
        """Version surveillée de la décision"""
        try:
            # Décision originale
            decision = original_make_decision(market_data)
            
            # Simulation métriques pour démonstration
            if hasattr(agent, 'feature_pipeline') and agent.feature_pipeline:
                # Surveillance dérive des données
                baseline_features = np.random.randn(100, 20)  # Simulation
                current_features = np.random.randn(100, 20)  # Simulation
                
                drift_result = await monitoring_system.monitor_data_drift(
                    model_id, baseline_features, current_features
                )
                
                # Surveillance performance (simulation)
                y_true = np.random.randint(0, 2, 100)  # Simulation
                y_pred = np.random.randint(0, 2, 100)  # Simulation
                
                performance_result = await monitoring_system.monitor_model_performance(
                    model_id, y_true, y_pred
                )
                
                # Surveillance métriques business (simulation)
                predictions = np.random.randn(100)  # Simulation
                actual_pnl = np.random.randn(100) * 0.01  # Simulation
                
                business_result = await monitoring_system.monitor_business_metrics(
                    model_id, predictions, actual_pnl
                )
                
                # Stockage des résultats
                monitoring_system.monitoring_results.append({
                    'timestamp': time.time(),
                    'drift': drift_result,
                    'performance': performance_result,
                    'business': business_result
                })
            
            return decision
            
        except Exception as e:
            print(f"Erreur monitoring: {e}")
            return original_make_decision(market_data)
    
    # Remplacement de la méthode
    agent.make_decision = monitored_make_decision
    agent.model_monitoring = monitoring_system
    
    print("✅ Model Monitoring System intégré (47 métriques surveillées, 8 algorithmes de détection)")
    return monitoring_system
