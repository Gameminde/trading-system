# FILE: online_learning_framework.py
"""
AGENT QUANTUM TRADING - MODULE 2: ONLINE LEARNING FRAMEWORK
Apprentissage incrémental et adaptation continue pour trading algorithmique
Impact: Streaming + modèles incrémentaux (River, VW), détection concept drift (ADWIN, DDM, EDDM), ensemble dynamique + A/B testing live
Basé sur: OnlineLearningFramework – Apprentissage incrémenta.md
"""

import numpy as np
import pandas as pd
import time
import asyncio
import warnings
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

class ModelType(Enum):
    ONLINE_GD = "online_gradient_descent"
    INCREMENTAL_SVM = "incremental_svm"
    STREAM_RF = "stream_random_forest"
    ADAPTIVE_BAGGING = "adaptive_bagging"

class DriftDetectorType(Enum):
    ADWIN = "adwin"
    DDM = "ddm"
    EDDM = "eddm"

@dataclass
class ModelPerformance:
    model_id: str
    timestamp: float
    loss: float
    accuracy: float
    predictions: np.ndarray
    confidence: float
    drift_detected: bool
    drift_confidence: float

class ConceptDriftDetector:
    """Détecteur de concept drift pour apprentissage en ligne"""
    
    def __init__(self, detector_type: DriftDetectorType, window_size: int = 30):
        self.detector_type = detector_type
        self.window_size = window_size
        self.error_history = deque(maxlen=window_size * 2)
        self.drift_history = []
        
    def detect_drift(self, current_error: float, current_confidence: float = None) -> Dict:
        """Détection de concept drift"""
        self.error_history.append(current_error)
        
        if len(self.error_history) < self.window_size:
            return {'drift_detected': False, 'confidence': 0.0}
        
        if self.detector_type == DriftDetectorType.ADWIN:
            return self._adwin_detection()
        elif self.detector_type == DriftDetectorType.DDM:
            return self._ddm_detection()
        elif self.detector_type == DriftDetectorType.EDDM:
            return self._eddm_detection()
        else:
            return self._simple_detection()
    
    def _adwin_detection(self) -> Dict:
        """Détection ADWIN (ADaptive WINdowing)"""
        if len(self.error_history) < self.window_size * 2:
            return {'drift_detected': False, 'confidence': 0.0}
        
        # Division en deux fenêtres
        w0 = list(self.error_history)[:self.window_size]
        w1 = list(self.error_history)[self.window_size:]
        
        # Calcul statistiques
        mean_w0 = np.mean(w0)
        mean_w1 = np.mean(w1)
        var_w0 = np.var(w0)
        var_w1 = np.var(w1)
        
        # Test de différence significative
        n0, n1 = len(w0), len(w1)
        pooled_var = ((n0 - 1) * var_w0 + (n1 - 1) * var_w1) / (n0 + n1 - 2)
        
        if pooled_var == 0:
            drift_detected = abs(mean_w1 - mean_w0) > 0.1
        else:
            t_stat = abs(mean_w1 - mean_w0) / np.sqrt(pooled_var * (1/n0 + 1/n1))
            drift_detected = t_stat > 2.0  # Seuil t-test
        
        confidence = min(1.0, abs(mean_w1 - mean_w0) / (np.std(self.error_history) + 1e-8))
        
        return {
            'drift_detected': drift_detected,
            'confidence': confidence,
            'mean_difference': mean_w1 - mean_w0,
            't_statistic': t_stat if 't_stat' in locals() else 0.0
        }
    
    def _ddm_detection(self) -> Dict:
        """Détection DDM (Drift Detection Method)"""
        if len(self.error_history) < self.window_size:
            return {'drift_detected': False, 'confidence': 0.0}
        
        recent_errors = list(self.error_history)[-self.window_size:]
        mean_error = np.mean(recent_errors)
        std_error = np.std(recent_errors)
        
        # Détection si erreur dépasse 2σ
        drift_detected = mean_error > 2 * std_error
        
        confidence = min(1.0, mean_error / (std_error + 1e-8))
        
        return {
            'drift_detected': drift_detected,
            'confidence': confidence,
            'mean_error': mean_error,
            'std_error': std_error
        }
    
    def _eddm_detection(self) -> Dict:
        """Détection EDDM (Early Drift Detection Method)"""
        if len(self.error_history) < self.window_size:
            return {'drift_detected': False, 'confidence': 0.0}
        
        recent_errors = list(self.error_history)[-self.window_size:]
        
        # Calcul distance moyenne entre erreurs consécutives
        distances = [abs(recent_errors[i] - recent_errors[i-1]) for i in range(1, len(recent_errors))]
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Détection si distance dépasse 3σ
        drift_detected = mean_distance > 3 * std_distance
        
        confidence = min(1.0, mean_distance / (std_distance + 1e-8))
        
        return {
            'drift_detected': drift_detected,
            'confidence': confidence,
            'mean_distance': mean_distance,
            'std_distance': std_distance
        }
    
    def _simple_detection(self) -> Dict:
        """Détection simple basée sur tendance des erreurs"""
        if len(self.error_history) < self.window_size:
            return {'drift_detected': False, 'confidence': 0.0}
        
        recent_errors = list(self.error_history)[-self.window_size:]
        
        # Régression linéaire pour détecter tendance
        x = np.arange(len(recent_errors))
        slope = np.polyfit(x, recent_errors, 1)[0]
        
        # Détection si pente positive significative
        drift_detected = slope > 0.01
        
        confidence = min(1.0, abs(slope) * 100)
        
        return {
            'drift_detected': drift_detected,
            'confidence': confidence,
            'slope': slope
        }

class IncrementalModel:
    """Modèle d'apprentissage incrémental de base"""
    
    def __init__(self, model_type: ModelType, model_id: str):
        self.model_type = model_type
        self.model_id = model_id
        self.is_trained = False
        self.training_samples = 0
        self.last_update = time.time()
        self.performance_history = []
        
        # Paramètres du modèle
        self.learning_rate = 0.01
        self.weights = None
        self.bias = 0.0
        
        # Initialisation selon le type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialisation du modèle selon le type"""
        if self.model_type == ModelType.ONLINE_GD:
            self.weights = np.zeros(20)  # 20 features par défaut
            self.bias = 0.0
        elif self.model_type == ModelType.INCREMENTAL_SVM:
            self.weights = np.zeros(20)
            self.bias = 0.0
            self.support_vectors = []
        elif self.model_type == ModelType.STREAM_RF:
            self.trees = []
            self.max_trees = 10
        elif self.model_type == ModelType.ADAPTIVE_BAGGING:
            self.base_models = []
            self.weights = []
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Mise à jour incrémentale du modèle"""
        try:
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            if self.model_type == ModelType.ONLINE_GD:
                return self._online_gd_update(X, y)
            elif self.model_type == ModelType.INCREMENTAL_SVM:
                return self._incremental_svm_update(X, y)
            elif self.model_type == ModelType.STREAM_RF:
                return self._stream_rf_update(X, y)
            elif self.model_type == ModelType.ADAPTIVE_BAGGING:
                return self._adaptive_bagging_update(X, y)
            
            return False
            
        except Exception as e:
            print(f"Erreur partial_fit pour {self.model_id}: {e}")
            return False
    
    def _online_gd_update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Mise à jour Online Gradient Descent"""
        for i in range(len(X)):
            # Prédiction
            prediction = np.dot(X[i], self.weights) + self.bias
            
            # Calcul gradient
            error = y[i] - prediction
            grad_weights = -2 * error * X[i]
            grad_bias = -2 * error
            
            # Mise à jour
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias
        
        self.is_trained = True
        self.training_samples += len(X)
        self.last_update = time.time()
        return True
    
    def _incremental_svm_update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Mise à jour Incremental SVM (simplifié)"""
        for i in range(len(X)):
            # Prédiction
            prediction = np.dot(X[i], self.weights) + self.bias
            
            # Mise à jour seulement si marge < 1
            margin = y[i] * prediction
            if margin < 1:
                # Mise à jour des poids
                self.weights += self.learning_rate * y[i] * X[i]
                self.bias += self.learning_rate * y[i]
        
        self.is_trained = True
        self.training_samples += len(X)
        self.last_update = time.time()
        return True
    
    def _stream_rf_update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Mise à jour Stream Random Forest (simplifié)"""
        # Ajout d'un nouvel arbre si nécessaire
        if len(self.trees) < self.max_trees:
            # Création d'un arbre simple
            tree = self._create_simple_tree(X, y)
            self.trees.append(tree)
        
        # Mise à jour des arbres existants
        for tree in self.trees:
            self._update_tree(tree, X, y)
        
        self.is_trained = True
        self.training_samples += len(X)
        self.last_update = time.time()
        return True
    
    def _adaptive_bagging_update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Mise à jour Adaptive Bagging (simplifié)"""
        # Ajout d'un nouveau modèle de base si nécessaire
        if len(self.base_models) < 5:
            base_model = IncrementalModel(ModelType.ONLINE_GD, f"{self.model_id}_base_{len(self.base_models)}")
            base_model.partial_fit(X, y)
            self.base_models.append(base_model)
            self.weights.append(1.0)
        
        # Mise à jour des modèles existants
        for i, model in enumerate(self.base_models):
            model.partial_fit(X, y)
        
        # Mise à jour des poids
        self._update_ensemble_weights(X, y)
        
        self.is_trained = True
        self.training_samples += len(X)
        self.last_update = time.time()
        return True
    
    def _create_simple_tree(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Création d'un arbre simple pour Stream RF"""
        # Arbre simplifié avec une seule division
        if len(X) == 0:
            return {'leaf': True, 'prediction': 0.0}
        
        # Division basée sur la première feature
        feature_idx = 0
        threshold = np.median(X[:, feature_idx])
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold
        
        left_pred = np.mean(y[left_mask]) if np.any(left_mask) else 0.0
        right_pred = np.mean(y[right_mask]) if np.any(right_mask) else 0.0
        
        return {
            'leaf': False,
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': {'leaf': True, 'prediction': left_pred},
            'right': {'leaf': True, 'prediction': right_pred}
        }
    
    def _update_tree(self, tree: Dict, X: np.ndarray, y: np.ndarray):
        """Mise à jour d'un arbre (simplifié)"""
        # Mise à jour simple des prédictions des feuilles
        if tree['leaf']:
            return
        
        left_mask = X[:, tree['feature_idx']] <= tree['threshold']
        right_mask = X[:, tree['feature_idx']] > tree['threshold']
        
        if np.any(left_mask):
            tree['left']['prediction'] = np.mean(y[left_mask])
        if np.any(right_mask):
            tree['right']['prediction'] = np.mean(y[right_mask])
    
    def _update_ensemble_weights(self, X: np.ndarray, y: np.ndarray):
        """Mise à jour des poids de l'ensemble"""
        if len(self.base_models) == 0:
            return
        
        # Calcul des erreurs pour chaque modèle
        errors = []
        for model in self.base_models:
            predictions = model.predict(X)
            error = np.mean((y - predictions) ** 2)
            errors.append(error)
        
        # Mise à jour des poids (inversement proportionnel à l'erreur)
        total_error = sum(errors)
        if total_error > 0:
            self.weights = [1.0 / (e + 1e-8) for e in errors]
            # Normalisation
            weight_sum = sum(self.weights)
            self.weights = [w / weight_sum for w in self.weights]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédiction avec le modèle"""
        if not self.is_trained:
            return np.zeros(len(X))
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if self.model_type == ModelType.ONLINE_GD:
            return np.dot(X, self.weights) + self.bias
        elif self.model_type == ModelType.INCREMENTAL_SVM:
            return np.dot(X, self.weights) + self.bias
        elif self.model_type == ModelType.STREAM_RF:
            return self._predict_tree(X)
        elif self.model_type == ModelType.ADAPTIVE_BAGGING:
            return self._predict_ensemble(X)
        
        return np.zeros(len(X))
    
    def _predict_tree(self, X: np.ndarray) -> np.ndarray:
        """Prédiction avec Stream RF"""
        predictions = np.zeros(len(X))
        
        for i, x in enumerate(X):
            tree_preds = []
            for tree in self.trees:
                tree_preds.append(self._predict_single_tree(tree, x))
            predictions[i] = np.mean(tree_preds)
        
        return predictions
    
    def _predict_single_tree(self, tree: Dict, x: np.ndarray) -> float:
        """Prédiction avec un seul arbre"""
        if tree['leaf']:
            return tree['prediction']
        
        if x[tree['feature_idx']] <= tree['threshold']:
            return self._predict_single_tree(tree['left'], x)
        else:
            return self._predict_single_tree(tree['right'], x)
    
    def _predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Prédiction avec l'ensemble"""
        if len(self.base_models) == 0:
            return np.zeros(len(X))
        
        predictions = np.zeros(len(X))
        
        for i, x in enumerate(X):
            weighted_sum = 0.0
            for j, model in enumerate(self.base_models):
                pred = model.predict(x.reshape(1, -1))[0]
                weighted_sum += self.weights[j] * pred
            predictions[i] = weighted_sum
        
        return predictions
    
    def get_performance_metrics(self) -> Dict:
        """Récupération des métriques de performance"""
        return {
            'model_id': self.model_id,
            'model_type': self.model_type.value,
            'is_trained': self.is_trained,
            'training_samples': self.training_samples,
            'last_update': self.last_update,
            'performance_history': self.performance_history
        }

class OnlineLearningFramework:
    """Framework d'apprentissage en ligne complet"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.models = {}
        self.drift_detectors = {}
        self.ensemble_weights = {}
        self.performance_history = []
        self.ab_testing_config = {}
        
        # Configuration par défaut
        self.default_config = {
            'model_types': [ModelType.ONLINE_GD, ModelType.INCREMENTAL_SVM, ModelType.STREAM_RF],
            'drift_detectors': [DriftDetectorType.ADWIN, DriftDetectorType.DDM],
            'ensemble_method': 'weighted_average',
            'ab_testing_enabled': True,
            'model_retrain_threshold': 0.15,
            'max_models': 5
        }
        
        self.config = {**self.default_config, **self.config}
        
        # Initialisation des modèles
        self._initialize_models()
        
        # Thread pool pour traitement asynchrone
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _initialize_models(self):
        """Initialisation des modèles d'apprentissage"""
        for i, model_type in enumerate(self.config['model_types']):
            model_id = f"{model_type.value}_{i}"
            model = IncrementalModel(model_type, model_id)
            self.models[model_id] = model
            
            # Initialisation des détecteurs de drift
            for detector_type in self.config['drift_detectors']:
                detector_id = f"{model_id}_{detector_type.value}"
                detector = ConceptDriftDetector(detector_type)
                self.drift_detectors[detector_id] = detector
            
            # Poids initiaux de l'ensemble
            self.ensemble_weights[model_id] = 1.0 / len(self.config['model_types'])
    
    async def process_streaming_data(self, X: np.ndarray, y: np.ndarray, 
                                   metadata: Dict = None) -> Dict:
        """Traitement des données streaming"""
        start_time = time.time()
        
        # Mise à jour de tous les modèles
        update_results = {}
        for model_id, model in self.models.items():
            try:
                success = model.partial_fit(X, y)
                update_results[model_id] = {
                    'success': success,
                    'training_samples': model.training_samples
                }
            except Exception as e:
                update_results[model_id] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Détection de concept drift
        drift_results = await self._detect_concept_drift(X, y)
        
        # Mise à jour des poids de l'ensemble
        ensemble_update = self._update_ensemble_weights(X, y)
        
        # A/B testing si activé
        ab_results = {}
        if self.config['ab_testing_enabled']:
            ab_results = await self._run_ab_testing(X, y, metadata)
        
        # Calcul des performances
        performance_metrics = self._calculate_performance_metrics(X, y)
        
        # Stockage de l'historique
        self.performance_history.append({
            'timestamp': time.time(),
            'update_results': update_results,
            'drift_results': drift_results,
            'ensemble_update': ensemble_update,
            'ab_results': ab_results,
            'performance_metrics': performance_metrics
        })
        
        processing_time = time.time() - start_time
        
        return {
            'processing_time': processing_time,
            'update_results': update_results,
            'drift_results': drift_results,
            'ensemble_update': ensemble_update,
            'ab_results': ab_results,
            'performance_metrics': performance_metrics
        }
    
    async def _detect_concept_drift(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Détection de concept drift pour tous les modèles"""
        drift_results = {}
        
        for model_id, model in self.models.items():
            if not model.is_trained:
                continue
            
            # Prédictions et erreurs
            predictions = model.predict(X)
            errors = (y - predictions) ** 2
            
            model_drift_results = {}
            
            # Test avec tous les détecteurs
            for detector_id, detector in self.drift_detectors.items():
                if detector_id.startswith(model_id):
                    drift_result = detector.detect_drift(np.mean(errors))
                    model_drift_results[detector_id] = drift_result
                    
                    # Si drift détecté, déclencher actions
                    if drift_result['drift_detected']:
                        await self._handle_concept_drift(model_id, detector_id, drift_result)
            
            drift_results[model_id] = model_drift_results
        
        return drift_results
    
    async def _handle_concept_drift(self, model_id: str, detector_id: str, drift_result: Dict):
        """Gestion du concept drift détecté"""
        print(f"🚨 Concept drift détecté pour {model_id} par {detector_id}")
        print(f"   Confiance: {drift_result.get('confidence', 0.0):.3f}")
        
        # Actions automatiques
        if drift_result.get('confidence', 0.0) > 0.7:
            # Drift critique - retrain complet
            print(f"   🔄 Retrain complet du modèle {model_id}")
            await self._retrain_model(model_id)
        else:
            # Drift modéré - ajustement des paramètres
            print(f"   ⚙️  Ajustement des paramètres pour {model_id}")
            self._adjust_model_parameters(model_id, drift_result)
    
    async def _retrain_model(self, model_id: str):
        """Retrain complet d'un modèle"""
        if model_id not in self.models:
            return
        
        model = self.models[model_id]
        
        # Sauvegarde des données récentes pour retrain
        recent_data = self._get_recent_training_data()
        
        if recent_data and len(recent_data['X']) > 100:
            # Retrain avec données récentes
            model.partial_fit(recent_data['X'], recent_data['y'])
            print(f"   ✅ Retrain terminé pour {model_id}")
        else:
            print(f"   ⚠️  Données insuffisantes pour retrain de {model_id}")
    
    def _adjust_model_parameters(self, model_id: str, drift_result: Dict):
        """Ajustement des paramètres du modèle"""
        if model_id not in self.models:
            return
        
        model = self.models[model_id]
        
        # Ajustement du learning rate selon la sévérité du drift
        confidence = drift_result.get('confidence', 0.0)
        if confidence > 0.5:
            model.learning_rate *= 1.5  # Augmentation pour adaptation rapide
        else:
            model.learning_rate *= 0.8  # Réduction pour stabilité
    
    def _get_recent_training_data(self) -> Optional[Dict]:
        """Récupération des données récentes pour retrain"""
        if len(self.performance_history) < 10:
            return None
        
        # Agrégation des données récentes
        recent_X = []
        recent_y = []
        
        for entry in self.performance_history[-10:]:
            if 'metadata' in entry and 'X' in entry['metadata']:
                recent_X.extend(entry['metadata']['X'])
                recent_y.extend(entry['metadata']['y'])
        
        if recent_X and recent_y:
            return {
                'X': np.array(recent_X),
                'y': np.array(recent_y)
            }
        
        return None
    
    def _update_ensemble_weights(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Mise à jour des poids de l'ensemble"""
        if len(self.models) == 0:
            return {}
        
        # Calcul des erreurs pour chaque modèle
        errors = {}
        for model_id, model in self.models.items():
            if not model.is_trained:
                errors[model_id] = float('inf')
                continue
            
            try:
                predictions = model.predict(X)
                error = np.mean((y - predictions) ** 2)
                errors[model_id] = error
            except:
                errors[model_id] = float('inf')
        
        # Mise à jour des poids (inversement proportionnel à l'erreur)
        total_error = sum(errors.values())
        if total_error > 0:
            for model_id in self.models:
                if errors[model_id] < float('inf'):
                    self.ensemble_weights[model_id] = 1.0 / (errors[model_id] + 1e-8)
                else:
                    self.ensemble_weights[model_id] = 0.0
            
            # Normalisation
            weight_sum = sum(self.ensemble_weights.values())
            if weight_sum > 0:
                for model_id in self.models:
                    self.ensemble_weights[model_id] /= weight_sum
        
        return {
            'weights': self.ensemble_weights.copy(),
            'errors': errors
        }
    
    async def _run_ab_testing(self, X: np.ndarray, y: np.ndarray, metadata: Dict = None) -> Dict:
        """Exécution des tests A/B"""
        if not self.config['ab_testing_enabled']:
            return {}
        
        # Configuration A/B testing
        ab_config = self.config.get('ab_testing_config', {
            'traffic_split': {'model_v1': 0.6, 'model_v2': 0.2, 'shadow_next': 0.2},
            'switch_criteria': {'sharpe_30d': 0.05, 'drawdown': -0.10}
        })
        
        # Simulation de tests A/B
        ab_results = {
            'traffic_distribution': ab_config['traffic_split'],
            'performance_comparison': {},
            'switch_recommendation': False
        }
        
        # Comparaison des performances
        for model_id, model in self.models.items():
            if not model.is_trained:
                continue
            
            try:
                predictions = model.predict(X)
                performance = self._calculate_single_model_performance(y, predictions)
                ab_results['performance_comparison'][model_id] = performance
            except:
                continue
        
        # Recommandation de switch
        ab_results['switch_recommendation'] = self._evaluate_switch_criteria(ab_results['performance_comparison'])
        
        return ab_results
    
    def _calculate_single_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calcul des métriques de performance pour un modèle"""
        if len(y_true) == 0 or len(y_pred) == 0:
            return {}
        
        # Métriques de base
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Corrélation
        correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
        
        # Direction accuracy (pour trading)
        if len(y_true) > 1:
            direction_correct = np.sum(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
            direction_accuracy = direction_correct / (len(y_true) - 1)
        else:
            direction_accuracy = 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'direction_accuracy': direction_accuracy
        }
    
    def _evaluate_switch_criteria(self, performance_comparison: Dict) -> bool:
        """Évaluation des critères de switch"""
        if not performance_comparison:
            return False
        
        # Critères de switch (simplifiés)
        best_model = min(performance_comparison.keys(), 
                        key=lambda x: performance_comparison[x].get('mse', float('inf')))
        
        best_mse = performance_comparison[best_model].get('mse', float('inf'))
        
        # Switch si amélioration significative
        for model_id, perf in performance_comparison.items():
            if model_id != best_model:
                mse = perf.get('mse', float('inf'))
                if mse > best_mse * 1.2:  # 20% d'amélioration requise
                    return True
        
        return False
    
    def _calculate_performance_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Calcul des métriques de performance globales"""
        if len(self.models) == 0:
            return {}
        
        # Prédictions de l'ensemble
        ensemble_predictions = self.predict_ensemble(X)
        
        # Métriques globales
        global_metrics = self._calculate_single_model_performance(y, ensemble_predictions)
        
        # Métriques par modèle
        model_metrics = {}
        for model_id, model in self.models.items():
            if not model.is_trained:
                continue
            
            try:
                predictions = model.predict(X)
                model_metrics[model_id] = self._calculate_single_model_performance(y, predictions)
            except:
                continue
        
        return {
            'global': global_metrics,
            'models': model_metrics,
            'ensemble_weights': self.ensemble_weights.copy()
        }
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Prédiction avec l'ensemble de modèles"""
        if len(self.models) == 0:
            return np.zeros(len(X))
        
        # Prédictions individuelles
        model_predictions = {}
        for model_id, model in self.models.items():
            if not model.is_trained:
                continue
            
            try:
                predictions = model.predict(X)
                model_predictions[model_id] = predictions
            except:
                continue
        
        if not model_predictions:
            return np.zeros(len(X))
        
        # Combinaison pondérée
        ensemble_prediction = np.zeros(len(X))
        total_weight = 0.0
        
        for model_id, predictions in model_predictions.items():
            weight = self.ensemble_weights.get(model_id, 0.0)
            ensemble_prediction += weight * predictions
            total_weight += weight
        
        if total_weight > 0:
            ensemble_prediction /= total_weight
        
        return ensemble_prediction
    
    def get_active_models(self) -> List[str]:
        """Retourne la liste des modèles actifs (entraînés)"""
        return [model_id for model_id, model in self.models.items() if model.is_trained]
    
    def get_framework_summary(self) -> Dict:
        """Résumé complet du framework"""
        return {
            'total_models': len(self.models),
            'active_models': len([m for m in self.models.values() if m.is_trained]),
            'drift_detectors': len(self.drift_detectors),
            'ensemble_weights': self.ensemble_weights.copy(),
            'ab_testing_enabled': self.config['ab_testing_enabled'],
            'performance_history_length': len(self.performance_history),
            'last_update': max([m.last_update for m in self.models.values()]) if self.models else 0
        }

def integrate_online_learning(agent, config: Dict = None) -> OnlineLearningFramework:
    """Intègre le framework d'apprentissage en ligne dans un agent existant"""
    
    # Initialisation du framework
    framework = OnlineLearningFramework(config)
    
    # Sauvegarde de la méthode originale
    original_make_decision = agent.make_decision
    
    async def enhanced_make_decision(market_data):
        """Version améliorée avec apprentissage en ligne"""
        try:
            # Décision originale
            decision = original_make_decision(market_data)
            
            # Simulation données d'apprentissage pour démonstration
            if hasattr(agent, 'feature_pipeline') and agent.feature_pipeline:
                # Génération features
                features = agent.feature_pipeline.generate_core_features(market_data)
                
                if len(features) > 10:
                    # Données d'entraînement simulées
                    X = features.iloc[-10:].values
                    y = np.random.randn(10) * 0.01  # P&L simulé
                    
                    # Traitement streaming
                    streaming_result = await framework.process_streaming_data(X, y, {
                        'X': X,
                        'y': y,
                        'timestamp': time.time()
                    })
                    
                    # Stockage des résultats
                    agent.online_learning_results = streaming_result
            
            return decision
            
        except Exception as e:
            print(f"Erreur apprentissage en ligne: {e}")
            return original_make_decision(market_data)
    
    # Remplacement de la méthode
    agent.make_decision = enhanced_make_decision
    agent.online_learning = framework
    
    print("✅ Online Learning Framework intégré (streaming, modèles incrémentaux, détection drift)")
    return framework
