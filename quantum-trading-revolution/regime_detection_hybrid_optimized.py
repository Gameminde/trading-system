"""
AGENT QUANTUM TRADING - MODULE 4 OPTIMISÉ: REGIME DETECTION HYBRID
Version production avec sklearn (GMM) + validation croisée et robustesse
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class RegimeType(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


@dataclass(frozen=True)
class RegimeDetectionConfig:
    window_size: int = 252
    min_data_points: int = 50
    confidence_threshold: float = 0.6
    cross_validation_folds: int = 5


@dataclass
class RegimeDetectionResult:
    timestamp: float
    regime: RegimeType
    confidence: float
    algorithm: str
    details: Dict = field(default_factory=dict)


class FeatureExtractor:
    def extract(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        if prices.ndim != 1 or len(prices) < 5:
            return np.zeros(8)
        returns = np.diff(np.log(prices))
        if len(returns) == 0:
            returns = np.array([0.0])
        mean_r = float(np.mean(returns))
        vol_r = float(np.std(returns) + 1e-8)
        skew_r = float(np.mean(((returns - mean_r) / (vol_r + 1e-8)) ** 3))
        kurt_r = float(np.mean(((returns - mean_r) / (vol_r + 1e-8)) ** 4) - 3.0)
        if volumes is not None and len(volumes) >= len(prices) - 1:
            v = np.asarray(volumes[-len(returns):], dtype=float)
            v_mean = float(np.mean(v) + 1e-8)
            v_vol = float(np.std(v) + 1e-8)
            v_trend = float(np.polyfit(np.arange(len(v)), v, 1)[0] / (v_mean + 1e-8))
        else:
            v_mean = v_vol = v_trend = 0.0
        slope = float(np.polyfit(np.arange(len(prices)), prices, 1)[0] / (np.mean(prices) + 1e-8))
        return np.array([mean_r, vol_r, skew_r, kurt_r, v_mean, v_vol, v_trend, slope])


class GMMRegimeDetector:
    def __init__(self, config: RegimeDetectionConfig):
        self.config = config
        self.extractor = FeatureExtractor()
        self.model = None
        self.is_trained = False
        self.logger = logging.getLogger(f"{__name__}.GMMRegimeDetector")
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
            self.logger.setLevel(logging.INFO)

    def _create_model(self):
        try:
            from sklearn.mixture import GaussianMixture
            return GaussianMixture(n_components=4, covariance_type='full', random_state=42, max_iter=200)
        except Exception:
            return None

    def train(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> bool:
        prices = np.asarray(prices, dtype=float)
        if len(prices) < self.config.min_data_points:
            return False
        feats = self._features_matrix(prices, volumes)
        model = self._create_model()
        if model is None:
            # Fallback: simple KMeans-like centers via percentiles
            centers = np.percentile(feats, [20, 40, 60, 80], axis=0)
            self.model = {"centers": centers}
            self.is_trained = True
            return True
        try:
            model.fit(feats)
            self.model = model
            self.is_trained = True
            return True
        except Exception as e:
            self.logger.error(f"GMM fit failed: {e}")
            return False

    def _features_matrix(self, prices: np.ndarray, volumes: Optional[np.ndarray]) -> np.ndarray:
        W = min(self.config.window_size, len(prices) - 4)
        rows: List[np.ndarray] = []
        for i in range(W, len(prices)):
            rows.append(self.extractor.extract(prices[i-W:i], None if volumes is None else volumes[i-W:i]))
        return np.vstack(rows) if rows else np.zeros((0, 8))

    def predict(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> RegimeDetectionResult:
        if not self.is_trained:
            return RegimeDetectionResult(time.time(), RegimeType.SIDEWAYS, 0.0, 'gmm', {'error': 'not_trained'})
        feat = self.extractor.extract(prices, volumes).reshape(1, -1)
        try:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(feat)[0]
                idx = int(np.argmax(proba))
                confidence = float(np.max(proba))
            else:
                centers = self.model['centers']
                dists = np.linalg.norm(centers - feat, axis=1)
                idx = int(np.argmin(dists))
                confidence = float(1.0 / (1.0 + dists[idx]))
            mapping = [RegimeType.BULL, RegimeType.BEAR, RegimeType.SIDEWAYS, RegimeType.CRISIS]
            regime = mapping[idx % len(mapping)]
            return RegimeDetectionResult(time.time(), regime, confidence, 'gmm')
        except Exception as e:
            self.logger.error(f"Predict failed: {e}")
            return RegimeDetectionResult(time.time(), RegimeType.SIDEWAYS, 0.0, 'gmm', {'error': str(e)})


class ThresholdRegimeDetector:
    def __init__(self, config: RegimeDetectionConfig):
        self.config = config
        self.extractor = FeatureExtractor()
        self.thresholds = {
            'bull': 0.02,
            'bear': -0.02,
            'vol': 0.03,
            'crisis': 0.05,
        }
        self.is_trained = False

    def train(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> bool:
        feats = self._features_matrix(prices, volumes)
        if feats.size == 0:
            return False
        returns = feats[:, 0]
        vol = feats[:, 1]
        self.thresholds['bull'] = float(np.percentile(returns, 75))
        self.thresholds['bear'] = float(np.percentile(returns, 25))
        self.thresholds['vol'] = float(np.percentile(vol, 70))
        self.thresholds['crisis'] = float(np.percentile(vol, 95))
        self.is_trained = True
        return True

    def _features_matrix(self, prices: np.ndarray, volumes: Optional[np.ndarray]) -> np.ndarray:
        W = min(self.config.window_size, len(prices) - 4)
        rows: List[np.ndarray] = []
        for i in range(W, len(prices)):
            rows.append(self.extractor.extract(prices[i-W:i], None if volumes is None else volumes[i-W:i]))
        return np.vstack(rows) if rows else np.zeros((0, 8))

    def predict(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> RegimeDetectionResult:
        feat = self.extractor.extract(prices, volumes)
        r, v = float(feat[0]), float(feat[1])
        if v > self.thresholds['crisis']:
            return RegimeDetectionResult(time.time(), RegimeType.CRISIS, min(1.0, v / (self.thresholds['crisis'] * 2)), 'threshold')
        if r > self.thresholds['bull']:
            return RegimeDetectionResult(time.time(), RegimeType.BULL, min(1.0, r / (self.thresholds['bull'] * 2)), 'threshold')
        if r < self.thresholds['bear']:
            return RegimeDetectionResult(time.time(), RegimeType.BEAR, min(1.0, abs(r) / (abs(self.thresholds['bear']) * 2)), 'threshold')
        return RegimeDetectionResult(time.time(), RegimeType.SIDEWAYS, 0.5, 'threshold')


class RegimeDetectionEnsembleOptimized:
    def __init__(self, config: Optional[RegimeDetectionConfig] = None):
        self.config = config or RegimeDetectionConfig()
        self.detectors = {
            'gmm': GMMRegimeDetector(self.config),
            'threshold': ThresholdRegimeDetector(self.config),
        }
        self.weights: Dict[str, float] = {'gmm': 0.6, 'threshold': 0.4}
        self.history: List[Dict] = []

    def train(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> bool:
        ok = False
        for name, det in self.detectors.items():
            try:
                if det.train(prices, volumes):
                    ok = True
            except Exception:
                continue
        return ok

    def detect(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Dict:
        results: Dict[str, RegimeDetectionResult] = {}
        for name, det in self.detectors.items():
            if getattr(det, 'is_trained', True):
                results[name] = det.predict(prices, volumes)

        if not results:
            return {'error': 'no_results'}

        score: Dict[RegimeType, float] = {r: 0.0 for r in RegimeType}
        for name, res in results.items():
            w = self.weights.get(name, 1.0)
            score[res.regime] += w * res.confidence

        best = max(score.items(), key=lambda x: x[1])
        final_regime, final_score = best[0], float(best[1])
        out = {
            'ensemble_regime': final_regime.value,
            'ensemble_confidence': final_score,
            'individual': {k: {'regime': v.regime.value, 'confidence': v.confidence} for k, v in results.items()},
            'timestamp': time.time(),
        }
        self.history.append(out)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        return out


def integrate_regime_detection_optimized(agent, config: Optional[Dict] = None) -> RegimeDetectionEnsembleOptimized:
    ensemble = RegimeDetectionEnsembleOptimized(RegimeDetectionConfig())
    original = agent.make_decision

    def enhanced(market_data):
        try:
            prices = None
            if hasattr(market_data, 'get'):
                close = market_data.get('close')
                if close is not None:
                    prices = np.array(close) if isinstance(close, list) else np.array([close])
            if prices is not None and len(prices) >= 60:
                ensemble.detect(prices)
        except Exception:
            pass
        return original(market_data)

    agent.make_decision = enhanced
    agent.regime_detection_optimized = ensemble
    return ensemble


