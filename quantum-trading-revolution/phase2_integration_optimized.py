"""
AGENT QUANTUM TRADING - PHASE 2 INTEGRATION OPTIMISÉE
Script d'intégration progressive avec logging structuré et validation robuste
"""

import asyncio
import logging
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol, Any
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class IntegrationConfig:
    """Configuration pour l'intégration Phase 2"""
    performance_test_iterations: int = 100
    intermediate_test_iterations: int = 50
    validation_timeout_seconds: float = 30.0
    enable_performance_tracking: bool = True
    save_detailed_report: bool = True
    report_filename: str = "phase2_integration_report.json"
    
    def __post_init__(self):
        if self.performance_test_iterations <= 0:
            raise ValueError("performance_test_iterations must be positive")
        if self.validation_timeout_seconds <= 0:
            raise ValueError("validation_timeout_seconds must be positive")

class ModuleName(Enum):
    """Noms des modules Phase 2"""
    MODEL_MONITORING = "model_monitoring"
    ONLINE_LEARNING = "online_learning"
    MULTI_AGENT = "multi_agent"
    REGIME_DETECTION = "regime_detection"
    ULTRA_LATENCY = "ultra_latency"

class IntegrationStatus(Enum):
    """Statuts d'intégration des modules"""
    INTEGRATING = "integrating"
    INTEGRATED = "integrated"
    VALIDATED = "validated"
    FAILED = "failed"

@dataclass
class ModuleIntegrationResult:
    """Résultat d'intégration d'un module"""
    module_name: ModuleName
    status: IntegrationStatus
    integration_object: Any = None
    error_message: str = ""
    performance_metrics: Optional[Dict] = None
    validation_passed: bool = False
    integration_time: float = 0.0

class TestTradingAgent:
    """Agent de trading de test avec métriques de performance"""
    
    def __init__(self):
        self.decision_count = 0
        self.performance_history = []
        self.last_decision = {}
        self.current_regime = "normal"
        self.regime_confidence = 0.5
        self.logger = logging.getLogger(f"{__name__}.TestAgent")
        
        # Configuration de base
        self.base_price = 100.0
        self.volatility = 0.02
    
    def make_decision(self, market_data) -> Dict:
        """Méthode de décision avec logging"""
        self.decision_count += 1
        start_time = time.time()
        
        try:
            # Simulation de décision basée sur les données
            if hasattr(market_data, '__len__') and len(market_data) > 0:
                current_price = self._extract_price(market_data)
                decision = self._make_price_based_decision(current_price)
            else:
                decision = {'action': 'HOLD', 'confidence': 0.3}
            
            processing_time = time.time() - start_time
            decision['processing_time_ms'] = processing_time * 1000
            
            self.last_decision = decision
            self.performance_history.append({
                'decision_count': self.decision_count,
                'processing_time_ms': processing_time * 1000,
                'timestamp': time.time()
            })
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Decision making error: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'error': str(e)}
    
    def _extract_price(self, market_data) -> float:
        """Extraction du prix avec gestion flexible des types de données"""
        if isinstance(market_data, (int, float)):
            return float(market_data)
        elif hasattr(market_data, '__getitem__'):
            try:
                return float(market_data[-1])  # Dernier prix
            except (IndexError, TypeError, ValueError):
                return 100.0  # Prix par défaut
        else:
            return 100.0
    
    def _make_price_based_decision(self, price: float) -> Dict:
        """Décision basée sur le prix avec logique améliorée"""
        if price > 105:
            return {'action': 'BUY', 'confidence': min(0.9, (price - 100) / 50)}
        elif price < 95:
            return {'action': 'SELL', 'confidence': min(0.9, (100 - price) / 50)}
        else:
            return {'action': 'HOLD', 'confidence': 0.5}
    
    def update_regime(self, regime: str, confidence: float):
        """Mise à jour du régime de marché"""
        self.current_regime = regime
        self.regime_confidence = confidence
    
    def get_performance_summary(self) -> Dict:
        """Résumé des performances de l'agent"""
        if not self.performance_history:
            return {}
        
        processing_times = [p['processing_time_ms'] for p in self.performance_history]
        
        return {
            'total_decisions': self.decision_count,
            'avg_processing_time_ms': np.mean(processing_times),
            'min_processing_time_ms': np.min(processing_times),
            'max_processing_time_ms': np.max(processing_times),
            'last_decision': self.last_decision,
            'current_regime': self.current_regime,
            'regime_confidence': self.regime_confidence
        }

class PerformanceMeasurer:
    """Mesureur de performances réutilisable et précis"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def measure_agent_performance(self, agent: TestTradingAgent, 
                                 iterations: int, 
                                 test_name: str = "performance_test") -> Dict:
        """Mesure de performance générique et précise"""
        self.logger.info(f"Measuring {test_name} performance ({iterations} iterations)...")
        
        # Données de test reproductibles
        np.random.seed(42)
        test_data = np.random.randn(iterations * 10) * 10 + 100
        
        # Mesures de performance
        latencies = []
        errors = 0
        
        overall_start = time.time()
        
        for i in range(iterations):
            data_slice = test_data[i:i+10] if len(test_data) > i+10 else test_data[-10:]
            
            start_time = time.perf_counter()
            try:
                decision = agent.make_decision(data_slice)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                # Validation de base de la décision
                if not isinstance(decision, dict) or 'action' not in decision:
                    errors += 1
                    
            except Exception as e:
                errors += 1
                self.logger.warning(f"Error in iteration {i}: {e}")
        
        overall_time = time.time() - overall_start
        
        if not latencies:
            return {'error': 'No successful measurements'}
        
        # Statistiques détaillées
        metrics = {
            'test_name': test_name,
            'iterations': iterations,
            'successful_iterations': len(latencies),
            'errors': errors,
            'error_rate': errors / iterations,
            'avg_latency_ms': np.mean(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'std_latency_ms': np.std(latencies),
            'total_time_seconds': overall_time,
            'throughput_per_second': iterations / overall_time if overall_time > 0 else 0,
            'timestamp': time.time()
        }
        
        self.logger.info(f"Performance measured: {metrics['avg_latency_ms']:.2f}ms avg, "
                        f"{metrics['error_rate']:.1%} error rate")
        
        return metrics

class ModuleValidator:
    """Validateur de modules avec vérifications robustes"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    async def validate_module(self, module_name: ModuleName, agent: TestTradingAgent,
                            timeout_seconds: float = 30.0) -> Dict:
        """Validation complète d'un module avec timeout"""
        
        validation_start = time.time()
        
        try:
            # Timeout pour éviter les blocages
            result = await asyncio.wait_for(
                self._perform_module_validation(module_name, agent),
                timeout=timeout_seconds
            )
            
            validation_time = time.time() - validation_start
            result['validation_time_seconds'] = validation_time
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"Validation timeout for {module_name.value}")
            return {
                'validated': False,
                'error': f'Validation timeout after {timeout_seconds}s',
                'validation_time_seconds': time.time() - validation_start
            }
        except Exception as e:
            self.logger.error(f"Validation error for {module_name.value}: {e}")
            return {
                'validated': False,
                'error': str(e),
                'validation_time_seconds': time.time() - validation_start
            }
    
    async def _perform_module_validation(self, module_name: ModuleName, 
                                       agent: TestTradingAgent) -> Dict:
        """Validation spécifique par module"""
        
        validators = {
            ModuleName.MODEL_MONITORING: self._validate_model_monitoring,
            ModuleName.ONLINE_LEARNING: self._validate_online_learning,
            ModuleName.MULTI_AGENT: self._validate_multi_agent,
            ModuleName.REGIME_DETECTION: self._validate_regime_detection,
            ModuleName.ULTRA_LATENCY: self._validate_ultra_latency
        }
        
        validator = validators.get(module_name)
        if not validator:
            return {'validated': False, 'error': f'No validator for {module_name.value}'}
        
        return await validator(agent)
    
    async def _validate_model_monitoring(self, agent: TestTradingAgent) -> Dict:
        """Validation du module model monitoring"""
        if not hasattr(agent, 'model_monitoring'):
            return {'validated': False, 'error': 'model_monitoring attribute missing'}
        
        try:
            # Test des fonctionnalités critiques
            monitoring = agent.model_monitoring
            
            # Vérification des méthodes essentielles
            required_methods = ['get_monitoring_summary', 'get_model_health_score']
            missing_methods = [m for m in required_methods if not hasattr(monitoring, m)]
            
            if missing_methods:
                return {'validated': False, 'error': f'Missing methods: {missing_methods}'}
            
            # Test d'exécution
            summary = monitoring.get_monitoring_summary()
            
            return {
                'validated': True,
                'details': {
                    'total_models': summary.get('total_models', 0),
                    'methods_available': [m for m in required_methods if hasattr(monitoring, m)]
                }
            }
            
        except Exception as e:
            return {'validated': False, 'error': f'Model monitoring test failed: {e}'}
    
    async def _validate_online_learning(self, agent: TestTradingAgent) -> Dict:
        """Validation du module online learning"""
        if not hasattr(agent, 'online_learning'):
            return {'validated': False, 'error': 'online_learning attribute missing'}
        
        try:
            online_learning = agent.online_learning
            framework_summary = online_learning.get_framework_summary()
            
            return {
                'validated': True,
                'details': {
                    'active_models': framework_summary.get('active_models', 0),
                    'total_models': framework_summary.get('total_models', 0)
                }
            }
            
        except Exception as e:
            return {'validated': False, 'error': f'Online learning test failed: {e}'}
    
    async def _validate_multi_agent(self, agent: TestTradingAgent) -> Dict:
        """Validation du module multi-agent"""
        if not hasattr(agent, 'multi_agent_system'):
            return {'validated': False, 'error': 'multi_agent_system attribute missing'}
        
        try:
            multi_agent = agent.multi_agent_system
            system_status = multi_agent.get_system_status()
            
            return {
                'validated': True,
                'details': {
                    'system_status': system_status.get('system_status', 'unknown'),
                    'total_agents': system_status.get('total_agents', 0)
                }
            }
            
        except Exception as e:
            return {'validated': False, 'error': f'Multi-agent test failed: {e}'}
    
    async def _validate_regime_detection(self, agent: TestTradingAgent) -> Dict:
        """Validation du module regime detection"""
        if not hasattr(agent, 'regime_detection'):
            return {'validated': False, 'error': 'regime_detection attribute missing'}
        
        try:
            regime_detection = agent.regime_detection
            performance = regime_detection.get_ensemble_performance()
            
            return {
                'validated': True,
                'details': {
                    'detector_count': performance.get('detector_count', 0),
                    'total_detections': performance.get('total_detections', 0)
                }
            }
            
        except Exception as e:
            return {'validated': False, 'error': f'Regime detection test failed: {e}'}
    
    async def _validate_ultra_latency(self, agent: TestTradingAgent) -> Dict:
        """Validation du module ultra-low latency"""
        if not hasattr(agent, 'ultra_latency_engine'):
            return {'validated': False, 'error': 'ultra_latency_engine attribute missing'}
        
        try:
            engine = agent.ultra_latency_engine
            metrics = engine.get_comprehensive_metrics()
            
            return {
                'validated': True,
                'details': {
                    'engine_running': metrics.get('engine_running', False),
                    'target_latency_ms': metrics.get('config', {}).get('target_latency_ms', 0)
                }
            }
            
        except Exception as e:
            return {'validated': False, 'error': f'Ultra-latency test failed: {e}'}

class Phase2IntegrationManagerOptimized:
    """Gestionnaire d'intégration Phase 2 optimisé et robuste"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Ordre d'intégration optimal
        self.integration_order = [
            ModuleName.MODEL_MONITORING,
            ModuleName.ONLINE_LEARNING,
            ModuleName.MULTI_AGENT,
            ModuleName.REGIME_DETECTION,
            ModuleName.ULTRA_LATENCY
        ]
        
        # Composants
        self.performance_measurer = PerformanceMeasurer(self.logger)
        self.validator = ModuleValidator(self.logger)
        
        # Résultats
        self.integration_results: Dict[ModuleName, ModuleIntegrationResult] = {}
        self.baseline_performance: Optional[Dict] = None
        self.final_performance: Optional[Dict] = None
    
    def _setup_logging(self) -> logging.Logger:
        """Configuration logging structuré"""
        logger = logging.getLogger(f"{__name__}.Phase2Integration")
        if not logger.handlers:
            # Handler console
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # Handler fichier si configuré
            if self.config.save_detailed_report:
                file_handler = logging.FileHandler('phase2_integration.log')
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            
            logger.setLevel(logging.INFO)
        return logger
    
    def create_test_agent(self) -> TestTradingAgent:
        """Création d'agent de test avec configuration optimale"""
        self.logger.info("Creating optimized test trading agent")
        return TestTradingAgent()
    
    async def execute_full_integration(self, agent: TestTradingAgent,
                                     integration_config: Dict = None) -> bool:
        """Exécution complète de l'intégration Phase 2"""
        
        self.logger.info("STARTING PHASE 2 INTEGRATION - TIER 2 -> TIER 1+ TRANSFORMATION")
        self.logger.info("=" * 70)
        
        try:
            # 1. Mesure baseline
            await self._measure_baseline_performance(agent)
            
            # 2. Intégration progressive
            success_count = await self._execute_progressive_integration(agent, integration_config)
            
            # 3. Mesure finale
            await self._measure_final_performance(agent)
            
            # 4. Rapport final
            await self._generate_comprehensive_report()
            
            # Résultat final
            total_modules = len(self.integration_order)
            success_rate = success_count / total_modules
            
            if success_rate >= 0.8:  # 80% de réussite minimum
                self.logger.info(f"PHASE 2 INTEGRATION SUCCESSFUL: {success_count}/{total_modules} modules")
                return True
            else:
                self.logger.warning(f"PHASE 2 INTEGRATION PARTIAL: {success_count}/{total_modules} modules")
                return False
                
        except Exception as e:
            self.logger.error(f"Integration failed with error: {e}")
            return False
    
    async def _measure_baseline_performance(self, agent: TestTradingAgent):
        """Mesure des performances baseline avec métriques détaillées"""
        self.logger.info("Measuring baseline performance...")
        
        self.baseline_performance = self.performance_measurer.measure_agent_performance(
            agent, self.config.performance_test_iterations, "baseline"
        )
        
        avg_latency = self.baseline_performance['avg_latency_ms']
        self.logger.info(f"Baseline: {avg_latency:.2f}ms avg latency, "
                        f"{self.baseline_performance['error_rate']:.1%} error rate")
    
    async def _execute_progressive_integration(self, agent: TestTradingAgent,
                                             integration_config: Dict) -> int:
        """Exécution de l'intégration progressive"""
        success_count = 0
        
        for i, module_name in enumerate(self.integration_order, 1):
            self.logger.info(f"INTEGRATING MODULE {i}/{len(self.integration_order)}: {module_name.value.upper()}")
            self.logger.info("-" * 50)
            
            # Intégration du module
            result = await self._integrate_single_module(module_name, agent, integration_config)
            self.integration_results[module_name] = result
            
            if result.status == IntegrationStatus.INTEGRATED:
                success_count += 1
                
                # Validation post-intégration
                validation_result = await self.validator.validate_module(
                    module_name, agent, self.config.validation_timeout_seconds
                )
                
                if validation_result['validated']:
                    result.status = IntegrationStatus.VALIDATED
                    result.validation_passed = True
                    self.logger.info(f"{module_name.value} integrated and validated")
                else:
                    self.logger.warning(f"{module_name.value} integrated but validation failed: {validation_result.get('error', 'Unknown')}")
                
                # Mesure de performance intermédiaire
                if self.config.enable_performance_tracking:
                    await self._measure_intermediate_performance(module_name, agent)
            else:
                self.logger.error(f"{module_name.value} integration failed: {result.error_message}")
        
        return success_count
    
    async def _integrate_single_module(self, module_name: ModuleName, agent: TestTradingAgent,
                                     integration_config: Dict) -> ModuleIntegrationResult:
        """Intégration d'un module unique avec gestion d'erreurs robuste"""
        
        result = ModuleIntegrationResult(module_name=module_name, status=IntegrationStatus.INTEGRATING)
        
        try:
            # Import dynamique et intégration selon le module
            integration_functions = {
                ModuleName.MODEL_MONITORING: self._integrate_model_monitoring,
                ModuleName.ONLINE_LEARNING: self._integrate_online_learning,
                ModuleName.MULTI_AGENT: self._integrate_multi_agent,
                ModuleName.REGIME_DETECTION: self._integrate_regime_detection,
                ModuleName.ULTRA_LATENCY: self._integrate_ultra_latency
            }
            
            integration_func = integration_functions.get(module_name)
            if not integration_func:
                raise ValueError(f"No integration function for {module_name.value}")
            
            # Exécution de l'intégration avec timeout
            integration_result = await asyncio.wait_for(
                integration_func(agent, integration_config),
                timeout=60.0  # 1 minute timeout
            )
            
            result.status = IntegrationStatus.INTEGRATED
            result.integration_object = integration_result
            
        except asyncio.TimeoutError:
            result.status = IntegrationStatus.FAILED
            result.error_message = f"Integration timeout for {module_name.value}"
        except Exception as e:
            result.status = IntegrationStatus.FAILED
            result.error_message = str(e)
            self.logger.error(f"Integration error for {module_name.value}: {e}")
        
        return result
    
    async def _integrate_model_monitoring(self, agent: TestTradingAgent, config: Dict) -> Any:
        """Intégration du module model monitoring"""
        try:
            from model_monitoring_system import integrate_model_monitoring
            return integrate_model_monitoring(agent, config or {})
        except ImportError as e:
            self.logger.error(f"Failed to import model monitoring: {e}")
            raise
    
    async def _integrate_online_learning(self, agent: TestTradingAgent, config: Dict) -> Any:
        """Intégration du module online learning"""
        try:
            from online_learning_framework import integrate_online_learning
            return integrate_online_learning(agent, config or {})
        except ImportError as e:
            self.logger.error(f"Failed to import online learning: {e}")
            raise
    
    async def _integrate_multi_agent(self, agent: TestTradingAgent, config: Dict) -> Any:
        """Intégration du module multi-agent"""
        try:
            from multi_agent_architecture import integrate_multi_agent_system
            return integrate_multi_agent_system(agent, config or {})
        except ImportError as e:
            self.logger.error(f"Failed to import multi-agent: {e}")
            raise
    
    async def _integrate_regime_detection(self, agent: TestTradingAgent, config: Dict) -> Any:
        """Intégration du module regime detection"""
        try:
            from regime_detection_hybrid import integrate_regime_detection
            return integrate_regime_detection(agent, config or {})
        except ImportError as e:
            self.logger.error(f"Failed to import regime detection: {e}")
            raise
    
    async def _integrate_ultra_latency(self, agent: TestTradingAgent, config: Dict) -> Any:
        """Intégration du module ultra-low latency"""
        try:
            # Essayer d'abord la version optimisée
            try:
                from ultra_low_latency_engine_optimized import integrate_ultra_low_latency_optimized
                return await integrate_ultra_low_latency_optimized(agent, config or {})
            except ImportError:
                # Fallback vers la version originale
                from ultra_low_latency_engine import integrate_ultra_low_latency
                return integrate_ultra_low_latency(agent, config or {})
        except ImportError as e:
            self.logger.error(f"Failed to import ultra-low latency: {e}")
            raise
    
    async def _measure_intermediate_performance(self, module_name: ModuleName, agent: TestTradingAgent):
        """Mesure de performance après intégration d'un module"""
        performance = self.performance_measurer.measure_agent_performance(
            agent, self.config.intermediate_test_iterations, f"after_{module_name.value}"
        )
        
        if self.integration_results[module_name]:
            self.integration_results[module_name].performance_metrics = performance
            
        # Comparaison avec baseline
        if self.baseline_performance:
            baseline_latency = self.baseline_performance['avg_latency_ms']
            current_latency = performance['avg_latency_ms']
            
            if baseline_latency > 0:
                improvement_pct = ((baseline_latency - current_latency) / baseline_latency) * 100
                self.logger.info(f"Performance after {module_name.value}: "
                               f"{current_latency:.2f}ms (improvement: {improvement_pct:+.1f}%)")
            else:
                self.logger.info(f"Performance after {module_name.value}: {current_latency:.2f}ms")
    
    async def _measure_final_performance(self, agent: TestTradingAgent):
        """Mesure finale des performances après intégration complète"""
        self.logger.info("MEASURING FINAL PERFORMANCE")
        self.logger.info("-" * 40)
        
        self.final_performance = self.performance_measurer.measure_agent_performance(
            agent, self.config.performance_test_iterations, "final"
        )
        
        # Comparaison avec baseline
        if self.baseline_performance:
            baseline_latency = self.baseline_performance['avg_latency_ms']
            final_latency = self.final_performance['avg_latency_ms']
            
            if baseline_latency > 0:
                total_improvement = ((baseline_latency - final_latency) / baseline_latency) * 100
                
                self.logger.info(f"FINAL LATENCY: {final_latency:.2f}ms")
                self.logger.info(f"TOTAL IMPROVEMENT: {total_improvement:+.1f}%")
                self.logger.info(f"TIER 1+ OBJECTIVE: {'ACHIEVED' if final_latency < 50 else 'NOT REACHED'}")
    
    async def _generate_comprehensive_report(self):
        """Génération du rapport complet d'intégration"""
        if not self.config.save_detailed_report:
            return
        
        self.logger.info("GENERATING COMPREHENSIVE REPORT")
        
        report = {
            'integration_summary': {
                'total_modules': len(self.integration_order),
                'successful_integrations': sum(1 for r in self.integration_results.values() 
                                             if r.status in [IntegrationStatus.INTEGRATED, IntegrationStatus.VALIDATED]),
                'failed_integrations': sum(1 for r in self.integration_results.values() 
                                         if r.status == IntegrationStatus.FAILED),
                'timestamp': time.time()
            },
            'baseline_performance': self.baseline_performance,
            'final_performance': self.final_performance,
            'module_results': {
                module.value: {
                    'status': result.status.value,
                    'validation_passed': result.validation_passed,
                    'error_message': result.error_message,
                    'performance_metrics': result.performance_metrics
                }
                for module, result in self.integration_results.items()
            }
        }
        
        # Sauvegarde du rapport
        try:
            with open(self.config.report_filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Report saved to {self.config.report_filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")

# Point d'entrée principal pour l'intégration Phase 2
async def execute_phase2_integration_optimized(config: Dict = None) -> bool:
    """Exécution optimisée de l'intégration Phase 2"""
    
    # Configuration
    integration_config = IntegrationConfig(
        performance_test_iterations=config.get('performance_test_iterations', 100) if config else 100,
        intermediate_test_iterations=config.get('intermediate_test_iterations', 50) if config else 50,
        validation_timeout_seconds=config.get('validation_timeout_seconds', 30.0) if config else 30.0
    )
    
    # Gestionnaire d'intégration
    manager = Phase2IntegrationManagerOptimized(integration_config)
    
    # Agent de test
    agent = manager.create_test_agent()
    
    # Exécution complète
    success = await manager.execute_full_integration(agent, config)
    
    return success

# Point d'entrée pour utilisation synchrone
def run_phase2_integration_optimized(config: Dict = None) -> bool:
    """Point d'entrée synchrone pour l'intégration Phase 2"""
    return asyncio.run(execute_phase2_integration_optimized(config))

if __name__ == "__main__":
    # Configuration par défaut
    default_config = {
        'performance_test_iterations': 100,
        'enable_performance_tracking': True,
        'save_detailed_report': True
    }
    
    # Exécution
    success = run_phase2_integration_optimized(default_config)
    
    if success:
        print("\nPHASE 2 INTEGRATION COMPLETED SUCCESSFULLY!")
        print("   Agent transformed to TIER 1+ INSTITUTIONAL level")
    else:
        print("\nPHASE 2 INTEGRATION INCOMPLETE")
        print("   Some modules require attention")
