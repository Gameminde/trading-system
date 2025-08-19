"""
AGENT QUANTUM TRADING - MODULE 5 OPTIMISÉ: ULTRA-LOW LATENCY ENGINE
Version production avec architecture modulaire et gestion robuste des ressources
"""

import asyncio
import logging
import time
import threading
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol, Any, Callable
import numpy as np
import ctypes
from multiprocessing import Value, shared_memory
import weakref
import os

# Configuration centralisée et immutable
@dataclass(frozen=True)
class UltraLatencyConfig:
    """Configuration immutable pour le moteur ultra-low latency"""
    target_latency_ms: float = 50.0
    max_queue_size: int = 10000
    ring_buffer_size: int = 1024  # Changed to power of 2
    shared_memory_size: int = 1024 * 1024  # 1MB
    enable_hardware_optimization: bool = True
    event_timeout_ms: float = 1.0
    
    def __post_init__(self):
        if self.target_latency_ms <= 0:
            raise ValueError("target_latency_ms must be positive")
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if self.ring_buffer_size <= 0 or (self.ring_buffer_size & (self.ring_buffer_size - 1)) != 0:
            raise ValueError("ring_buffer_size must be a positive power of 2")

class EventType(Enum):
    """Types d'événements de trading"""
    MARKET_DATA = "market_data"
    SIGNAL_GENERATED = "signal_generated"
    ORDER_READY = "order_ready"
    ORDER_EXECUTED = "order_executed"
    RISK_ALERT = "risk_alert"

@dataclass
class TradingEvent:
    """Événement de trading avec métadonnées"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    priority: int = 1

class PerformanceTracker:
    """Tracker de performance thread-safe avec métriques avancées"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        # Métriques
        self.latencies: List[float] = []
        self.throughput_counter = 0
        self.error_counter = 0
    
    def record_latency(self, latency_ns: float):
        """Enregistrement d'une latence en nanosecondes"""
        with self._lock:
            self.latencies.append(latency_ns)
            if len(self.latencies) > self.window_size:
                self.latencies = self.latencies[-self.window_size:]
    
    def record_throughput(self):
        """Enregistrement d'un événement pour le throughput"""
        with self._lock:
            self.throughput_counter += 1
    
    def record_error(self):
        """Enregistrement d'une erreur"""
        with self._lock:
            self.error_counter += 1
    
    def get_metrics(self) -> Dict:
        """Récupération thread-safe des métriques"""
        with self._lock:
            if not self.latencies:
                return {'no_data': True}
            
            latencies_ms = [lat / 1_000_000 for lat in self.latencies]
            uptime = time.time() - self.start_time
            
            return {
                'avg_latency_ms': np.mean(latencies_ms),
                'min_latency_ms': np.min(latencies_ms),
                'max_latency_ms': np.max(latencies_ms),
                'p95_latency_ms': np.percentile(latencies_ms, 95),
                'p99_latency_ms': np.percentile(latencies_ms, 99),
                'throughput_per_sec': self.throughput_counter / uptime if uptime > 0 else 0,
                'error_rate': self.error_counter / self.throughput_counter if self.throughput_counter > 0 else 0,
                'total_events': self.throughput_counter,
                'total_errors': self.error_counter
            }

class LockFreeRingBufferOptimized:
    """Ring buffer lock-free optimisé avec gestion d'erreurs"""
    
    def __init__(self, size: int):
        if size <= 0 or (size & (size - 1)) != 0:
            raise ValueError("Size must be a positive power of 2")
        
        self.size = size
        self.mask = size - 1
        self.buffer = np.zeros(size, dtype=np.float64)
        self.head = Value(ctypes.c_longlong, 0)
        self.tail = Value(ctypes.c_longlong, 0)
        self.logger = logging.getLogger(f"{__name__}.RingBuffer")
    
    def push(self, value: float) -> bool:
        """Push atomique avec vérification de capacité"""
        current_head = self.head.value
        next_head = (current_head + 1) & self.mask
        
        if next_head == self.tail.value:
            self.logger.warning("Ring buffer full, dropping value")
            return False
        
        self.buffer[current_head] = value
        self.head.value = next_head
        return True
    
    def pop(self) -> Optional[float]:
        """Pop atomique avec vérification"""
        current_tail = self.tail.value
        
        if current_tail == self.head.value:
            return None  # Buffer vide
        
        value = self.buffer[current_tail]
        self.tail.value = (current_tail + 1) & self.mask
        return value
    
    def get_utilization(self) -> float:
        """Utilisation actuelle du buffer"""
        head = self.head.value
        tail = self.tail.value
        used = (head - tail) & self.mask
        return used / self.size

class SharedMemoryManagerOptimized:
    """Gestionnaire de mémoire partagée avec cleanup automatique"""
    
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.shm = None
        self.buffer = None
        self._cleanup_registered = False
        self.logger = logging.getLogger(f"{__name__}.SharedMemory")
        
        # Enregistrement du cleanup automatique
        self._register_cleanup()
    
    def _register_cleanup(self):
        """Enregistrement du cleanup automatique"""
        if not self._cleanup_registered:
            weakref.finalize(self, self._cleanup_shared_memory, self.name)
            self._cleanup_registered = True
    
    @staticmethod
    def _cleanup_shared_memory(name: str):
        """Nettoyage statique de la mémoire partagée"""
        try:
            shm = shared_memory.SharedMemory(name)
            shm.close()
            shm.unlink()
        except:
            pass  # Silencieux car appelé dans finalize
    
    def initialize(self) -> bool:
        """Initialisation de la mémoire partagée"""
        try:
            self.shm = shared_memory.SharedMemory(
                self.name, create=True, size=self.size
            )
            self.buffer = memoryview(self.shm.buf)
            self.logger.info(f"Shared memory initialized: {self.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize shared memory: {e}")
            return False
    
    def cleanup(self):
        """Nettoyage explicite"""
        try:
            if self.shm:
                self.shm.close()
                self.shm.unlink()
                self.shm = None
                self.buffer = None
                self.logger.info(f"Shared memory cleaned up: {self.name}")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

class EventBusOptimized:
    """Event bus optimisé avec gestion avancée des erreurs et métriques"""
    
    def __init__(self, config: UltraLatencyConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.EventBus")
        self.handlers: Dict[EventType, List[Callable]] = {}
        self.event_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.processing = False
        self.performance_tracker = PerformanceTracker()
        
    async def __aenter__(self):
        """Context manager pour démarrage automatique"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager pour arrêt automatique"""
        await self.stop()
    
    async def start(self):
        """Démarrage du bus d'événements"""
        if self.processing:
            self.logger.warning("EventBus already running")
            return
        
        self.processing = True
        self.process_task = asyncio.create_task(self._process_events_loop())
        self.logger.info("EventBus started")
    
    async def stop(self):
        """Arrêt propre du bus d'événements"""
        if not self.processing:
            return
        
        self.processing = False
        
        if hasattr(self, 'process_task'):
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("EventBus stopped")
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """Abonnement avec validation du handler"""
        if not callable(handler):
            raise ValueError("Handler must be callable")
        
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        
        self.handlers[event_type].append(handler)
        self.logger.debug(f"Handler registered for {event_type}")
    
    async def publish(self, event: TradingEvent) -> bool:
        """Publication avec gestion de backpressure"""
        try:
            self.event_queue.put_nowait(event)
            self.performance_tracker.record_throughput()
            return True
        except asyncio.QueueFull:
            self.performance_tracker.record_error()
            self.logger.warning("Event queue full, event dropped")
            return False
    
    async def _process_events_loop(self):
        """Boucle principale de traitement des événements"""
        while self.processing:
            try:
                # Timeout configuré pour éviter les blocages
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=self.config.event_timeout_ms / 1000
                )
                
                await self._handle_event(event)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Event processing error: {e}")
                self.performance_tracker.record_error()
    
    async def _handle_event(self, event: TradingEvent):
        """Gestion d'un événement avec métriques"""
        start_time = time.time_ns()
        
        try:
            handlers = self.handlers.get(event.event_type, [])
            
            if not handlers:
                self.logger.debug(f"No handlers for {event.event_type}")
                return
            
            # Exécution parallèle des handlers
            tasks = [self._safe_handler_call(handler, event) for handler in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Event handling error: {e}")
            self.performance_tracker.record_error()
        finally:
            end_time = time.time_ns()
            self.performance_tracker.record_latency(end_time - start_time)
    
    async def _safe_handler_call(self, handler: Callable, event: TradingEvent):
        """Appel sécurisé d'un handler"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            self.logger.error(f"Handler error: {e}")
            self.performance_tracker.record_error()

class MarketDataProcessor:
    """Processeur de données de marché optimisé"""
    
    def __init__(self, config: UltraLatencyConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.MarketDataProcessor")
        self.cache = {}
        self.cache_max_age = 60  # 60 secondes
    
    def process_market_data(self, data: np.ndarray) -> Dict:
        """Traitement optimisé des données de marché"""
        if len(data) == 0:
            return {}
        
        # Cache check
        data_hash = hash(data.tobytes())
        if data_hash in self.cache:
            cache_entry = self.cache[data_hash]
            if time.time() - cache_entry['timestamp'] < self.cache_max_age:
                return cache_entry['result']
        
        # Calcul des signaux
        signals = self._compute_signals(data)
        
        # Nettoyage du cache
        self._cleanup_cache()
        
        result = {'signals': signals, 'processed_at': time.time()}
        self.cache[data_hash] = {'result': result, 'timestamp': time.time()}
        
        return result
    
    def _compute_signals(self, data: np.ndarray) -> np.ndarray:
        """Calcul des signaux avec optimisations numériques"""
        if len(data) < 20:
            return np.array([0.0])
        
        # Moyennes mobiles exponentielles optimisées
        alpha = 0.1
        ema_fast = np.zeros_like(data)
        ema_slow = np.zeros_like(data)
        
        ema_fast[0] = data[0]
        ema_slow[0] = data[0]
        
        for i in range(1, len(data)):
            ema_fast[i] = alpha * data[i] + (1 - alpha) * ema_fast[i-1]
            ema_slow[i] = (alpha/2) * data[i] + (1 - alpha/2) * ema_slow[i-1]
        
        # Signaux basés sur croisement des moyennes
        signals = np.zeros_like(data)
        for i in range(1, len(data)):
            if ema_fast[i] > ema_slow[i] * 1.01:  # 1% seuil
                signals[i] = 1.0
            elif ema_fast[i] < ema_slow[i] * 0.99:
                signals[i] = -1.0
        
        return signals
    
    def _cleanup_cache(self):
        """Nettoyage du cache expiré"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry['timestamp'] > self.cache_max_age
        ]
        
        for key in expired_keys:
            del self.cache[key]

class UltraLowLatencyEngineOptimized:
    """Moteur de latence ultra-faible optimisé et modulaire"""
    
    def __init__(self, config: UltraLatencyConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Composants modulaires
        self.event_bus = EventBusOptimized(config)
        self.market_processor = MarketDataProcessor(config)
        self.performance_tracker = PerformanceTracker()
        
        # Structures optimisées
        self.ring_buffer = LockFreeRingBufferOptimized(config.ring_buffer_size)
        self.shared_memory = SharedMemoryManagerOptimized(
            "ultra_latency_engine", config.shared_memory_size
        )
        
        # État
        self.running = False
        
        # Optimisations système
        if config.enable_hardware_optimization:
            self._optimize_system()
    
    def _setup_logging(self) -> logging.Logger:
        """Configuration logging structuré"""
        logger = logging.getLogger(f"{__name__}.UltraLatencyEngine")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _optimize_system(self):
        """Optimisations système pour latence minimale"""
        try:
            if os.name == 'posix':
                os.nice(-10)
                self.logger.info("Process priority optimized (Unix)")
            elif os.name == 'nt':
                import psutil
                p = psutil.Process()
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                self.logger.info("Process priority optimized (Windows)")
        except Exception as e:
            self.logger.warning(f"System optimization partial: {e}")
    
    @asynccontextmanager
    async def managed_lifecycle(self):
        """Gestionnaire de contexte pour cycle de vie complet"""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()
    
    async def start(self):
        """Démarrage complet du moteur"""
        if self.running:
            self.logger.warning("Engine already running")
            return
        
        self.logger.info("Starting Ultra-Low Latency Engine...")
        
        # Initialisation des composants
        if not self.shared_memory.initialize():
            raise RuntimeError("Failed to initialize shared memory")
        
        # Démarrage du bus d'événements
        await self.event_bus.start()
        
        # Enregistrement des handlers
        self._setup_event_handlers()
        
        self.running = True
        self.logger.info(f"Engine started (target: {self.config.target_latency_ms}ms)")
    
    async def stop(self):
        """Arrêt propre du moteur"""
        if not self.running:
            return
        
        self.logger.info("Stopping Ultra-Low Latency Engine...")
        
        # Arrêt du bus d'événements
        await self.event_bus.stop()
        
        # Nettoyage des ressources
        self.shared_memory.cleanup()
        
        self.running = False
        self.logger.info("Engine stopped")
    
    def _setup_event_handlers(self):
        """Configuration des handlers d'événements"""
        self.event_bus.subscribe(EventType.MARKET_DATA, self._handle_market_data)
        self.event_bus.subscribe(EventType.SIGNAL_GENERATED, self._handle_signals)
        self.event_bus.subscribe(EventType.ORDER_READY, self._handle_order_ready)
    
    async def _handle_market_data(self, event: TradingEvent):
        """Handler pour données de marché"""
        try:
            market_data = event.data.get('data', np.array([]))
            
            if len(market_data) > 0:
                # Traitement via le processeur optimisé
                result = self.market_processor.process_market_data(market_data)
                
                if 'signals' in result:
                    # Publication des signaux
                    signal_event = TradingEvent(
                        EventType.SIGNAL_GENERATED,
                        {'signals': result['signals'], 'market_data': market_data},
                        correlation_id=event.correlation_id
                    )
                    
                    await self.event_bus.publish(signal_event)
                    
        except Exception as e:
            self.logger.error(f"Market data handling error: {e}")
    
    async def _handle_signals(self, event: TradingEvent):
        """Handler pour signaux générés"""
        try:
            signals = event.data.get('signals', np.array([]))
            
            if len(signals) > 0:
                # Évaluation risque rapide
                risk_ok = self._assess_risk_fast(signals)
                
                if risk_ok:
                    order_event = TradingEvent(
                        EventType.ORDER_READY,
                        {'signals': signals, 'risk_approved': True},
                        correlation_id=event.correlation_id
                    )
                    
                    await self.event_bus.publish(order_event)
                    
        except Exception as e:
            self.logger.error(f"Signal handling error: {e}")
    
    async def _handle_order_ready(self, event: TradingEvent):
        """Handler pour ordres prêts"""
        try:
            signals = event.data.get('signals', np.array([]))
            
            if len(signals) > 0:
                # Génération ordre optimisée
                order = self._generate_order_optimized(signals)
                
                execution_event = TradingEvent(
                    EventType.ORDER_EXECUTED,
                    {'order': order, 'signals': signals},
                    correlation_id=event.correlation_id
                )
                
                await self.event_bus.publish(execution_event)
                
        except Exception as e:
            self.logger.error(f"Order handling error: {e}")
    
    def _assess_risk_fast(self, signals: np.ndarray) -> bool:
        """Évaluation de risque optimisée"""
        if len(signals) == 0:
            return False
        
        # Analyse des signaux récents
        recent_signals = signals[-min(5, len(signals)):]
        signal_strength = np.mean(np.abs(recent_signals))
        
        # Seuil de risque conservateur
        return signal_strength < 0.8
    
    def _generate_order_optimized(self, signals: np.ndarray) -> Dict:
        """Génération d'ordre optimisée"""
        if len(signals) == 0:
            return {}
        
        last_signal = signals[-1]
        signal_strength = abs(last_signal)
        
        return {
            'action': 'BUY' if last_signal > 0 else 'SELL' if last_signal < 0 else 'HOLD',
            'size': min(signal_strength * 100, 1000),  # Taille basée sur force du signal
            'confidence': signal_strength,
            'timestamp': time.time()
        }
    
    async def process_market_data_async(self, market_data: np.ndarray) -> Dict:
        """Point d'entrée principal pour traitement de données"""
        if not self.running:
            raise RuntimeError("Engine not running")
        
        start_time = time.time_ns()
        
        try:
            # Création et publication de l'événement
            event = TradingEvent(
                EventType.MARKET_DATA,
                {'data': market_data}
            )
            
            success = await self.event_bus.publish(event)
            
            if not success:
                return {'status': 'error', 'reason': 'Event queue full'}
            
            # Attente courte pour traitement
            await asyncio.sleep(0.001)  # 1ms
            
            end_time = time.time_ns()
            latency_ns = end_time - start_time
            
            # Métriques
            self.performance_tracker.record_latency(latency_ns)
            
            return {
                'status': 'processed',
                'latency_ms': latency_ns / 1_000_000,
                'correlation_id': event.correlation_id,
                'ring_buffer_utilization': self.ring_buffer.get_utilization()
            }
            
        except Exception as e:
            self.logger.error(f"Market data processing error: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def get_comprehensive_metrics(self) -> Dict:
        """Métriques complètes du moteur"""
        return {
            'engine_running': self.running,
            'config': {
                'target_latency_ms': self.config.target_latency_ms,
                'max_queue_size': self.config.max_queue_size,
                'ring_buffer_size': self.config.ring_buffer_size
            },
            'performance': self.performance_tracker.get_metrics(),
            'event_bus_performance': self.event_bus.performance_tracker.get_metrics(),
            'ring_buffer_utilization': self.ring_buffer.get_utilization(),
            'cache_stats': {
                'market_processor_cache_size': len(self.market_processor.cache)
            }
        }

# Fonction d'intégration optimisée
async def integrate_ultra_low_latency_optimized(agent, config: Dict = None) -> UltraLowLatencyEngineOptimized:
    """Intégration optimisée du moteur ultra-low latency"""
    
    # Configuration
    ultra_config = UltraLatencyConfig(
        target_latency_ms=config.get('target_latency_ms', 50.0) if config else 50.0,
        max_queue_size=config.get('max_queue_size', 10000) if config else 10000,
        enable_hardware_optimization=config.get('enable_hardware_optimization', True) if config else True
    )
    
    # Moteur optimisé
    engine = UltraLowLatencyEngineOptimized(ultra_config)
    
    # Sauvegarde de la méthode originale
    original_make_decision = agent.make_decision
    
    async def enhanced_make_decision_async(market_data):
        """Version améliorée avec gestion complète du cycle de vie"""
        try:
            # Conversion des données si nécessaire
            if hasattr(market_data, 'values'):
                data_array = market_data.values
            elif isinstance(market_data, (list, tuple)):
                data_array = np.array(market_data)
            elif isinstance(market_data, np.ndarray):
                data_array = market_data
            else:
                # Fallback
                data_array = np.array([100.0])  # Prix par défaut
            
            # Traitement via le moteur
            result = await engine.process_market_data_async(data_array)
            
            # Décision originale
            decision = original_make_decision(market_data)
            
            # Stockage du moteur
            agent.ultra_latency_engine = engine
            
            return decision
            
        except Exception as e:
            logging.error(f"Ultra-low latency engine error: {e}")
            return original_make_decision(market_data)
    
    def enhanced_make_decision_sync(market_data):
        """Version synchrone avec gestion d'event loop"""
        try:
            # Gestion de l'event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Event loop déjà en cours, utiliser run_until_complete pourrait bloquer
                    task = loop.create_task(enhanced_make_decision_async(market_data))
                    return original_make_decision(market_data)  # Fallback immédiat
                else:
                    return loop.run_until_complete(enhanced_make_decision_async(market_data))
            except RuntimeError:
                # Pas d'event loop, en créer un
                return asyncio.run(enhanced_make_decision_async(market_data))
                
        except Exception as e:
            logging.error(f"Ultra-low latency sync wrapper error: {e}")
            return original_make_decision(market_data)
    
    # Remplacement approprié selon le type
    if asyncio.iscoroutinefunction(original_make_decision):
        agent.make_decision = enhanced_make_decision_async
    else:
        agent.make_decision = enhanced_make_decision_sync
    
    # Démarrage du moteur
    await engine.start()
    agent.ultra_latency_engine = engine
    
    print("✅ Ultra-Low Latency Engine optimisé intégré (architecture modulaire, cleanup automatique)")
    return engine
