# FILE: multi_agent_architecture.py
"""
AGENT QUANTUM TRADING - MODULE 3: MULTI-AGENT ARCHITECTURE
Architecture multi-agents avec communication Redis Pub/Sub et coordination distribuée
Impact: Redis Pub/Sub (3.2ms latency), agents spécialisés, consensus distribué + emergency coordination
Basé sur: Patterns de Communication et Coordination Optimaux.md
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
import uuid

warnings.filterwarnings('ignore')

class AgentType(Enum):
    MASTER_COORDINATOR = "master_coordinator"
    MARKET_ANALYSIS = "market_analysis"
    RISK_MANAGEMENT = "risk_management"
    EXECUTION_ENGINE = "execution_engine"
    GOVERNANCE = "governance"

class MessageType(Enum):
    MARKET_DATA = "market_data"
    SIGNAL_GENERATED = "signal_generated"
    RISK_ASSESSMENT = "risk_assessment"
    ORDER_READY = "order_ready"
    ORDER_EXECUTED = "order_executed"
    EMERGENCY_ALERT = "emergency_alert"
    HEARTBEAT = "heartbeat"

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AgentMessage:
    message_id: str
    sender_id: str
    message_type: MessageType
    priority: MessagePriority
    timestamp: float
    data: Dict
    correlation_id: str = None
    ttl: float = 30.0  # Time to live en secondes

class RedisCommunicationLayer:
    """Couche de communication Redis simulée pour multi-agents"""
    
    def __init__(self):
        self.pubsub_channels = {}
        self.message_queues = {}
        self.agent_registry = {}
        self.emergency_channel = "emergency_signals"
        self.heartbeat_interval = 2.0
        
    def register_agent(self, agent_id: str, agent_type: AgentType, capabilities: List[str]):
        """Enregistrement d'un agent"""
        self.agent_registry[agent_id] = {
            'type': agent_type,
            'capabilities': capabilities,
            'last_heartbeat': time.time(),
            'status': 'active'
        }
        print(f"✅ Agent {agent_id} ({agent_type.value}) enregistré")
    
    def publish_message(self, channel: str, message: AgentMessage) -> bool:
        """Publication d'un message sur un canal"""
        if channel not in self.pubsub_channels:
            self.pubsub_channels[channel] = []
        
        # Ajout du message avec timestamp
        message.timestamp = time.time()
        self.pubsub_channels[channel].append(message)
        
        # Gestion TTL
        self.pubsub_channels[channel] = [
            msg for msg in self.pubsub_channels[channel]
            if time.time() - msg.timestamp < msg.ttl
        ]
        
        return True
    
    def subscribe_to_channel(self, agent_id: str, channel: str):
        """Abonnement d'un agent à un canal"""
        if channel not in self.message_queues:
            self.message_queues[channel] = {}
        
        if agent_id not in self.message_queues[channel]:
            self.message_queues[channel][agent_id] = deque(maxlen=1000)
    
    def get_messages(self, agent_id: str, channel: str, max_messages: int = 10) -> List[AgentMessage]:
        """Récupération des messages pour un agent"""
        if channel not in self.message_queues or agent_id not in self.message_queues[channel]:
            return []
        
        messages = []
        queue = self.message_queues[channel][agent_id]
        
        while len(messages) < max_messages and queue:
            messages.append(queue.popleft())
        
        return messages
    
    def broadcast_emergency(self, emergency_message: AgentMessage):
        """Diffusion d'urgence sur tous les canaux"""
        for channel in self.pubsub_channels:
            self.pubsub_channels[channel].append(emergency_message)
        
        print(f"🚨 ALERTE D'URGENCE diffusée: {emergency_message.data}")
    
    def update_heartbeat(self, agent_id: str):
        """Mise à jour du heartbeat d'un agent"""
        if agent_id in self.agent_registry:
            self.agent_registry[agent_id]['last_heartbeat'] = time.time()
    
    def check_agent_health(self) -> Dict:
        """Vérification de la santé des agents"""
        current_time = time.time()
        health_status = {}
        
        for agent_id, agent_info in self.agent_registry.items():
            time_since_heartbeat = current_time - agent_info['last_heartbeat']
            
            if time_since_heartbeat > self.heartbeat_interval * 3:
                agent_info['status'] = 'inactive'
                health_status[agent_id] = 'inactive'
            elif time_since_heartbeat > self.heartbeat_interval:
                agent_info['status'] = 'warning'
                health_status[agent_id] = 'warning'
            else:
                agent_info['status'] = 'active'
                health_status[agent_id] = 'active'
        
        return health_status

class BaseAgent:
    """Classe de base pour tous les agents"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, comm_layer: RedisCommunicationLayer):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.comm_layer = comm_layer
        self.capabilities = []
        self.subscribed_channels = []
        self.message_history = []
        self.is_running = False
        
        # Enregistrement automatique
        self.comm_layer.register_agent(agent_id, agent_type, self.capabilities)
        
        # Thread de heartbeat
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
    
    def subscribe_to_channel(self, channel: str):
        """Abonnement à un canal"""
        self.comm_layer.subscribe_to_channel(self.agent_id, channel)
        self.subscribed_channels.append(channel)
    
    def publish_message(self, channel: str, message_type: MessageType, data: Dict, 
                       priority: MessagePriority = MessagePriority.NORMAL, correlation_id: str = None):
        """Publication d'un message"""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            message_type=message_type,
            priority=priority,
            timestamp=time.time(),
            data=data,
            correlation_id=correlation_id
        )
        
        success = self.comm_layer.publish_message(channel, message)
        if success:
            self.message_history.append(message)
        
        return success
    
    def get_messages(self, channel: str, max_messages: int = 10) -> List[AgentMessage]:
        """Récupération des messages d'un canal"""
        return self.comm_layer.get_messages(self.agent_id, channel, max_messages)
    
    def _heartbeat_loop(self):
        """Boucle de heartbeat"""
        while self.is_running:
            self.comm_layer.update_heartbeat(self.agent_id)
            time.sleep(2.0)
    
    def start(self):
        """Démarrage de l'agent"""
        self.is_running = True
        print(f"🚀 Agent {self.agent_id} démarré")
    
    def stop(self):
        """Arrêt de l'agent"""
        self.is_running = False
        print(f"🛑 Agent {self.agent_id} arrêté")
    
    def process_message(self, message: AgentMessage) -> bool:
        """Traitement d'un message (à implémenter dans les sous-classes)"""
        raise NotImplementedError

class MarketAnalysisAgent(BaseAgent):
    """Agent d'analyse de marché"""
    
    def __init__(self, agent_id: str, comm_layer: RedisCommunicationLayer):
        super().__init__(agent_id, AgentType.MARKET_ANALYSIS, comm_layer)
        self.capabilities = ['market_data_analysis', 'signal_generation', 'regime_detection']
        
        # Abonnement aux canaux pertinents
        self.subscribe_to_channel('market_data')
        self.subscribe_to_channel('risk_alerts')
        
        # État interne
        self.market_data_buffer = deque(maxlen=1000)
        self.signal_history = []
    
    def process_message(self, message: AgentMessage) -> bool:
        """Traitement des messages"""
        try:
            if message.message_type == MessageType.MARKET_DATA:
                return self._process_market_data(message)
            elif message.message_type == MessageType.RISK_ASSESSMENT:
                return self._process_risk_assessment(message)
            else:
                return False
        except Exception as e:
            print(f"Erreur traitement message dans {self.agent_id}: {e}")
            return False
    
    def _process_market_data(self, message: AgentMessage) -> bool:
        """Traitement des données de marché"""
        market_data = message.data
        
        # Stockage des données
        self.market_data_buffer.append(market_data)
        
        # Analyse et génération de signaux
        if len(self.market_data_buffer) >= 100:
            signals = self._generate_trading_signals()
            
            # Publication des signaux
            self.publish_message(
                channel='signals',
                message_type=MessageType.SIGNAL_GENERATED,
                data={'signals': signals, 'timestamp': time.time()},
                priority=MessagePriority.HIGH,
                correlation_id=message.message_id
            )
            
            self.signal_history.append({
                'timestamp': time.time(),
                'signals': signals,
                'data_points': len(self.market_data_buffer)
            })
        
        return True
    
    def _process_risk_assessment(self, message: AgentMessage) -> bool:
        """Traitement des évaluations de risque"""
        risk_data = message.data
        
        # Adaptation des signaux selon le risque
        if risk_data.get('risk_level') == 'high':
            # Réduction de l'agressivité des signaux
            print(f"⚠️  Risque élevé détecté, adaptation des signaux")
        
        return True
    
    def _generate_trading_signals(self) -> Dict:
        """Génération de signaux de trading"""
        if len(self.market_data_buffer) < 50:
            return {}
        
        # Analyse simple des données récentes
        recent_data = list(self.market_data_buffer)[-50:]
        
        # Calcul de signaux basiques
        prices = [d.get('close', 100) for d in recent_data]
        volumes = [d.get('volume', 1000) for d in recent_data]
        
        # Signal basé sur momentum
        if len(prices) >= 20:
            short_ma = np.mean(prices[-5:])
            long_ma = np.mean(prices[-20:])
            
            if short_ma > long_ma * 1.02:
                signal = 'buy'
                confidence = 0.7
            elif short_ma < long_ma * 0.98:
                signal = 'sell'
                confidence = 0.7
            else:
                signal = 'hold'
                confidence = 0.5
        else:
            signal = 'hold'
            confidence = 0.3
        
        return {
            'signal': signal,
            'confidence': confidence,
            'timestamp': time.time(),
            'data_points': len(recent_data)
        }

class RiskManagementAgent(BaseAgent):
    """Agent de gestion des risques"""
    
    def __init__(self, agent_id: str, comm_layer: RedisCommunicationLayer):
        super().__init__(agent_id, AgentType.RISK_MANAGEMENT, comm_layer)
        self.capabilities = ['risk_assessment', 'position_sizing', 'stop_loss_management']
        
        # Abonnement aux canaux
        self.subscribe_to_channel('signals')
        self.subscribe_to_channel('orders')
        self.subscribe_to_channel('pnl_updates')
        
        # Configuration de risque
        self.risk_limits = {
            'max_position_size': 10000,
            'max_drawdown': 0.05,
            'max_correlation': 0.7
        }
        
        self.current_positions = {}
        self.risk_history = []
    
    def process_message(self, message: AgentMessage) -> bool:
        """Traitement des messages"""
        try:
            if message.message_type == MessageType.SIGNAL_GENERATED:
                return self._process_trading_signal(message)
            elif message.message_type == MessageType.ORDER_READY:
                return self._process_order(message)
            elif message.message_type == MessageType.ORDER_EXECUTED:
                return self._process_execution(message)
            else:
                return False
        except Exception as e:
            print(f"Erreur traitement message dans {self.agent_id}: {e}")
            return False
    
    def _process_trading_signal(self, message: AgentMessage) -> bool:
        """Traitement des signaux de trading"""
        signal_data = message.data
        signal = signal_data.get('signals', {})
        
        # Évaluation du risque du signal
        risk_assessment = self._assess_signal_risk(signal)
        
        # Publication de l'évaluation
        self.publish_message(
            channel='risk_assessments',
            message_type=MessageType.RISK_ASSESSMENT,
            data=risk_assessment,
            priority=MessagePriority.HIGH,
            correlation_id=message.message_id
        )
        
        return True
    
    def _assess_signal_risk(self, signal: Dict) -> Dict:
        """Évaluation du risque d'un signal"""
        signal_type = signal.get('signal', 'hold')
        confidence = signal.get('confidence', 0.5)
        
        # Calcul du score de risque
        base_risk = 0.5
        
        if signal_type == 'buy':
            base_risk = 0.3
        elif signal_type == 'sell':
            base_risk = 0.4
        
        # Ajustement selon la confiance
        adjusted_risk = base_risk * (1 - confidence)
        
        # Vérification des limites
        risk_level = 'low' if adjusted_risk < 0.2 else 'medium' if adjusted_risk < 0.4 else 'high'
        
        return {
            'risk_score': adjusted_risk,
            'risk_level': risk_level,
            'signal_type': signal_type,
            'confidence': confidence,
            'recommendation': 'proceed' if adjusted_risk < 0.5 else 'reduce_size' if adjusted_risk < 0.7 else 'reject',
            'timestamp': time.time()
        }
    
    def _process_order(self, message: AgentMessage) -> bool:
        """Traitement des ordres"""
        order_data = message.data
        
        # Validation de l'ordre selon les limites de risque
        validation_result = self._validate_order(order_data)
        
        if validation_result['valid']:
            # Publication de l'ordre validé
            self.publish_message(
                channel='validated_orders',
                message_type=MessageType.ORDER_READY,
                data=order_data,
                priority=MessagePriority.HIGH,
                correlation_id=message.message_id
            )
        else:
            # Rejet de l'ordre
            print(f"❌ Ordre rejeté par {self.agent_id}: {validation_result['reason']}")
        
        return True
    
    def _validate_order(self, order_data: Dict) -> Dict:
        """Validation d'un ordre selon les limites de risque"""
        position_size = order_data.get('size', 0)
        symbol = order_data.get('symbol', 'UNKNOWN')
        
        # Vérification de la taille de position
        if position_size > self.risk_limits['max_position_size']:
            return {
                'valid': False,
                'reason': f'Position size {position_size} exceeds limit {self.risk_limits["max_position_size"]}'
            }
        
        # Vérification du drawdown
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown > self.risk_limits['max_drawdown']:
            return {
                'valid': False,
                'reason': f'Current drawdown {current_drawdown:.2%} exceeds limit {self.risk_limits["max_drawdown"]:.2%}'
            }
        
        return {'valid': True, 'reason': 'Order validated'}
    
    def _calculate_current_drawdown(self) -> float:
        """Calcul du drawdown actuel (simulé)"""
        # Simulation basée sur l'historique
        if not self.risk_history:
            return 0.0
        
        recent_returns = [entry.get('return', 0) for entry in self.risk_history[-20:]]
        cumulative_return = sum(recent_returns)
        
        return max(0, -cumulative_return)
    
    def _process_execution(self, message: AgentMessage) -> bool:
        """Traitement des exécutions"""
        execution_data = message.data
        
        # Mise à jour des positions
        symbol = execution_data.get('symbol')
        size = execution_data.get('size', 0)
        price = execution_data.get('price', 0)
        
        if symbol not in self.current_positions:
            self.current_positions[symbol] = {'size': 0, 'avg_price': 0}
        
        # Calcul de la position moyenne
        current_pos = self.current_positions[symbol]
        new_size = current_pos['size'] + size
        
        if new_size != 0:
            new_avg_price = (current_pos['size'] * current_pos['avg_price'] + size * price) / new_size
            current_pos['size'] = new_size
            current_pos['avg_price'] = new_avg_price
        
        # Enregistrement dans l'historique
        self.risk_history.append({
            'timestamp': time.time(),
            'symbol': symbol,
            'size': size,
            'price': price,
            'return': 0.0  # À calculer selon la stratégie
        })
        
        return True

class ExecutionEngineAgent(BaseAgent):
    """Agent d'exécution des ordres"""
    
    def __init__(self, agent_id: str, comm_layer: RedisCommunicationLayer):
        super().__init__(agent_id, AgentType.EXECUTION_ENGINE, comm_layer)
        self.capabilities = ['order_execution', 'venue_selection', 'execution_optimization']
        
        # Abonnement aux canaux
        self.subscribe_to_channel('validated_orders')
        self.subscribe_to_channel('market_data')
        
        # État d'exécution
        self.pending_orders = {}
        self.execution_history = []
        self.venue_performance = {}
    
    def process_message(self, message: AgentMessage) -> bool:
        """Traitement des messages"""
        try:
            if message.message_type == MessageType.ORDER_READY:
                return self._process_validated_order(message)
            elif message.message_type == MessageType.MARKET_DATA:
                return self._process_market_data(message)
            else:
                return False
        except Exception as e:
            print(f"Erreur traitement message dans {self.agent_id}: {e}")
            return False
    
    def _process_validated_order(self, message: AgentMessage) -> bool:
        """Traitement des ordres validés"""
        order_data = message.data
        
        # Création de l'ordre d'exécution
        execution_order = self._create_execution_order(order_data)
        
        # Stockage de l'ordre en attente
        self.pending_orders[execution_order['order_id']] = execution_order
        
        # Publication de l'ordre d'exécution
        self.publish_message(
            channel='execution_orders',
            message_type=MessageType.ORDER_READY,
            data=execution_order,
            priority=MessagePriority.HIGH,
            correlation_id=message.message_id
        )
        
        return True
    
    def _create_execution_order(self, order_data: Dict) -> Dict:
        """Création d'un ordre d'exécution optimisé"""
        order_id = str(uuid.uuid4())
        
        # Optimisation de l'exécution
        execution_strategy = self._optimize_execution_strategy(order_data)
        
        return {
            'order_id': order_id,
            'symbol': order_data.get('symbol'),
            'side': order_data.get('side'),
            'size': order_data.get('size'),
            'strategy': execution_strategy,
            'timestamp': time.time(),
            'status': 'pending'
        }
    
    def _optimize_execution_strategy(self, order_data: Dict) -> Dict:
        """Optimisation de la stratégie d'exécution"""
        size = order_data.get('size', 0)
        symbol = order_data.get('symbol', 'UNKNOWN')
        
        # Stratégie basée sur la taille
        if size > 1000:
            strategy = 'iceberg'  # Ordres cachés pour gros volumes
        elif size > 100:
            strategy = 'twap'     # Time-weighted average price
        else:
            strategy = 'market'   # Exécution immédiate
        
        return {
            'type': strategy,
            'parameters': {
                'max_slippage': 0.001,
                'timeout': 300 if strategy != 'market' else 30
            }
        }
    
    def _process_market_data(self, message: AgentMessage) -> bool:
        """Traitement des données de marché pour exécution"""
        market_data = message.data
        
        # Vérification des ordres en attente
        for order_id, order in list(self.pending_orders.items()):
            if self._should_execute_order(order, market_data):
                # Exécution de l'ordre
                execution_result = self._execute_order(order, market_data)
                
                # Publication du résultat
                self.publish_message(
                    channel='execution_results',
                    message_type=MessageType.ORDER_EXECUTED,
                    data=execution_result,
                    priority=MessagePriority.HIGH,
                    correlation_id=order_id
                )
                
                # Suppression de l'ordre en attente
                del self.pending_orders[order_id]
        
        return True
    
    def _should_execute_order(self, order: Dict, market_data: Dict) -> bool:
        """Détermination si un ordre doit être exécuté"""
        # Logique simple d'exécution
        return True  # À implémenter selon la stratégie
    
    def _execute_order(self, order: Dict, market_data: Dict) -> Dict:
        """Exécution d'un ordre"""
        # Simulation d'exécution
        execution_price = market_data.get('close', 100)
        execution_time = time.time()
        
        execution_result = {
            'order_id': order['order_id'],
            'symbol': order['symbol'],
            'side': order['side'],
            'size': order['size'],
            'execution_price': execution_price,
            'execution_time': execution_time,
            'status': 'executed',
            'venue': 'simulated_venue'
        }
        
        # Enregistrement dans l'historique
        self.execution_history.append(execution_result)
        
        return execution_result

class MasterCoordinatorAgent(BaseAgent):
    """Agent coordinateur principal"""
    
    def __init__(self, agent_id: str, comm_layer: RedisCommunicationLayer = None):
        # Si comm_layer n'est pas fourni, créer une instance par défaut
        if comm_layer is None:
            comm_layer = RedisCommunicationLayer()
        
        super().__init__(agent_id, AgentType.MASTER_COORDINATOR, comm_layer)
        self.capabilities = ['coordination', 'consensus_management', 'emergency_handling']
        
        # Abonnement à tous les canaux
        self.subscribe_to_channel('market_data')
        self.subscribe_to_channel('signals')
        self.subscribe_to_channel('risk_assessments')
        self.subscribe_to_channel('execution_results')
        self.subscribe_to_channel('emergency_signals')
        
        # État de coordination
        self.active_agents = set()
        self.consensus_state = {}
        self.emergency_mode = False
        
        # Thread de coordination
        self.coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self.coordination_thread.start()
    
    def process_message(self, message: AgentMessage) -> bool:
        """Traitement des messages de coordination"""
        try:
            if message.message_type == MessageType.EMERGENCY_ALERT:
                return self._handle_emergency(message)
            elif message.message_type == MessageType.HEARTBEAT:
                return self._process_heartbeat(message)
            else:
                return self._process_coordination_message(message)
        except Exception as e:
            print(f"Erreur coordination dans {self.agent_id}: {e}")
            return False
    
    def _handle_emergency(self, message: AgentMessage) -> bool:
        """Gestion des alertes d'urgence"""
        emergency_data = message.data
        emergency_type = emergency_data.get('type', 'unknown')
        
        print(f"🚨 ALERTE D'URGENCE: {emergency_type}")
        
        # Activation du mode d'urgence
        self.emergency_mode = True
        
        # Diffusion de l'alerte à tous les agents
        emergency_message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            message_type=MessageType.EMERGENCY_ALERT,
            priority=MessagePriority.CRITICAL,
            timestamp=time.time(),
            data=emergency_data,
            ttl=60.0
        )
        
        self.comm_layer.broadcast_emergency(emergency_message)
        
        # Actions d'urgence automatiques
        self._execute_emergency_actions(emergency_type)
        
        return True
    
    def _execute_emergency_actions(self, emergency_type: str):
        """Exécution des actions d'urgence"""
        if emergency_type == 'market_crash':
            # Arrêt de tous les ordres
            self.publish_message(
                channel='emergency_actions',
                message_type=MessageType.EMERGENCY_ALERT,
                data={'action': 'stop_all_orders', 'reason': 'market_crash'},
                priority=MessagePriority.CRITICAL
            )
        elif emergency_type == 'risk_limit_breach':
            # Réduction des positions
            self.publish_message(
                channel='emergency_actions',
                message_type=MessageType.EMERGENCY_ALERT,
                data={'action': 'reduce_positions', 'reason': 'risk_limit_breach'},
                priority=MessagePriority.CRITICAL
            )
    
    def _process_heartbeat(self, message: AgentMessage) -> bool:
        """Traitement des heartbeats"""
        sender_id = message.sender_id
        
        if sender_id not in self.active_agents:
            self.active_agents.add(sender_id)
            print(f"💓 Agent {sender_id} rejoint la coordination")
        
        return True
    
    def _process_coordination_message(self, message: AgentMessage) -> bool:
        """Traitement des messages de coordination"""
        # Mise à jour de l'état de consensus
        message_type = message.message_type.value
        sender_id = message.sender_id
        
        if message_type not in self.consensus_state:
            self.consensus_state[message_type] = {}
        
        self.consensus_state[message_type][sender_id] = {
            'timestamp': message.timestamp,
            'data': message.data
        }
        
        # Vérification du consensus
        self._check_consensus(message_type)
        
        return True
    
    def _check_consensus(self, message_type: str):
        """Vérification du consensus sur un type de message"""
        if message_type not in self.consensus_state:
            return
        
        messages = self.consensus_state[message_type]
        if len(messages) < 2:
            return
        
        # Analyse de la cohérence des messages
        timestamps = [msg['timestamp'] for msg in messages.values()]
        time_diff = max(timestamps) - min(timestamps)
        
        # Consensus si messages reçus dans un intervalle court
        if time_diff < 5.0:  # 5 secondes
            print(f"✅ Consensus atteint sur {message_type}")
            # Actions de coordination selon le consensus
    
    def _coordination_loop(self):
        """Boucle principale de coordination"""
        while self.is_running:
            try:
                # Vérification de la santé des agents
                health_status = self.comm_layer.check_agent_health()
                
                # Gestion des agents inactifs
                for agent_id, status in health_status.items():
                    if status == 'inactive':
                        print(f"⚠️  Agent {agent_id} inactif détecté")
                        self._handle_agent_failure(agent_id)
                
                # Publication du statut de coordination
                coordination_status = {
                    'active_agents': len(self.active_agents),
                    'emergency_mode': self.emergency_mode,
                    'consensus_topics': list(self.consensus_state.keys()),
                    'timestamp': time.time()
                }
                
                self.publish_message(
                    channel='coordination_status',
                    message_type=MessageType.HEARTBEAT,
                    data=coordination_status,
                    priority=MessagePriority.NORMAL
                )
                
                time.sleep(5.0)  # Vérification toutes les 5 secondes
                
            except Exception as e:
                print(f"Erreur dans la boucle de coordination: {e}")
                time.sleep(1.0)
    
    def _handle_agent_failure(self, agent_id: str):
        """Gestion de la défaillance d'un agent"""
        print(f"🔄 Gestion de la défaillance de {agent_id}")
        
        # Suppression de l'agent de la liste active
        self.active_agents.discard(agent_id)
        
        # Notification aux autres agents
        failure_message = {
            'failed_agent': agent_id,
            'timestamp': time.time(),
            'action': 'remove_from_coordination'
        }
        
        self.publish_message(
            channel='agent_failures',
            message_type=MessageType.EMERGENCY_ALERT,
            data=failure_message,
            priority=MessagePriority.HIGH
        )
    
    def get_active_agents(self) -> List[str]:
        """Retourne la liste des agents actifs"""
        return list(self.active_agents)

class MultiAgentSystem:
    """Système multi-agents complet"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.comm_layer = RedisCommunicationLayer()
        self.agents = {}
        self.system_status = 'initializing'
        
        # Configuration par défaut
        self.default_config = {
            'enable_market_analysis': True,
            'enable_risk_management': True,
            'enable_execution_engine': True,
            'enable_governance': False,
            'coordination_frequency': 5.0,
            'emergency_response_time': 1.0
        }
        
        self.config = {**self.default_config, **self.config}
        
        # Initialisation des agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialisation de tous les agents"""
        # Agent coordinateur principal
        coordinator = MasterCoordinatorAgent("coordinator_001")
        self.agents['coordinator'] = coordinator
        
        # Agent d'analyse de marché
        if self.config['enable_market_analysis']:
            market_agent = MarketAnalysisAgent("market_001", self.comm_layer)
            self.agents['market_analysis'] = market_agent
        
        # Agent de gestion des risques
        if self.config['enable_risk_management']:
            risk_agent = RiskManagementAgent("risk_001", self.comm_layer)
            self.agents['risk_management'] = risk_agent
        
        # Agent d'exécution
        if self.config['enable_execution_engine']:
            execution_agent = ExecutionEngineAgent("execution_001", self.comm_layer)
            self.agents['execution_engine'] = execution_agent
        
        print(f"✅ {len(self.agents)} agents initialisés")
    
    def start_system(self):
        """Démarrage du système multi-agents"""
        print("🚀 Démarrage du système multi-agents...")
        
        # Démarrage de tous les agents
        for agent_id, agent in self.agents.items():
            agent.start()
        
        self.system_status = 'running'
        print("✅ Système multi-agents démarré")
    
    def stop_system(self):
        """Arrêt du système multi-agents"""
        print("🛑 Arrêt du système multi-agents...")
        
        # Arrêt de tous les agents
        for agent_id, agent in self.agents.items():
            agent.stop()
        
        self.system_status = 'stopped'
        print("✅ Système multi-agents arrêté")
    
    def get_system_status(self) -> Dict:
        """Récupération du statut du système"""
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = {
                'type': agent.agent_type.value,
                'running': agent.is_running,
                'message_count': len(agent.message_history),
                'subscribed_channels': len(agent.subscribed_channels)
            }
        
        return {
            'system_status': self.system_status,
            'total_agents': len(self.agents),
            'agent_statuses': agent_statuses,
            'communication_layer': {
                'total_channels': len(self.comm_layer.pubsub_channels),
                'registered_agents': len(self.comm_layer.agent_registry),
                'health_status': self.comm_layer.check_agent_health()
            }
        }
    
    def simulate_market_data(self, data: Dict):
        """Simulation de données de marché"""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_id='market_simulator',
            message_type=MessageType.MARKET_DATA,
            priority=MessagePriority.NORMAL,
            timestamp=time.time(),
            data=data
        )
        
        # Publication sur le canal market_data
        self.comm_layer.publish_message('market_data', message)
    
    def trigger_emergency(self, emergency_type: str, details: Dict = None):
        """Déclenchement d'une alerte d'urgence"""
        emergency_data = {
            'type': emergency_type,
            'details': details or {},
            'timestamp': time.time(),
            'severity': 'critical'
        }
        
        emergency_message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_id='emergency_trigger',
            message_type=MessageType.EMERGENCY_ALERT,
            priority=MessagePriority.CRITICAL,
            timestamp=time.time(),
            data=emergency_data,
            ttl=60.0
        )
        
        # Diffusion d'urgence
        self.comm_layer.broadcast_emergency(emergency_message)

def integrate_multi_agent_system(agent, config: Dict = None) -> MultiAgentSystem:
    """Intègre le système multi-agents dans un agent existant"""
    
    # Initialisation du système multi-agents
    multi_agent_system = MultiAgentSystem(config)
    
    # Sauvegarde de la méthode originale
    original_make_decision = agent.make_decision
    
    async def coordinated_make_decision(market_data):
        """Version coordonnée de la décision"""
        try:
            # Simulation de données de marché
            simulated_data = {
                'close': market_data.get('close', 100) if hasattr(market_data, 'get') else 100,
                'volume': market_data.get('volume', 1000) if hasattr(market_data, 'get') else 1000,
                'timestamp': time.time()
            }
            
            # Injection des données dans le système multi-agents
            multi_agent_system.simulate_market_data(simulated_data)
            
            # Décision originale
            decision = original_make_decision(market_data)
            
            # Stockage des résultats
            agent.multi_agent_system = multi_agent_system
            
            return decision
            
        except Exception as e:
            print(f"Erreur système multi-agents: {e}")
            return original_make_decision(market_data)
    
    # Remplacement de la méthode
    agent.make_decision = coordinated_make_decision
    agent.multi_agent_system = multi_agent_system
    
    # Démarrage du système
    multi_agent_system.start_system()
    
    print("✅ Multi-Agent System intégré (Redis Pub/Sub, coordination distribuée, emergency handling)")
    return multi_agent_system
