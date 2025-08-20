"""
AGENT QUANTUM TRADING - MODULE 3 OPTIMISÉ: MULTI-AGENT ARCHITECTURE
Version production avec Redis réel (ou fallback thread-safe), logging et thread-safety
"""

import asyncio
import logging
import threading
import time
import uuid
import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


# Configuration centralisée
@dataclass(frozen=True)
class MultiAgentConfig:
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    heartbeat_interval: float = 2.0
    message_ttl: float = 30.0
    max_message_queue_size: int = 1000
    coordination_frequency: float = 5.0


class MessageType(Enum):
    MARKET_DATA = "market_data"
    SIGNAL_GENERATED = "signal_generated"
    HEARTBEAT = "heartbeat"
    EMERGENCY_ALERT = "emergency_alert"


class MessagePriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentMessage:
    message_id: str
    sender_id: str
    message_type: MessageType
    priority: MessagePriority
    data: Dict
    timestamp: float = time.time()
    correlation_id: Optional[str] = None
    ttl: float = 30.0

    def to_json(self) -> str:
        return json.dumps({
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'data': self.data,
            'timestamp': self.timestamp,
            'correlation_id': self.correlation_id,
            'ttl': self.ttl
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        data = json.loads(json_str)
        return cls(
            message_id=data['message_id'],
            sender_id=data['sender_id'],
            message_type=MessageType(data['message_type']),
            priority=MessagePriority(data['priority']),
            data=data['data'],
            timestamp=data.get('timestamp', time.time()),
            correlation_id=data.get('correlation_id'),
            ttl=data.get('ttl', 30.0)
        )


class ThreadSafeFallbackRedis:
    """Fallback interne thread-safe lorsque Redis n'est pas disponible."""

    def __init__(self):
        self._data: Dict[str, Dict] = {}
        self._pubsub_channels: Dict[str, List[Dict]] = {}
        self._lock = threading.RLock()

    async def ping(self):
        return True

    async def hset(self, key: str, mapping: Dict):
        with self._lock:
            self._data[key] = mapping

    async def publish(self, channel: str, message: str):
        with self._lock:
            if channel not in self._pubsub_channels:
                self._pubsub_channels[channel] = []
            self._pubsub_channels[channel].append({'type': 'message', 'data': message, 'time': time.time()})

    async def setex(self, key: str, ttl: int, value: str):
        with self._lock:
            self._data[key] = {'value': value, 'expires_at': time.time() + ttl}

    def listen_channel(self, channel: str) -> List[Dict]:
        with self._lock:
            return list(self._pubsub_channels.get(channel, []))


class RedisRealCommunicationLayer:
    """Couche de communication Redis réelle avec fallback thread-safe et logging."""

    def __init__(self, config: MultiAgentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self._redis_client = None
        self._pubsub = None
        self._lock = asyncio.Lock()
        self._agent_registry: Dict[str, Dict] = {}
        self._message_stats = {'published': 0, 'received': 0, 'errors': 0}

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    async def _ensure_redis_connection(self):
        if self._redis_client is not None:
            return
        try:
            import redis.asyncio as aioredis  # type: ignore
            self._redis_client = aioredis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=True,
                socket_connect_timeout=5
            )
            await self._redis_client.ping()
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.warning(f"Redis unavailable, using thread-safe fallback: {e}")
            self._redis_client = ThreadSafeFallbackRedis()

    async def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str]) -> bool:
        async with self._lock:
            try:
                await self._ensure_redis_connection()
                agent_info = {
                    'type': agent_type,
                    'capabilities': capabilities,
                    'last_heartbeat': time.time(),
                    'status': 'active'
                }
                await self._redis_client.hset(f"agent:{agent_id}", mapping=agent_info)  # type: ignore
                self._agent_registry[agent_id] = agent_info
                return True
            except Exception as e:
                self.logger.error(f"Failed to register agent {agent_id}: {e}")
                return False

    async def publish_message(self, channel: str, message: AgentMessage) -> bool:
        try:
            await self._ensure_redis_connection()
            json_message = message.to_json()
            await self._redis_client.publish(channel, json_message)  # type: ignore
            await self._redis_client.setex(
                f"message:{channel}:{message.message_id}", int(message.ttl), json_message
            )  # type: ignore
            self._message_stats['published'] += 1
            return True
        except Exception as e:
            self._message_stats['errors'] += 1
            self.logger.error(f"Failed to publish message: {e}")
            return False

    async def subscribe_to_channel(self, agent_id: str, channel: str) -> None:
        try:
            await self._ensure_redis_connection()
            # For real Redis, we'd create a pubsub and subscribe; fallback uses memory
            if hasattr(self._redis_client, 'pubsub'):
                if self._pubsub is None:
                    self._pubsub = self._redis_client.pubsub()  # type: ignore
                await self._pubsub.subscribe(channel)  # type: ignore
            # Fallback requires no setup
        except Exception as e:
            self.logger.error(f"Failed to subscribe {agent_id} to {channel}: {e}")

    async def get_messages(self, agent_id: str, channel: str, max_messages: int = 10) -> List[AgentMessage]:
        messages: List[AgentMessage] = []
        try:
            await self._ensure_redis_connection()
            # Real Redis: pull from pubsub
            if self._pubsub is not None:
                # Non-blocking fetch: iterate limited messages available now
                fetched = 0
                while fetched < max_messages:
                    data = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=0.001)  # type: ignore
                    if not data:
                        break
                    if data.get('type') == 'message':
                        try:
                            messages.append(AgentMessage.from_json(data['data']))
                            fetched += 1
                            self._message_stats['received'] += 1
                        except Exception:
                            self._message_stats['errors'] += 1
                            continue
            else:
                # Fallback: read from in-memory list
                raw_list = self._redis_client.listen_channel(channel)  # type: ignore
                for item in raw_list[-max_messages:]:
                    try:
                        messages.append(AgentMessage.from_json(item['data']))
                        self._message_stats['received'] += 1
                    except Exception:
                        self._message_stats['errors'] += 1
                        continue
        except Exception as e:
            self._message_stats['errors'] += 1
            self.logger.error(f"Get messages failed: {e}")
        return messages

    def get_communication_stats(self) -> Dict:
        return {
            'published_messages': self._message_stats['published'],
            'received_messages': self._message_stats['received'],
            'communication_errors': self._message_stats['errors'],
            'registered_agents': len(self._agent_registry),
        }


class AgentType(Enum):
    MARKET_ANALYSIS = "market_analysis"
    RISK_MANAGEMENT = "risk_management"
    EXECUTION_ENGINE = "execution_engine"
    MASTER_COORDINATOR = "master_coordinator"


class BaseAgentOptimized:
    """Agent de base optimisé avec thread-safety et boucles asynchrones."""

    def __init__(self, agent_id: str, agent_type: AgentType, comm_layer: RedisRealCommunicationLayer):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.comm_layer = comm_layer
        self.capabilities: List[str] = []
        self.subscribed_channels: List[str] = []
        self.message_history: List[str] = []
        self._running = threading.Event()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._message_task: Optional[asyncio.Task] = None
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.{self.agent_id}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    async def start(self):
        if self._running.is_set():
            return
        success = await self.comm_layer.register_agent(self.agent_id, self.agent_type.value, self.capabilities)
        if not success:
            raise RuntimeError("Failed to register agent")
        self._running.set()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._message_task = asyncio.create_task(self._message_loop())

    async def stop(self):
        if not self._running.is_set():
            return
        self._running.clear()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._message_task:
            self._message_task.cancel()

    async def process_message(self, message: AgentMessage) -> bool:
        return False

    async def _heartbeat_loop(self):
        while self._running.is_set():
            try:
                hb = AgentMessage(
                    message_id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    message_type=MessageType.HEARTBEAT,
                    priority=MessagePriority.NORMAL,
                    data={'heartbeat': True, 'timestamp': time.time()},
                    ttl=10.0,
                )
                await self.comm_layer.publish_message('heartbeat', hb)
                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(1.0)

    async def _message_loop(self):
        while self._running.is_set():
            try:
                for ch in self.subscribed_channels:
                    msgs = await self.comm_layer.get_messages(self.agent_id, ch, max_messages=10)
                    for m in msgs:
                        await self.process_message(m)
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Message loop error: {e}")
                await asyncio.sleep(0.2)


class MarketAnalysisAgentOptimized(BaseAgentOptimized):
    def __init__(self, agent_id: str, comm_layer: RedisRealCommunicationLayer):
        super().__init__(agent_id, AgentType.MARKET_ANALYSIS, comm_layer)
        self.capabilities = ['market_data_analysis', 'signal_generation']
        self.subscribed_channels = ['market_data']

    async def process_message(self, message: AgentMessage) -> bool:
        try:
            if message.message_type == MessageType.MARKET_DATA:
                signal = {
                    'signal': 'hold',
                    'confidence': 0.5,
                    'ts': time.time(),
                }
                out = AgentMessage(
                    message_id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    message_type=MessageType.SIGNAL_GENERATED,
                    priority=MessagePriority.NORMAL,
                    data=signal,
                    correlation_id=message.message_id,
                )
                return await self.comm_layer.publish_message('signals', out)
            return False
        except Exception as e:
            self.logger.error(f"process_message failed: {e}")
            return False


class MultiAgentSystemOptimized:
    def __init__(self, config: Optional[MultiAgentConfig] = None):
        self.config = config or MultiAgentConfig()
        self.comm_layer = RedisRealCommunicationLayer(self.config)
        self.agents: Dict[str, BaseAgentOptimized] = {}
        self._running = threading.Event()
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"{__name__}.MultiAgentSystemOptimized")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - SYSTEM - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    async def add_agent(self, agent: BaseAgentOptimized):
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent {agent.agent_id} already exists")
        self.agents[agent.agent_id] = agent
        if self._running.is_set():
            await agent.start()

    async def start_system(self):
        if self._running.is_set():
            return
        for agent in self.agents.values():
            try:
                await agent.start()
            except Exception as e:
                self.logger.error(f"Failed to start {agent.agent_id}: {e}")
        self._running.set()

    async def stop_system(self):
        if not self._running.is_set():
            return
        for agent in self.agents.values():
            try:
                await agent.stop()
            except Exception as e:
                self.logger.error(f"Failed to stop {agent.agent_id}: {e}")
        self._running.clear()

    def get_system_metrics(self) -> Dict:
        return {
            'system_running': self._running.is_set(),
            'total_agents': len(self.agents),
            'communication_stats': self.comm_layer.get_communication_stats(),
        }


def integrate_multi_agent_system_optimized(agent, config: Optional[Dict] = None) -> MultiAgentSystemOptimized:
    """Intègre la version optimisée sans casser l'existant."""
    ma_config = MultiAgentConfig(
        redis_host=(config or {}).get('redis_host', 'localhost'),
        redis_port=(config or {}).get('redis_port', 6379),
        heartbeat_interval=(config or {}).get('heartbeat_interval', 2.0),
    )
    system = MultiAgentSystemOptimized(ma_config)

    original_make_decision = agent.make_decision

    async def coordinated_make_decision(market_data):
        try:
            # Publication d'un message marché minimal si possible
            msg = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender_id='external_agent',
                message_type=MessageType.MARKET_DATA,
                priority=MessagePriority.NORMAL,
                data={
                    'close': float(market_data[-1]) if hasattr(market_data, '__len__') else 100.0,
                    'volume': 1000,
                    'timestamp': time.time()
                }
            )
            await system.comm_layer.publish_message('market_data', msg)
        except Exception:
            pass
        return original_make_decision(market_data)

    agent.make_decision = coordinated_make_decision
    agent.multi_agent_system_optimized = system

    async def _setup():
        await system.add_agent(MarketAnalysisAgentOptimized('market_001', system.comm_layer))
        await system.start_system()

    asyncio.get_event_loop().create_task(_setup())
    return system


