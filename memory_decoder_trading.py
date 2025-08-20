"""
🧠 MEMORY DECODER FOR ALGORITHMIC TRADING
Implementation basée sur le papier "Transformer Decoder as Memory"
Adapté spécifiquement pour le trading algorithmique
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging
from datetime import datetime
import pickle
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MEMORY_DECODER")

class FinancialPositionalEncoding(nn.Module):
    """Encodage positionnel adapté aux patterns cycliques financiers"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Encodage positionnel classique
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Encodages cycliques pour patterns financiers
        self.hourly_encoding = nn.Embedding(24, d_model // 4)  # Heures du jour
        self.daily_encoding = nn.Embedding(7, d_model // 4)    # Jours de la semaine
        self.monthly_encoding = nn.Embedding(31, d_model // 4)  # Jours du mois
        self.regime_encoding = nn.Embedding(5, d_model // 4)   # Régimes de marché
        
    def forward(self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None, 
                market_regime: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model]
        timestamps: [batch_size, seq_len] contenant les timestamps
        market_regime: [batch_size] contenant le régime actuel
        """
        batch_size, seq_len, _ = x.shape
        
        # Encodage positionnel de base
        x = x + self.pe[:, :seq_len, :]
        
        # Si timestamps fournis, ajouter encodages cycliques
        if timestamps is not None:
            # Extraire composants temporels (simplifiés pour l'exemple)
            hours = torch.zeros(batch_size, seq_len, dtype=torch.long)
            days = torch.zeros(batch_size, seq_len, dtype=torch.long)
            
            # Ajouter encodages cycliques
            x = x + self.hourly_encoding(hours)[:, :, :self.d_model//4].repeat(1, 1, 4)
            x = x + self.daily_encoding(days)[:, :, :self.d_model//4].repeat(1, 1, 4)
        
        # Si régime de marché fourni
        if market_regime is not None:
            regime_enc = self.regime_encoding(market_regime).unsqueeze(1)
            x = x + regime_enc.repeat(1, seq_len, 1)[:, :, :self.d_model]
        
        return x

class MultiTimeframeAttention(nn.Module):
    """Attention simultanée sur plusieurs timeframes"""
    
    def __init__(self, d_model: int, n_heads: int = 8, 
                 timeframes: List[int] = [1, 5, 15, 60]):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.timeframes = timeframes
        
        # Une tête d'attention pour chaque timeframe
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in timeframes
        ])
        
        # Fusion des timeframes
        self.fusion_layer = nn.Linear(d_model * len(timeframes), d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process multiple timeframes and fuse results"""
        attended_outputs = []
        
        for i, (tf, attn_layer) in enumerate(zip(self.timeframes, self.attention_layers)):
            # Simuler différents timeframes (dans la pratique, utiliser vraies données)
            # Pour l'instant, on utilise les mêmes données
            attended, _ = attn_layer(x, x, x, attn_mask=mask)
            attended_outputs.append(attended)
        
        # Concaténer et fusionner
        concatenated = torch.cat(attended_outputs, dim=-1)
        fused = self.fusion_layer(concatenated)
        
        # Connexion résiduelle et normalisation
        output = self.layer_norm(x + fused)
        
        return output

class TradingMemoryDecoder(nn.Module):
    """
    Memory Decoder optimisé pour trading algorithmique
    Basé sur le papier avec adaptations spécifiques au domaine financier
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # Configuration par défaut optimisée pour le trading
        default_config = {
            "d_model": 256,           # Dimension du modèle
            "n_heads": 8,             # Têtes d'attention
            "n_layers": 6,            # Couches du decoder
            "d_ff": 1024,             # Dimension feed-forward
            "vocab_size": 10000,      # Taille vocabulaire financier
            "max_seq_length": 512,    # Longueur séquence max
            "dropout": 0.1,           # Dropout rate
            "k_neighbors": 32,        # Voisins k-NN
            "memory_size": 100000,    # Taille du datastore
            "use_multi_timeframe": True,
            "use_regime_detection": True
        }
        
        self.config = config or default_config
        
        # Composants principaux
        self.embedding = nn.Embedding(self.config["vocab_size"], self.config["d_model"])
        self.pos_encoding = FinancialPositionalEncoding(
            self.config["d_model"], 
            self.config["max_seq_length"]
        )
        
        # Couches du decoder
        self.decoder_layers = nn.ModuleList([
            self._create_decoder_layer() for _ in range(self.config["n_layers"])
        ])
        
        # Multi-timeframe attention si activée
        if self.config["use_multi_timeframe"]:
            self.multi_timeframe_attn = MultiTimeframeAttention(self.config["d_model"])
        
        # Tête de prédiction pour trading
        self.trading_head = nn.Sequential(
            nn.Linear(self.config["d_model"], self.config["d_model"] // 2),
            nn.ReLU(),
            nn.Dropout(self.config["dropout"]),
            nn.Linear(self.config["d_model"] // 2, 5)  # 5 actions: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
        )
        
        # Datastore k-NN pour mémoire
        self.memory_datastore = {
            'keys': [],
            'values': [],
            'metadata': []
        }
        
        # Détecteur de régime de marché
        if self.config["use_regime_detection"]:
            self.regime_detector = MarketRegimeDetector()
        
        # Métriques de performance
        self.performance_tracker = TradingPerformanceTracker()
        
        logger.info(f"🧠 Memory Decoder initialisé avec config: {self.config}")
    
    def _create_decoder_layer(self) -> nn.TransformerDecoderLayer:
        """Créer une couche de decoder"""
        return nn.TransformerDecoderLayer(
            d_model=self.config["d_model"],
            nhead=self.config["n_heads"],
            dim_feedforward=self.config["d_ff"],
            dropout=self.config["dropout"],
            batch_first=True
        )
    
    def encode_market_context(self, market_data: Dict) -> torch.Tensor:
        """
        Encoder le contexte de marché en embeddings
        
        market_data: Dict contenant prix, indicateurs, volumes, etc.
        """
        # Créer une représentation numérique du contexte
        features = []
        
        # Prix normalisés
        if 'price' in market_data:
            price_norm = (market_data['price'] - 100) / 100  # Normalisation simple
            features.append(price_norm)
        
        # Indicateurs techniques
        if 'rsi' in market_data:
            rsi_norm = (market_data['rsi'] - 50) / 50
            features.append(rsi_norm)
        
        if 'macd' in market_data:
            macd_norm = market_data['macd'] / 10  # Normalisation MACD
            features.append(macd_norm)
        
        # Volume normalisé
        if 'volume' in market_data:
            volume_norm = np.log(market_data['volume'] + 1) / 20
            features.append(volume_norm)
        
        # Convertir en tensor et projeter dans l'espace d_model
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Projection linéaire vers d_model dimensions
        if not hasattr(self, 'feature_projection'):
            self.feature_projection = nn.Linear(len(features), self.config["d_model"])
        
        encoded = self.feature_projection(features_tensor)
        
        return encoded
    
    def forward(self, x: torch.Tensor, 
                memory_context: Optional[torch.Tensor] = None,
                use_knn_memory: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass avec mémoire k-NN optionnelle
        
        x: [batch_size, seq_len] ou [batch_size, seq_len, d_model]
        memory_context: Contexte additionnel pour la mémoire
        use_knn_memory: Utiliser la mémoire k-NN
        """
        # Embedding si nécessaire
        if x.dim() == 2:
            x = self.embedding(x)
        
        # Encodage positionnel
        x = self.pos_encoding(x)
        
        # Passer par les couches du decoder
        for layer in self.decoder_layers:
            x = layer(x, x)  # Self-attention
        
        # Multi-timeframe attention si activée
        if self.config["use_multi_timeframe"] and hasattr(self, 'multi_timeframe_attn'):
            x = self.multi_timeframe_attn(x)
        
        # Récupération k-NN si activée
        if use_knn_memory and len(self.memory_datastore['keys']) > 0:
            knn_output = self.retrieve_from_memory(x)
            # Combiner avec output du decoder
            x = 0.6 * x + 0.4 * knn_output
        
        # Prédiction finale pour trading
        trading_logits = self.trading_head(x[:, -1, :])  # Utiliser dernier token
        
        return {
            'logits': trading_logits,
            'hidden_states': x,
            'action_probs': F.softmax(trading_logits, dim=-1)
        }
    
    def retrieve_from_memory(self, query: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """
        Récupérer les k plus proches voisins du datastore
        
        query: [batch_size, seq_len, d_model]
        k: Nombre de voisins (par défaut config["k_neighbors"])
        """
        if len(self.memory_datastore['keys']) == 0:
            return query  # Pas de mémoire, retourner query
        
        k = k or self.config["k_neighbors"]
        batch_size, seq_len, d_model = query.shape
        
        # Convertir keys en tensor
        memory_keys = torch.stack(self.memory_datastore['keys'])
        
        # Calculer distances (utiliser dernier token de query)
        query_vector = query[:, -1, :].unsqueeze(1)  # [batch_size, 1, d_model]
        distances = torch.cdist(query_vector, memory_keys.unsqueeze(0))
        
        # Trouver k plus proches voisins
        k_actual = min(k, len(self.memory_datastore['keys']))
        _, indices = torch.topk(distances, k_actual, largest=False, dim=-1)
        
        # Récupérer valeurs correspondantes
        retrieved_values = []
        for batch_idx in range(batch_size):
            batch_values = []
            for idx in indices[batch_idx, 0]:
                batch_values.append(self.memory_datastore['values'][idx])
            
            # Moyenner les valeurs récupérées
            if batch_values:
                mean_value = torch.stack(batch_values).mean(dim=0)
                retrieved_values.append(mean_value)
            else:
                retrieved_values.append(torch.zeros(d_model))
        
        retrieved = torch.stack(retrieved_values).unsqueeze(1)
        retrieved = retrieved.repeat(1, seq_len, 1)
        
        return retrieved
    
    def update_memory(self, context: torch.Tensor, 
                     action: int, 
                     reward: float,
                     metadata: Optional[Dict] = None):
        """
        Mettre à jour le datastore avec nouvelle expérience
        
        context: Représentation du contexte [d_model]
        action: Action prise
        reward: Récompense obtenue
        metadata: Métadonnées additionnelles
        """
        # Limiter taille du datastore
        if len(self.memory_datastore['keys']) >= self.config["memory_size"]:
            # Supprimer les plus anciennes entrées (FIFO)
            self.memory_datastore['keys'].pop(0)
            self.memory_datastore['values'].pop(0)
            self.memory_datastore['metadata'].pop(0)
        
        # Ajouter nouvelle entrée
        self.memory_datastore['keys'].append(context.detach())
        
        # Créer value enrichie avec action et reward
        value = context.clone()
        value[0] = action  # Encoder action dans première dimension
        value[1] = reward  # Encoder reward dans deuxième dimension
        
        self.memory_datastore['values'].append(value.detach())
        
        # Ajouter métadonnées
        meta = metadata or {}
        meta['timestamp'] = datetime.now().isoformat()
        meta['action'] = action
        meta['reward'] = reward
        self.memory_datastore['metadata'].append(meta)
        
        logger.debug(f"📝 Mémoire mise à jour: {len(self.memory_datastore['keys'])} entrées")
    
    def save_memory(self, filepath: str):
        """Sauvegarder le datastore sur disque"""
        try:
            torch.save({
                'datastore': self.memory_datastore,
                'config': self.config,
                'model_state': self.state_dict()
            }, filepath)
            logger.info(f"💾 Mémoire sauvegardée: {filepath}")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde mémoire: {e}")
    
    def load_memory(self, filepath: str):
        """Charger le datastore depuis le disque"""
        try:
            checkpoint = torch.load(filepath)
            self.memory_datastore = checkpoint['datastore']
            self.load_state_dict(checkpoint['model_state'])
            logger.info(f"📂 Mémoire chargée: {len(self.memory_datastore['keys'])} entrées")
        except Exception as e:
            logger.error(f"❌ Erreur chargement mémoire: {e}")

class MarketRegimeDetector:
    """Détection automatique des régimes de marché"""
    
    def __init__(self):
        self.regimes = ['BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOL', 'CRISIS']
        self.current_regime = 'SIDEWAYS'
        self.regime_history = deque(maxlen=100)
        
    def detect_regime(self, market_data: Dict) -> str:
        """
        Détecter le régime de marché actuel
        
        market_data: Données de marché incluant prix, volume, volatilité
        """
        # Logique simplifiée de détection
        if 'volatility' in market_data:
            if market_data['volatility'] > 0.03:
                regime = 'HIGH_VOL'
            elif market_data.get('trend', 0) > 0.01:
                regime = 'BULL'
            elif market_data.get('trend', 0) < -0.01:
                regime = 'BEAR'
            else:
                regime = 'SIDEWAYS'
        else:
            regime = 'SIDEWAYS'
        
        self.current_regime = regime
        self.regime_history.append({
            'regime': regime,
            'timestamp': datetime.now(),
            'data': market_data
        })
        
        return regime

class TradingPerformanceTracker:
    """Tracker de performance pour le trading"""
    
    def __init__(self):
        self.trades = []
        self.performance_metrics = {}
        
    def record_trade(self, trade_info: Dict):
        """Enregistrer un trade"""
        self.trades.append({
            **trade_info,
            'timestamp': datetime.now()
        })
        
    def calculate_metrics(self) -> Dict:
        """Calculer les métriques de performance"""
        if not self.trades:
            return {}
        
        # Calculer métriques basiques
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.get('profit', 0) > 0)
        
        metrics = {
            'total_trades': total_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'avg_profit': np.mean([t.get('profit', 0) for t in self.trades]),
            'total_profit': sum(t.get('profit', 0) for t in self.trades)
        }
        
        self.performance_metrics = metrics
        return metrics

class TradingMemoryLoss(nn.Module):
    """Loss fonction adaptée au trading avec Memory Decoder"""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha  # Poids pour KL divergence
        self.beta = beta   # Poids pour trading performance
        
    def forward(self, pred_logits: torch.Tensor,
                knn_logits: Optional[torch.Tensor],
                true_action: torch.Tensor,
                future_returns: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculer la loss hybride
        
        pred_logits: Logits prédits par le modèle
        knn_logits: Logits des k-NN (si disponibles)
        true_action: Action vraie/optimale
        future_returns: Retours futurs pour loss de performance
        """
        # Cross-entropy loss standard
        ce_loss = F.cross_entropy(pred_logits, true_action)
        
        # KL divergence avec k-NN si disponible
        if knn_logits is not None:
            pred_probs = F.softmax(pred_logits, dim=-1)
            knn_probs = F.softmax(knn_logits, dim=-1)
            kl_loss = F.kl_div(pred_probs.log(), knn_probs, reduction='batchmean')
        else:
            kl_loss = torch.tensor(0.0)
        
        # Trading performance loss si retours futurs disponibles
        if future_returns is not None:
            # Simuler P&L basé sur prédictions
            predicted_action = torch.argmax(pred_logits, dim=-1)
            
            # Mapper actions à positions: 0=STRONG_SELL(-2), 1=SELL(-1), 2=HOLD(0), 3=BUY(1), 4=STRONG_BUY(2)
            positions = predicted_action.float() - 2.0
            
            # P&L = position * future_return
            pnl = positions * future_returns
            
            # Loss = -P&L (on veut maximiser P&L)
            trading_loss = -pnl.mean()
        else:
            trading_loss = torch.tensor(0.0)
        
        # Combiner les losses
        total_loss = ce_loss + self.alpha * kl_loss + self.beta * trading_loss
        
        return total_loss

def test_memory_decoder():
    """Test du Memory Decoder"""
    print("🧪 Test du Memory Decoder pour Trading")
    print("="*50)
    
    # Créer instance
    decoder = TradingMemoryDecoder()
    
    # Créer données de test
    batch_size = 2
    seq_len = 10
    
    # Input tokens (simulés)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    output = decoder(input_ids)
    
    print(f"✅ Shape logits: {output['logits'].shape}")
    print(f"✅ Action probs: {output['action_probs']}")
    
    # Test mise à jour mémoire
    context = torch.randn(decoder.config["d_model"])
    decoder.update_memory(context, action=2, reward=0.01)
    
    print(f"✅ Mémoire mise à jour: {len(decoder.memory_datastore['keys'])} entrées")
    
    # Test avec mémoire k-NN
    output_with_memory = decoder(input_ids, use_knn_memory=True)
    print(f"✅ Forward avec mémoire k-NN réussi")
    
    print("\n🎯 Memory Decoder opérationnel!")

if __name__ == "__main__":
    test_memory_decoder()
