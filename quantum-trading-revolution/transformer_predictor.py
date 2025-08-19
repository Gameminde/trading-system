"""
🚀 TRANSFORMER PREDICTOR - PRÉDICTION DE PRIX AVANCÉE
✅ Modèle Transformer basique pour prédiction temporelle
🎯 Objectif: +20-30% précision des signaux trading
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("TRANSFORMER_PREDICTOR")

class PositionalEncoding(nn.Module):
    """Encodage positionnel pour le Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    """Modèle Transformer pour séries temporelles"""
    
    def __init__(self, input_dim: int, model_dim: int = 64, num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.output_projection = nn.Linear(model_dim, 1)
        
        logger.info(f"🤖 Transformer créé: {input_dim} → {model_dim} → 1")
        logger.info(f"   Têtes d'attention: {num_heads}")
        logger.info(f"   Couches: {num_layers}")
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        # Transformer expects (seq_len, batch_size, model_dim)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        
        # Prendre la dernière prédiction
        prediction = self.output_projection(x[:, -1, :])
        return prediction

class PriceDataset(Dataset):
    """Dataset pour les données de prix"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PricePredictor:
    """Prédicteur de prix utilisant un Transformer"""
    
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_names = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd']
        
        logger.info(f"🚀 Price Predictor initialisé")
        logger.info(f"   Séquence: {sequence_length} périodes")
        logger.info(f"   Device: {self.device}")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Préparer données pour entraînement"""
        logger.info(f"📊 Préparation données: {len(df)} périodes")
        
        # Vérifier que toutes les colonnes nécessaires sont présentes
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            logger.warning(f"⚠️ Colonnes manquantes: {missing_features}")
            # Créer des colonnes par défaut
            for feature in missing_features:
                if feature in ['rsi', 'macd']:
                    df[feature] = 50.0  # Valeur neutre
                elif feature == 'volume':
                    df[feature] = 1000000  # Volume par défaut
                else:
                    df[feature] = df['close']  # Copier le prix de clôture
        
        # Sélectionner les features
        data = df[self.feature_names].values
        
        # Normaliser les données
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data)
        
        # Créer séquences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 3])  # Prix de clôture (index 3)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✅ Données préparées: X={X.shape}, y={y.shape}")
        return X, y
    
    def train_model(self, df: pd.DataFrame, epochs: int = 100, batch_size: int = 32):
        """Entraîner le modèle Transformer"""
        logger.info(f"🤖 Début entraînement Transformer")
        logger.info(f"   Époques: {epochs}")
        logger.info(f"   Batch size: {batch_size}")
        
        try:
            # Préparer les données
            X, y = self.prepare_data(df)
            
            if len(X) == 0:
                logger.error("❌ Pas assez de données pour l'entraînement")
                return False
            
            # Créer le modèle
            input_dim = X.shape[2]
            self.model = TimeSeriesTransformer(input_dim).to(self.device)
            
            # Préparer DataLoader
            dataset = PriceDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Entraînement
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    predictions = self.model(batch_X).squeeze()
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if epoch % 20 == 0:
                    avg_loss = total_loss / len(dataloader)
                    logger.info(f"📈 Epoch {epoch}, Loss: {avg_loss:.6f}")
            
            logger.info("✅ Entraînement Transformer terminé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement: {e}")
            return False
    
    def predict_next_price(self, recent_data: pd.DataFrame) -> Optional[float]:
        """Prédire le prochain prix"""
        if self.model is None:
            logger.warning("⚠️ Modèle non entraîné")
            return None
        
        try:
            # Vérifier qu'il y a assez de données
            if len(recent_data) < self.sequence_length:
                logger.warning(f"⚠️ Données insuffisantes: {len(recent_data)} < {self.sequence_length}")
                return None
            
            # Préparer les données récentes
            data = recent_data[self.feature_names].tail(self.sequence_length).values
            scaled_data = self.scaler.transform(data)
            
            # Prédiction
            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)
                prediction = self.model(x).cpu().numpy()[0]
            
            # Dé-normaliser la prédiction
            dummy = np.zeros((1, len(self.feature_names)))
            dummy[0, 3] = prediction  # Position du prix de clôture
            denormalized = self.scaler.inverse_transform(dummy)
            
            predicted_price = denormalized[0, 3]
            logger.info(f"🔮 Prix prédit: ${predicted_price:.2f}")
            
            return predicted_price
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction: {e}")
            return None
    
    def evaluate_model(self, test_df: pd.DataFrame) -> Dict:
        """Évaluer la performance du modèle"""
        if self.model is None:
            return {'error': 'Modèle non entraîné'}
        
        try:
            logger.info("📊 Évaluation du modèle Transformer")
            
            # Préparer données de test
            X_test, y_test = self.prepare_data(test_df)
            
            if len(X_test) == 0:
                return {'error': 'Données de test insuffisantes'}
            
            # Prédictions
            self.model.eval()
            predictions = []
            
            with torch.no_grad():
                for i in range(len(X_test)):
                    x = torch.FloatTensor(X_test[i:i+1]).to(self.device)
                    pred = self.model(x).cpu().numpy()[0]
                    predictions.append(pred)
            
            # Dé-normaliser
            dummy = np.zeros((len(predictions), len(self.feature_names)))
            dummy[:, 3] = predictions
            denormalized_preds = self.scaler.inverse_transform(dummy)[:, 3]
            
            # Dé-normaliser les vraies valeurs
            dummy[:, 3] = y_test
            denormalized_actuals = self.scaler.inverse_transform(dummy)[:, 3]
            
            # Calculer métriques
            mse = np.mean((denormalized_preds - denormalized_actuals) ** 2)
            mae = np.mean(np.abs(denormalized_preds - denormalized_actuals))
            mape = np.mean(np.abs((denormalized_actuals - denormalized_preds) / denormalized_actuals)) * 100
            
            results = {
                'mse': mse,
                'mae': mae,
                'mape': mape,
                'predictions_count': len(predictions),
                'avg_predicted_price': np.mean(denormalized_preds),
                'avg_actual_price': np.mean(denormalized_actuals)
            }
            
            logger.info(f"📊 MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur évaluation: {e}")
            return {'error': str(e)}
    
    def save_model(self, filepath: str):
        """Sauvegarder le modèle entraîné"""
        if self.model is None:
            logger.warning("⚠️ Aucun modèle à sauvegarder")
            return False
        
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'sequence_length': self.sequence_length,
                'feature_names': self.feature_names
            }, filepath)
            logger.info(f"💾 Modèle sauvegardé: {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")
            return False
    
    def load_model(self, filepath: str):
        """Charger un modèle pré-entraîné"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Recréer le modèle
            input_dim = len(checkpoint['feature_names'])
            self.model = TimeSeriesTransformer(input_dim).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restaurer les autres paramètres
            self.scaler = checkpoint['scaler']
            self.sequence_length = checkpoint['sequence_length']
            self.feature_names = checkpoint['feature_names']
            
            logger.info(f"✅ Modèle chargé: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement: {e}")
            return False

def create_sample_data(n_periods: int = 1000) -> pd.DataFrame:
    """Créer des données d'exemple pour test"""
    np.random.seed(42)
    
    # Prix simulés avec tendance et volatilité réaliste
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, n_periods):
        # Tendance + bruit + volatilité
        trend = 0.0001 * i  # Tendance légèrement haussière
        noise = np.random.normal(0, 0.02)  # 2% volatilité
        volatility_cluster = 0.01 * np.sin(i / 50)  # Clusters de volatilité
        
        new_price = prices[-1] * (1 + trend + noise + volatility_cluster)
        prices.append(max(new_price, 1.0))  # Prix minimum $1
    
    # Créer DataFrame avec OHLCV
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(100000, 1000000, n_periods),
        'rsi': np.random.uniform(20, 80, n_periods),
        'macd': np.random.uniform(-2, 2, n_periods)
    })
    
    return df

def main():
    """Test du système Transformer"""
    print("🚀" + "="*80 + "🚀")
    print("   🔥 TRANSFORMER PREDICTOR - PRÉDICTION DE PRIX AVANCÉE")
    print("="*84)
    print("   ✅ Modèle Transformer PyTorch")
    print("   ✅ Encodage positionnel")
    print("   ✅ Multi-head attention")
    print("   🎯 Objectif: +20-30% précision trading")
    print("🚀" + "="*80 + "🚀")
    
    # Créer données d'exemple
    print("\n📊 Création données d'exemple...")
    sample_data = create_sample_data(800)
    print(f"   ✅ {len(sample_data)} périodes créées")
    
    # Diviser en train/test
    split_idx = int(len(sample_data) * 0.8)
    train_data = sample_data[:split_idx]
    test_data = sample_data[split_idx:]
    
    print(f"   📈 Entraînement: {len(train_data)} périodes")
    print(f"   🧪 Test: {len(test_data)} périodes")
    
    # Initialiser le prédicteur
    predictor = PricePredictor(sequence_length=60)
    
    # Entraîner le modèle
    print("\n🤖 Entraînement du modèle Transformer...")
    success = predictor.train_model(train_data, epochs=50, batch_size=16)
    
    if success:
        # Évaluer le modèle
        print("\n📊 Évaluation du modèle...")
        results = predictor.evaluate_model(test_data)
        
        if 'error' not in results:
            print(f"   📈 MSE: {results['mse']:.4f}")
            print(f"   📊 MAE: {results['mae']:.4f}")
            print(f"   📊 MAPE: {results['mape']:.2f}%")
            print(f"   🔮 Prédictions: {results['predictions_count']}")
        else:
            print(f"   ❌ Erreur: {results['error']}")
        
        # Test de prédiction
        print("\n🔮 Test de prédiction...")
        recent_data = test_data.tail(100)  # 100 dernières périodes
        predicted_price = predictor.predict_next_price(recent_data)
        
        if predicted_price:
            current_price = recent_data['close'].iloc[-1]
            price_change = (predicted_price - current_price) / current_price * 100
            
            print(f"   💰 Prix actuel: ${current_price:.2f}")
            print(f"   🔮 Prix prédit: ${predicted_price:.2f}")
            print(f"   📈 Changement prédit: {price_change:+.2f}%")
            
            # Recommandation
            if price_change > 1:
                recommendation = "BUY"
            elif price_change < -1:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            print(f"   🎯 Recommandation: {recommendation}")
        
        # Sauvegarder le modèle
        print("\n💾 Sauvegarde du modèle...")
        predictor.save_model("transformer_model.pth")
        
    else:
        print("   ❌ Entraînement échoué")
    
    print("\n✅ Test Transformer terminé!")

if __name__ == "__main__":
    main()
