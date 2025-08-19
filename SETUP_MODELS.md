# 🤖 **SETUP MODELS LOCAUX**

## ⚠️ **FICHIERS EXCLUS DE GITHUB**

Les fichiers suivants sont **exclus de GitHub** (limite 100MB) mais **nécessaires** pour faire fonctionner l'agent :

```
models/trained/
├── trained_agent.pkl (500MB) ❌ Exclu GitHub
└── [autres modèles]
```

## 🔧 **RÉGÉNÉRATION AUTOMATIQUE**

### **Option 1 : Entraînement automatique**
```bash
# L'agent détecte automatiquement les modèles manquants
python AGENT_TRAINED_OPTIMIZED.py

# ✅ Si trained_agent.pkl manque → Entraînement automatique
# ⚡ Durée : 5-15 minutes selon les données
```

### **Option 2 : Entraînement forcé**
```bash
# Forcer le réentraînement même si le modèle existe
python historical_data_training.py --force-retrain
```

## 📊 **DONNÉES REQUISES**

L'entraînement utilise automatiquement :

### **📈 Actions (Yahoo Finance)**
- **AAPL** (Apple) - 5 ans historique
- **MSFT** (Microsoft) - 5 ans historique  
- **TSLA** (Tesla) - 5 ans historique
- **NVDA** (NVIDIA) - 5 ans historique

### **₿ Crypto (CoinGecko/Binance)**
- **BTC-USD** (Bitcoin) - 5 ans historique
- **ETH-USD** (Ethereum) - 5 ans historique
- **BNB-USD** (Binance Coin) - 5 ans historique

## 🚀 **TEMPS DE SETUP**

| Étape | Durée | Description |
|-------|--------|-------------|
| Installation dépendances | 2-5 min | `pip install -r requirements.txt` |
| Téléchargement données | 3-7 min | APIs Yahoo/CoinGecko automatique |
| Entraînement modèles | 5-15 min | LLM + patterns extraction |
| **TOTAL** | **10-27 min** | **Setup complet depuis zéro** |

## ⚡ **OPTIMISATIONS**

### **Cache intelligent :**
- Données historiques mises en cache (évite re-téléchargement)
- Modèles sauvegardés automatiquement après entraînement
- Patterns extraits réutilisables

### **Entraînement incrémental :**
- Détection automatique nouveaux données
- Mise à jour modèle sans repartir de zéro
- Conservation historique performance

## 🔍 **DEBUGGING**

### **Problème modèle manquant :**
```bash
# Vérifier existence
ls models/trained/

# Si vide → Entraînement automatique
python AGENT_TRAINED_OPTIMIZED.py
```

### **Problème données :**
```bash
# Test connexions APIs
python -c "import yfinance as yf; print('Yahoo OK' if yf.download('AAPL', period='1d') is not None else 'Yahoo ERROR')"
```

## 💡 **RECOMMANDATIONS**

1. **Première utilisation** → Lancer `AGENT_TRAINED_OPTIMIZED.py` directement
2. **Développement** → Garder modèles localement (performance)  
3. **Production** → Considérer cloud storage pour modèles (AWS S3, etc.)

---

🎯 **L'agent s'auto-configure automatiquement. Pas d'action manuelle requise !**
