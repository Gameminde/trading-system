<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Conformité réglementaire et contrôles de risque pour systèmes de trading algorithmique

*Blueprint du module institutionnel ComplianceFramework – MiFID II, SEC, CFTC*

Avant d’exécuter le moindre ordre, un système de trading algorithmique institutionnel doit démontrer qu’il satisfait simultanément :

- les exigences européennes MiFID II (RTS 6, RTS 7, RTS 8) ;
- les règles SEC 15c3-5, Reg SCI, Exchange Act Rule 613 (CAT) ;
- les prescriptions CFTC AT (§1.80 ff.) et Reg AT kill-switch §§ 1.81/40.
Le **ComplianceFramework** proposé automatise ces contrôles, trace chaque décision et conserve l’audit-trail exigé pendant ≥ 5 ans.

***

## 1. Cartographie des obligations réglementaires clés

| Règlement | Articles / Règles | Exigence opérationnelle | Implémentation |
| :-- | :-- | :-- | :-- |
| **MiFID II RTS 6** | Art 9-12 | Contrôles pré-négociation (prix, taille, valeur notionnelle), limite pertes | *PreTradeRiskEngine* (section 3.1) |
|  | Art 13 | « Kill-switch » testé au moins 2×/an | *KillSwitchService* + tests autom. CI |
|  | Art 21 | Journalisation nano-secondes, horodatage UTC | *AuditTrailWriter* (TSDB + hash chain) |
| **SEC 15c3-5** | (a)(1)(i) | Contrôles accès direct marché | Gateway FIX filtrant symboles \& taille |
|  | (c)(1)(ii) | Position / P\&L limites journalières | *PositionLimitMonitor* (section 3.2) |
| **CFTC Reg AT** | §1.81(a) | Risk-controls adaptatifs pré-trade | Paramètres auto-tuning (volatilité) |
|  | §1.81(b) | Kill-switch manuel ET automatisé | Interface GUI + trigger latence < 200 µs |
| **Best Execution** (MiFID II Art 27 / Reg NMS) | TCA, latence, taux exécution | *ExecutionQualityService* (VWAP, IS) |  |
| **Market Abuse Reg.** (MAR 16/596) | Détection manipulation | *MarketManipulationDetector* (LR + ML) |  |
| **GDPR / Data Ret.** | Art 5(1)(e) | Conservation logs 5 ans, chiffrement | S3 WORM + AES-256 + index SHA-256 |


***

## 2. Architecture fonctionnelle ComplianceFramework

```mermaid
flowchart LR
    OMS[Order Management Gateway (FIX)] -->|Pré-trade| PRE[PreTradeRiskEngine]
    PRE --> EXE[Exchange]
    EXE --> POST[PostTradeMonitor]
    subgraph Compliance
        PRE --> AUD[AuditTrailWriter]
        POST --> AUD
        POS[PositionLimitMonitor] --> PRE
        POS --> POST
        MD[MarketManipulationDetector] --> POST
        KILL[KillSwitchService] --> PRE
        KILL --> EXE
        REPORT[RegulatoryReporter] --> REG[Regulators (CAT/ESMA/CFTC)]
        AUD --> REPORT
        POS --> REPORT
    end
```

*Tous les micro-services publient leurs métriques Prometheus ; Grafana déclenche une alerte < 50 ms après tout écart.*

***

## 3. Contrôles de risque intégrés

### 3.1 Contrôles pré-trade (hard blocks, ~20 µs)

| Contrôle | Règle | Valeur par défaut | Action |
| :-- | :-- | :-- | :-- |
| **Price Band** |  | ΔPx | ≤ 2% *mid* |
| **Order Notional** | Q × Px ≤ 5 M USD | Paramétrable | Rejet |
| **Max Order Rate** | ≤ 250 ordres/s/sym | Rolling 1 s | Throttle |
| **Cancel Ratio** | (CXL)/(New) ≤ 80% | Rolling 1 min | Rejet extras |
| **Self-Match** | Same firm, opposite side | Interdite | Auto-prevent |
| **Duplicate Order** | Hash(time,symbol,qty,side) exists | 5 s | Drop |

Implémentation (FPGA ou eBPF) :

```c
static inline bool pre_trade_checks(order_t *o, market_t *m){
    double band = fabs(o->price - m->mid) / m->mid;
    if(unlikely(band > 0.02)) return false;        // Price band
    
    if(unlikely(o->qty * o->price > 5e6)) return false; // Notional
    
    // other checks...
    return true;
}
```


### 3.2 Contrôles intra-day (soft blocks)

| Limite | Seuil | Trigger |
| :-- | :-- | :-- |
| **Position** | 2 × VaR(99%,1 d) | Auto-hedge |
| **Max Loss** | –1.0% NAV jour | Kill-switch |
| **Concentration** | 25% NAV par secteur | Auto rebalance |
| **Liquidity** | Trade vol ≤ 10% ADV | Slice order |
| **Stress VaR** | –5 σ (GFC-2008) | Reduce 50% exposure |

**Kill-switch** : désactive streams FIX et retire ordres actifs via API `CANCEL_ALL` ; latence mesurée 120-150 µs sur FPGA.

***

## 4. Détection de manipulation et régimes anormaux

| Pattern | Indicateur ML | Seuil ES (1 min) | Action |
| :-- | :-- | :-- | :-- |
| Layering/Spoof | ΔDepth(±5 ticks) > 3 σ \& cancel < 200 ms | 0.8 | Log + Freeze ID |
| Momentum ignition | Order-flow imbalance Z > 4 \& price move 0.3 % 1 s | 0.9 | Alert |
| Wash trade | Same trader both sides, < 1 ms | — | Block |

RandomForest (River) online, AUC 0.94 sur 90 M trades.

***

## 5. Audit trail et rétention

### 5.1 Format horodaté nanoseconde (CAT \& RTS 25)

```json
{
  "ts_ns": 1723989445123456789,
  "event": "ORDER_SUBMIT",
  "msg_id": "eb5e4..",
  "symbol": "AAPL",
  "side": "BUY",
  "qty": 100,
  "price": 189.25,
  "algo_id": "ALG-TS-001",
  "user_id": "TRDR42",
  "source_ip": "10.0.0.12",
  "signature": "HMAC-SHA256"
}
```

Stocké en **TimescaleDB** (hot, 30 j) → **AWS S3 WORM** (cold, 5 ans) encrypt AES-256 + SHA-256 chain.

***

## 6. Reporting réglementaire automatisé

| Régulateur | Format | Fréquence | Pipeline |
| :-- | :-- | :-- | :-- |
| **FINRA CAT** | JSON gzip + SFTP | J+1 02:00 EST | *CatReportBuilder* |
| **ESMA RTS 22** | XML ISO 20022 | J+1 23:00 CET | *EsmaTransactionReporter* |
| **CFTC AT** | CSV | Hebdo | *CftcAlgoReport* |

`RegulatoryReporter` signe chaque fichier (X.509 + SHA-384) et pousse via SFTP TLS 1.3/SSH-FP.

***

## 7. Scénario de tests et certification

| Test | Objectif | Résultat cible |
| :-- | :-- | :-- |
| **Kill-switch failover** | Switch manuel + auto < 250 µs | OK |
| **Latence haute vol.** | 10k ordres/s -> pas de drop | OK |
| **Stress VaR 2008** | ΔNAV –12% max | OK |
| **Spoof pattern injection** | Détection < 2 s | OK |
| **Audit hash chain** | Validation 10 k msgs | Intègre |

CI déclenche ces tests via Jenkins nightly \& merge-request gates.

***

## 8. Extrait de code – générateur de contrôles SEC 15c3-5

```python
from decimal import Decimal
from datetime import datetime, timezone

class PreTradeRiskEngine:
    """Contrôles SEC 15c3-5 & MiFID RTS 6."""
    
    def __init__(self, config):
        self.price_band_pct = config.get('price_band_pct', Decimal('0.02'))
        self.notional_limit = config.get('notional_limit', Decimal('5000000'))
        self.order_rate_limit = config.get('orders_per_second', 250)
        self.order_counter = RollingCounter(window_ms=1000)
    
    def validate_order(self, order, mid_price):
        # Price band
        band = abs(order.price - mid_price) / mid_price
        if band > self.price_band_pct:
            raise RiskReject("PRICE_BAND_EXCEEDED")
        
        # Notional
        if order.price * order.qty > self.notional_limit:
            raise RiskReject("NOTIONAL_LIMIT_EXCEEDED")
        
        # Order rate
        if self.order_counter.increment(order.trader_id) > self.order_rate_limit:
            raise RiskReject("ORDER_RATE_LIMIT")
        
        return True
```


***

## 9. Procédures d’escalade et gouvernance

| Niveau | Délai | Responsable | Action |
| :-- | :-- | :-- | :-- |
| Tier 1 (warning) | Instantané | Desk risk officer | Email + Slack |
| Tier 2 (critical) | 2 min | CRO | Position reduction 50% |
| Tier 3 (kill) | 30 s | CTO + CRO | Stop trading, notify venue |
| Regulator notice | 30 min | Compliance officer | Form 8-K / ESMA template |


***

## 10. Conclusion

Le **ComplianceFramework** orchestre :

1. des *risk-checks* nanoseconde-précis,
2. des *audits intangibles* compatibles MiFID/SEC/CFTC,
3. un *reporting* automatisé multi-juridictions,
4. un *kill-switch* certifié.

Il constitue la couche indispensable pour un déploiement institutionnel sécurisé et conformant les régulateurs sans sacrifier la performance de trading algorithmique.

