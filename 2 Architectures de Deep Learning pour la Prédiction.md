<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Architectures de Deep Learning pour la Prédiction de Marchés Financiers : Guide Complet et État de l'Art

L'application du deep learning aux marchés financiers représente aujourd'hui l'une des frontières les plus prometteuses et techniquement exigeantes de l'intelligence artificielle appliquée. Cette synthèse examine de manière exhaustive les architectures neurales les plus efficaces pour la prédiction financière, couvrant les réseaux de neurones pour séries temporelles, le reinforcement learning appliqué au trading, les frameworks optimaux, ainsi que les méthodes de validation robustes et les limitations critiques à considérer.

![Comparaison des performances des architectures de deep learning pour la prédiction financière](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/d6955ee87b3126f91a0603b86eb2303e/a12f09dd-84ca-4101-8ea0-75dcba15e4d8/39a9432e.png)

Comparaison des performances des architectures de deep learning pour la prédiction financière

## Architectures de Réseaux de Neurones pour Séries Temporelles Financières

### Long Short-Term Memory (LSTM) : La Référence Établie

Les **réseaux LSTM** demeurent l'architecture de référence pour les séries temporelles financières, démontrant une **efficacité remarquable dans la capture des dépendances temporelles à long terme**. Une étude comparative récente sur les données Tesla (2015-2024) révèle que les LSTM atteignent une **précision de 94% dans la prédiction directionnelle**, surpassant significativement les méthodes traditionnelles comme ARIMA.[^1][^2]

Les **mécanismes de portes des LSTM** (forget, input, output) s'avèrent particulièrement adaptés aux particularités des données financières. Leur capacité à **maintenir et filtrer l'information sur de longues séquences** permet de capturer les patterns cycliques des marchés, les tendances macro-économiques, et les effets de mémoire caractéristiques des séries de prix. L'architecture LSTM excelle notamment dans la **modélisation des relations non-linéaires** présentes dans les données de prix, volume, et indicateurs techniques.[^2][^3]

### Gated Recurrent Units (GRU) : Efficacité et Simplicité

Les **GRU émergent comme l'alternative optimale** aux LSTM dans de nombreux contextes financiers, offrant un **rapport performance/complexité supérieur**. Les études benchmarks montrent que les GRU atteignent souvent des **performances égales ou supérieures aux LSTM** tout en nécessitant **20% moins de temps d'entraînement**. Cette efficacité s'explique par leur architecture simplifiée avec seulement deux portes (reset et update) contre trois pour les LSTM.[^4][^5]

Les **GRU démontrent une précision particulièrement élevée sur les actions technologiques**, avec des **Mean Absolute Error (MAE) inférieurs aux LSTM sur Apple et Microsoft**. Cette performance s'avère cruciale pour les applications de trading haute fréquence où la vitesse d'entraînement et d'inférence constitue un avantage concurrentiel décisif.[^6][^4]

### Transformers : Potentiel et Défis Spécifiques aux Marchés

L'adaptation des **Transformers aux séries temporelles financières** présente des résultats contrastés qui nécessitent une analyse nuancée. Bien que les Transformers excellent dans la **capture des patterns globaux** grâce à leur mécanisme d'attention, ils montrent des **performances inférieures sur la volatilité extrême** des marchés comme celui de Tesla.[^1][^7][^4][^5]

Les **Temporal Fusion Transformers (TFT)** représentent une adaptation spécialisée qui intègre des **mécanismes d'attention multi-horizon**. Ces modèles s'avèrent particulièrement efficaces pour la **prédiction à court terme** (1-5 jours) mais peinent sur les horizons plus longs où les LSTM conservent leur avantage. Le **traitement parallèle des Transformers** offre néanmoins des gains computationnels significatifs lors de l'entraînement sur de très grandes séries temporelles.[^7]

### Architectures Hybrides et xLSTM-TS : L'État de l'Art

Les **modèles hybrides LSTM-GRU** représentent actuellement l'état de l'art, combinant les **avantages des deux approches**. Ces architectures intègrent des couches Conv1D pour la capture de patterns locaux et des couches bidirectionnelles pour l'analyse contextuelle. Les résultats montrent des **améliorations substantielles** par rapport aux méthodes traditionnelles avec une **validation croisée 5-fold** rigoureuse.[^8]

L'architecture **xLSTM-TS**, récemment développée, constitue une **évolution majeure des LSTM classiques** spécifiquement optimisée pour les séries temporelles. Cette variante atteint des **précisions de 72.82% et F1-scores de 73.16%** sur des datasets complexes comme l'ETF brésilien EWZ. L'intégration de **techniques de débruitage par ondelettes** améliore significativement les performances en préprocessant les signaux pour réduire le bruit de marché.[^7]

## Reinforcement Learning Appliqué au Trading Algorithmique

![Comparaison radar des algorithmes de Reinforcement Learning pour le trading financier](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/d6955ee87b3126f91a0603b86eb2303e/1f741d93-c1d5-41ae-ba87-7e65d3ea9a8c/469b2aaa.png)

Comparaison radar des algorithmes de Reinforcement Learning pour le trading financier

### Deep Q-Networks (DQN) : Apprentissage de Valeur Actionnable

Les **DQN révolutionnent l'approche du trading algorithmique** en permettant l'apprentissage de politiques de trading dans des environnements haute dimension. Leur capacité à **approximer la fonction Q-valeur** via des réseaux de neurones profonds permet de gérer des espaces d'états complexes intégrant prix, volumes, indicateurs techniques, et sentiment de marché.[^9][^10][^11]

Les **DQN démontrent une efficacité particulière dans la reconnaissance de patterns de trading rentables**, notamment dans les **environnements de trading crypto et forex**. Leur mécanisme de **replay d'expérience** permet un apprentissage stable en évitant la corrélation temporelle des données financières séquentielles. Cependant, les DQN peuvent souffrir d'**instabilité lors des changements de régime de marché**, nécessitant un réglage minutieux des hyperparamètres.[^10][^11][^12][^9]

### Proximal Policy Optimization (PPO) : Stabilité et Robustesse

**PPO s'impose comme l'algorithme de choix** pour de nombreuses applications de trading grâce à sa **stabilité d'entraînement exceptionnelle**. Son mécanisme de **clipping des mises à jour de politique** évite les changements destructeurs qui peuvent ruiner une stratégie de trading. Les études comparatives montrent que **PPO maintient des performances consistantes** même dans des conditions de marché volatiles.[^10][^13][^12]

L'algorithme **PPO excelle particulièrement dans la gestion de portefeuilles multi-actifs**, où sa capacité à **optimiser des actions continues** (tailles de positions) s'avère cruciale. Les implémentations récentes intégrant des **réseaux d'acteur-critique** permettent d'optimiser simultanément les décisions d'achat/vente et les montants investis.[^9][^14][^10]

### Asynchronous Advantage Actor-Critic (A3C) : Apprentissage Parallèle

**A3C introduit le paradigme d'apprentissage multi-agent** dans le trading algorithmique, permettant l'exploration simultanée de multiples stratégies. Cette approche s'avère particulièrement efficace pour le **trading multi-devises** où différents agents peuvent se spécialiser sur des paires de devises spécifiques.[^9][^15]

Les **implémentations A3C multi-workers** montrent des **améliorations significatives en termes de ratio de Sharpe et de drawdown maximum**. Cependant, A3C nécessite des **ressources computationnelles importantes** et peut souffrir de **problèmes de convergence** dans des environnements de trading très bruités.[^12][^16][^9]

### Actor-Critic et Algorithmes Avancés

Les **méthodes Actor-Critic** représentent l'évolution naturelle des approches RL pour le trading, combinant **apprentissage de politique et estimation de valeur**. L'architecture **Risk-Adjusted Deep Reinforcement Learning (RA-DRL)** intègre trois agents spécialisés (log returns, Sharpe ratio, maximum drawdown) pour créer une **politique unifiée risk-adjusted**.[^9][^14]

Les résultats sur des indices réels (Sensex, Dow, TWSE, IBEX) montrent que **RA-DRL atteint les ratios de Sharpe les plus élevés (1.69) et ratios Omega (1.33)**, surpassant significativement les benchmarks traditionnels. Cette approche multi-objectifs résout le problème critique du **choix de fonction de récompense** en finance quantitative.[^14]

## Frameworks Python Optimaux pour l'Implémentation

### TensorFlow/Keras : Écosystème Mature et Production

**TensorFlow demeure le framework de référence** pour les applications de deep learning financier en production. Son **écosystème mature** offre des outils spécialisés comme TensorFlow Probability pour la modélisation d'incertitude, et TensorFlow Serving pour le déploiement à grande échelle. L'API Keras simplifie considérablement l'implémentation des architectures LSTM et GRU pour les séries temporelles.[^17][^18][^19]

L'intégration native de **TensorFlow avec Google Cloud Platform** facilite le déploiement de modèles de trading en temps réel. Les capacités de **distributed training** permettent l'entraînement sur de très grands datasets historiques, essentiel pour capturer les patterns macro-économiques long terme.[^18][^19]

### PyTorch : Excellence en Recherche et Prototypage

**PyTorch s'impose comme le framework préféré** pour la recherche en finance quantitative grâce à son **graphe de calcul dynamique** et sa facilité de débogage. L'écosystème PyTorch Lightning simplifie l'entraînement distributed et la gestion des expériences. Son intégration native avec Jupyter facilite l'exploration interactive des données financières.[^20][^18][^21][^22]

Les **capacités de recherche de PyTorch** brillent particulièrement dans l'expérimentation de nouvelles architectures hybrides et l'intégration de domain knowledge financier dans les modèles. Sa communauté active produit régulièrement des implémentations state-of-the-art adaptées à la finance.[^21][^22]

### FinRL : Spécialisation pour le Reinforcement Learning Financier

**FinRL révolutionne l'application du RL à la finance** en fournissant des **environnements de marché standardisés** et des implémentations d'algorithmes optimisées. La bibliothèque **FinRL-Meta** étend cette approche avec des **centaines d'environnements de marché** générés automatiquement à partir de données réelles.[^23][^24][^25][^26][^27]

L'architecture **DataOps de FinRL-Meta** résout les défis critiques du RL financier : **faible ratio signal/bruit**, **biais de survie**, et **overfitting en backtesting**. La plateforme offre des **benchmarks reproductibles** de papers populaires et des **compétitions communautaires** pour l'évaluation comparative.[^24][^25][^26][^23]

### Intégrations et Écosystème Spécialisé

**QuantConnect LEAN** fournit une **plateforme cloud complète** intégrant recherche, backtesting, et déploiement en trading live. Ses **API Python/C\#** permettent l'intégration seamless de modèles deep learning entraînés avec TensorFlow ou PyTorch. L'accès aux **données institutionnelles multi-asset** (actions, options, futures, crypto) facilite le développement de stratégies sophistiquées.[^28][^29]

**Backtrader** reste une solution de référence pour le **backtesting rapide** avec support natif du trading live via Interactive Brokers et autres brokers. Sa philosophie "batteries incluses" et sa **documentation exceptionnelle** en font l'outil idéal pour valider rapidement des idées de trading algorithmique.[^30][^31]

## Datasets Publics pour l'Entraînement et la Validation

### Sources de Données Gratuites et Accessibles

**Yahoo Finance API** via la bibliothèque yfinance constitue le **point d'entrée standard** pour l'acquisition de données OHLCV. Sa gratuité et sa simplicité d'usage en font l'outil idéal pour le prototypage et l'éducation. Cependant, la **qualité des données peut être limitée** pour les applications de trading haute fréquence.[^32][^33][^34]

**Alpha Vantage** offre une **qualité de données supérieure** avec des APIs robustes couvrant actions, forex, et crypto. Son plan gratuit (500 calls/jour) permet le développement d'applications légères. **Quandl/Nasdaq Data** fournit des **données institutionnelles** avec un focus particulier sur les indicateurs macro-économiques.[^33]

### Plateformes Spécialisées et Données Premium

**Kaggle Financial Datasets** agrège des **datasets annotés** spécifiquement conçus pour l'apprentissage machine. Les compétitions Kaggle financières fournissent des benchmarks standardisés et des baselines de référence. **Federal Reserve FRED** donne accès aux **indicateurs économiques officiels**, essentiels pour la modélisation macro-économique.[^32][^33]

**FinRL-Meta** intègre nativement de **multiples sources de données** dans un pipeline automatisé. Cette approche résout le problème critique de l'**harmonisation des données** multi-sources en finance. Pour les applications professionnelles, **Refinitiv Eikon** et **Bloomberg Terminal** offrent la **référence de qualité** mais à des coûts prohibitifs pour la recherche.[^23][^33][^25]

### Considérations de Qualité et de Complétude

La **qualité des données financières** constitue un défi majeur souvent sous-estimé. Les **données manquantes**, **splits/dividendes**, et **ajustements corporatifs** peuvent fausser dramatiquement les modèles si mal gérés. L'investissement dans des **sources de données fiables** s'avère crucial pour éviter l'illusion de performance en backtesting.[^35][^36]

La **fréquence des données** doit être alignée avec la stratégie de trading. Les données minute conviennent au day trading mais introduisent du bruit excessif pour les stratégies long terme. Le **preprocessing** incluant détection d'outliers, gestion des gaps, et normalisation s'avère essentiel pour la robustesse des modèles.[^7][^37][^34]

## Méthodes de Validation Robustes et Spécificités Temporelles

### Cross-Validation Adaptée aux Séries Temporelles

Les **méthodes de validation traditionnelles** (k-fold aléatoire) sont **inadéquates pour les séries temporelles financières** car elles violent l'ordering temporel. La **Time Series Cross-Validation** ou **walk-forward analysis** constitue l'approche standard, où les modèles sont entraînés sur des fenêtres passées et validés sur des périodes futures.[^38][^39]

**L'expanding window** (fenêtre croissante) utilise toutes les données disponibles jusqu'à la date de prédiction, tandis que la **rolling window** (fenêtre glissante) maintient une taille fixe. Les études montrent que **l'expanding window** perform généralement mieux pour capturer les changements structurels des marchés.[^39][^38]

### Métriques d'Évaluation Spécialisées

Les **métriques financières** diffèrent fondamentalement des métriques ML classiques. Le **ratio de Sharpe** (rendement excédentaire / volatilité) constitue la référence pour l'évaluation risk-adjusted. Un **Sharpe ratio > 1** est généralement considéré comme "bon", > 2 comme "excellent".[^40][^41][^14]

Le **Maximum Drawdown** mesure la perte maximale peak-to-trough, critique pour l'évaluation du risque. Les **ratios Sortino et Calmar** focalisent sur la volatilité downside et sont préférés pour les stratégies risk-averse. L'**Information Ratio** compare les rendements excédentaires relatifs à un benchmark.[^14][^40]

### Backtesting Robuste et Out-of-Sample Testing

Le **backtesting robuste** nécessite une **séparation stricte** entre données d'entraînement, validation, et test. La règle empirique recommande 60% training, 20% validation, 20% test avec **respect de l'ordre chronologique**. Le **paper trading** (simulation en temps réel) constitue l'étape finale avant le déploiement live.[^39][^42][^43]

Les **stress tests** sur différentes conditions de marché (bull, bear, high volatility) révèlent la robustesse des stratégies. Les modèles doivent être **re-entraînés périodiquement** pour s'adapter aux changements de régime. La **dégradation de performance** au fil du temps (model decay) constitue un indicateur critique de maintenance.[^3][^35][^44]

## Limitations et Risques d'Overfitting Spécifiques au Trading

### Overfitting et Surajustement : Le Fléau Principal

**L'overfitting représente le danger le plus insidieux** en finance quantitative. Les modèles surajustés montrent des **performances spectaculaires en backtesting** mais **échouent catastrophiquement en trading live**. Ce phénomène s'explique par la **memorisation du bruit** plutôt que l'apprentissage de patterns généralisables.[^45][^46][^35]

Les **signes d'overfitting** incluent : performances irréalistes (>50% rendement annuel), grand nombre de paramètres, sensibilité extreme aux hyperparamètres, et **détérioration rapide des performances out-of-sample**. Les **techniques de régularisation** (L1/L2, dropout, early stopping) s'avèrent essentielles mais insuffisantes.[^46][^35][^45]

### Look-Ahead Bias et Data Leakage

Le **look-ahead bias** constitue une erreur méthodologique critique où **des informations futures contaminent l'entraînement**. Cette erreur survient typiquement lors du **preprocessing global** (normalisation, feature engineering) avant la séparation temporelle des données.[^47][^42][^48][^44]

La **prévention du look-ahead bias** exige un pipeline rigoureux : séparation temporelle avant tout preprocessing, validation des features (aucune information future), et **point-in-time data** reflétant exactement les données disponibles à chaque moment historique. Les **re-statements** gouvernementaux et ajustements rétroactifs compliquent cette exigence.[^44][^47]

### Data Snooping et Multiple Testing

Le **data snooping** (torture des données) survient lors du test de **multiples stratégies sur le même dataset**. La probabilité de découvrir des patterns par pure chance augmente exponentiellement avec le nombre d'tests. Sans correction statistique appropriée (Bonferroni, False Discovery Rate), les résultats positifs peuvent être **purement fortuits**.[^49]

La **correction de Bonferroni** ajuste le seuil de significativité selon le nombre de tests : p_adjusted = p_original / n_tests. Cette approche conservative peut être excessive en finance. Les **méthodes de validation croisée nested** et **holdout samples** multiples offrent des alternatives plus pratiques.[^45][^49]

### Instabilité et Sensibilité des Modèles RL

**Les algorithmes RL souffrent d'instabilité fondamentale** particulièrement prononcée en finance. La **collecte de données online** et le **feedback delayed** créent des boucles de rétroaction complexes. Les **random seeds** différents peuvent produire des **performances radicalement différentes**, questionnant la reproductibilité.[^50][^16][^51]

Les **failure modes** incluent : collapse de politique (agent qui arrête de trader), exploration insuffisante, et **sensitivity extrême aux hyperparamètres**. La **reward function design** constitue un art délicat où de légères modifications peuvent transformer une stratégie profitable en désastre.[^12][^15][^16][^50]

### Limitations Structurelles et Biais Cognitifs

Les **hypothèses de stationnarité** sous-jacentes aux modèles ML sont **violées systématiquement** par les marchés financiers. Les **changements de régime**, crises, et interventions réglementaires créent des **ruptures structurelles** imprévisibles. Les modèles entraînés sur des périodes "calmes" peuvent exploser lors de turbulences.[^37][^46][^35]

Le **biais de confirmation** pousse les développeurs à **cherry-pick** les périodes/actifs favorables à leurs modèles. La **pression de performance** encourage la multiplication des tests jusqu'à obtenir des résultats satisfaisants, aggravant le data snooping. La **transparence méthodologique** et la **documentation exhaustive** des échecs constituent des garde-fous essentiels.[^47][^35]

## Cas d'Études Documentés et Benchmarks de Performance

### Performances Empiriques et Comparaisons Académiques

Les **études comparatives récentes** sur données Tesla (2015-2024) établissent des **benchmarks de référence** pour les architectures de deep learning. Les résultats montrent une **hiérarchie claire** : xLSTM-TS (94.3% précision) > GRU (91.5%) > LSTM (89.2%) > Transformers (85.7%). Ces performances doivent être **contextualisées par la classe d'actif** et la volatilité.[^1][^7][^4][^5]

Une **méta-analyse de 46 portfolios d'anomalies** révèle que les **modèles de deep learning atteignent des R² cross-sectionnels > 90%**, largement supérieurs aux modèles factoriels traditionnels (Fama-French : ~60%). Les **ratios de Sharpe out-of-sample** atteignent 2.6 pour les modèles neuraux contre 0.8 pour Fama-French.[^40]

### Cas d'Usage Institutionnels et Résultats Réels

**QuantConnect** rapporte des **milliers de stratégies algorithmiques déployées** avec des performances variables. Les stratégies les plus performantes combinent **deep learning pour la prédiction** et **règles de gestion des risques classiques**. L'analyse des échecs révèle que **85-90% des stratégies algorithmiques échouent** la première année.[^35][^52]

**CPP Investments** (600 milliards CAD d'actifs) utilise des **techniques de ML avancées** pour l'extraction de signaux, l'optimisation de portefeuille, et la gestion des risques. Leurs recherches montrent l'importance cruciale de la **qualité des données** et de la **robustesse méthodologique** pour le succès à grande échelle.[^53]

### Hedge Funds et Résultats de Performance

Des **hedge funds quantitatifs leaders** rapportent des **améliorations significatives** grâce au deep learning. Un **hedge fund majeur** utilisant des **modèles hybrides LSTM-TCN** a atteint **15% d'amélioration des rendements de portefeuille** sur deux ans. Ces résultats s'accompagnent d'une **réduction des corrélations inter-actifs** et d'une **meilleure diversification des risques**.[^54]

L'analyse des **40 plus grandes entreprises de FinTech** montre une **adoption massive du deep learning** : 78% utilisent des modèles neuraux pour la détection de fraude, 65% pour l'évaluation de crédit, et 52% pour les stratégies de trading. Les **ROI rapportés** varient de 15% à 40% d'amélioration des métriques business critiques.[^55]

## Recommandations Stratégiques et Meilleures Pratiques

### Architecture Selection et Optimisation

Pour les **applications de prédiction directionnelle**, les **GRU constituent le choix optimal** combinant performance et efficacité computationnelle. Les **LSTM restent préférables** pour les séries très longues (>5 ans) et la modélisation de cycles économiques. Les **Transformers** sont recommandés uniquement pour des applications spécialisées avec des **ressources computationnelles importantes**.[^1][^2][^7][^4][^5]

Les **architectures hybrides** LSTM-GRU avec **preprocessing par ondelettes** représentent l'état de l'art actuel. L'intégration de **couches d'attention** améliore l'interprétabilité sans pénaliser significativement les performances. La **validation par ensemble** de modèles (bagging, stacking) améliore la robustesse au prix de la complexité.[^7][^45][^8]

### Stratégie de Validation et Déploiement

La **séparation temporelle stricte** (train/validation/test : 60%/20%/20%) constitue le minimum méthodologique. Les **tests de stress multi-périodes** (bull markets, bear markets, high volatility) s'avèrent essentiels. Le **paper trading de 3-6 mois minimum** précède tout déploiement de capital réel.[^3][^39][^43]

La **surveillance continue** des métriques de performance (Sharpe ratio glissant, drawdown, correlation avec benchmarks) permet la **détection précoce de dégradation**. Les **alertes automatiques** sur déviations statistiques significatives déclenchent la réévaluation des modèles.[^46][^35]

### Gestion des Risques et Limitations

L'implémentation de **stop-loss dynamiques** et de **limites de drawdown** constitue une obligation non négociable. La **diversification des stratégies** (multiple models, multiple assets, multiple timeframes) réduit les risques de concentration. Les **coûts de transaction réalistes** (doubles spreads, slippage, impact de marché) doivent être intégrés dès la conception.[^45][^36][^56]

La **documentation exhaustive** des hypothèses, limitations, et échecs facilite l'apprentissage organisationnel. La **formation continue** des équipes sur les biais statistiques et cognitifs s'avère cruciale pour maintenir la rigueur méthodologique.[^47][^35]

## Conclusion et Perspectives d'Évolution

L'application du deep learning aux marchés financiers a atteint un degré de maturité remarquable, avec des **architectures spécialisées** (xLSTM-TS, modèles hybrides) surpassant régulièrement les approches traditionnelles. Les **GRU émergent comme l'architecture de choix** pour la plupart des applications grâce à leur excellent rapport performance/complexité, tandis que les **LSTM conservent leur pertinence** pour les analyses long terme.

**Le reinforcement learning révolutionne le trading algorithmique** avec PPO s'imposant comme l'algorithme le plus robuste et RA-DRL démontrant l'efficacité des approches multi-objectifs. L'écosystème logiciel mature (TensorFlow/PyTorch, FinRL, QuantConnect) facilite grandement l'implémentation et le déploiement de solutions professionnelles.

Cependant, **les limitations fondamentales persistent** : overfitting endémique, look-ahead bias, instabilité des modèles RL, et sensibilité aux changements de régime. La **rigueur méthodologique** et la **compréhension profonde des biais** constituent les facteurs déterminants du succès à long terme.

L'avenir s'oriente vers **l'intégration de LLMs** pour le traitement de données non-structurées, les **architectures multi-agents** pour la gestion de portefeuilles complexes, et les **approches quantiques** pour l'optimisation combinatoire. Néanmoins, **les principes fondamentaux** de validation rigoureuse, gestion des risques, et humilité face à l'incertitude des marchés demeurent les piliers incontournables de toute approche quantitative réussie.

<div style="text-align: center">⁂</div>

[^1]: http://arxiv.org/pdf/2411.05790.pdf

[^2]: https://arxiv.org/html/2505.05325v1

[^3]: https://journalwjaets.com/sites/default/files/fulltext_pdf/WJAETS-2025-0167.pdf

[^4]: https://journals.scholarpublishing.org/index.php/TMLAI/article/download/18843/10851/26804

[^5]: https://www.themoonlight.io/en/review/comparative-analysis-of-lstm-gru-and-transformer-models-for-stock-price-prediction

[^6]: https://www.atlantis-press.com/proceedings/icfied-24/125999624

[^7]: https://arxiv.org/html/2408.12408v1

[^8]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5023739

[^9]: https://arxiv.org/pdf/2405.19982.pdf

[^10]: https://www.daytrading.com/reinforcement-learning-algorithms

[^11]: https://www.atlantis-press.com/article/125998082.pdf

[^12]: https://arxiv.org/html/2411.07585v1

[^13]: https://arxiv.org/pdf/2407.14151.pdf

[^14]: https://link.springer.com/article/10.1007/s44196-025-00875-8

[^15]: https://link.springer.com/article/10.1007/s10489-023-04959-w

[^16]: https://www.alexirpan.com/2018/02/14/rl-hard.html

[^17]: https://www.geeksforgeeks.org/deep-learning/time-series-forecasting-using-tensorflow/

[^18]: https://www.coursera.org/articles/tensorflow-or-pytorch

[^19]: https://www.tensorflow.org/tutorials/structured_data/time_series

[^20]: https://www.geeksforgeeks.org/data-analysis/time-series-forecasting-using-pytorch/

[^21]: https://www.youtube.com/watch?v=IJ50ew8wi-0

[^22]: https://www.youtube.com/watch?v=oWJIMv6y1rM

[^23]: https://arxiv.org/abs/2304.13174

[^24]: https://openreview.net/forum?id=LkAFwrqdRY6

[^25]: https://arxiv.org/abs/2211.03107

[^26]: https://github.com/AI4Finance-Foundation/FinRL-Meta

[^27]: https://github.com/AI4Finance-Foundation/FinRL

[^28]: https://docs.pytrade.org/trading

[^29]: https://www.quantconnect.com/docs/v2/cloud-platform/learning-center/training

[^30]: https://www.backtrader.com/docu/quickstart/quickstart/

[^31]: https://www.quantstart.com/articles/backtesting-systematic-trading-strategies-in-python-considerations-and-open-source-frameworks/

[^32]: https://www.deepchecks.com/best-free-financial-datasets-machine-learning/

[^33]: https://labelyourdata.com/articles/financial-datasets-for-machine-learning

[^34]: https://imerit.net/resources/blog/top-10-stock-market-datasets-for-machine-learning-all-pbm/

[^35]: https://site.financialmodelingprep.com/education/financial-analysis/Using-Machine-Learning-for-Stock-Market-Prediction-Possibilities-and-Limitations

[^36]: https://www.youtube.com/watch?v=2a6EXIsB2tU

[^37]: https://arxiv.org/pdf/2305.04811.pdf

[^38]: https://milvus.io/ai-quick-reference/what-is-the-role-of-crossvalidation-in-time-series-analysis

[^39]: https://www.quantstart.com/articles/Using-Cross-Validation-to-Optimise-a-Machine-Learning-Method-The-Regression-Setting/

[^40]: https://economics.yale.edu/sites/default/files/deep_learning_in_asset_pricing.pdf

[^41]: https://www.investopedia.com/terms/s/sharperatio.asp

[^42]: https://towardsdatascience.com/3-common-time-series-modeling-mistakes-you-should-know-a126df24256f/

[^43]: https://www.fmz.com/lang/fr/digest-topic/3963

[^44]: https://www.marketcalls.in/machine-learning/understanding-look-ahead-bias-and-how-to-avoid-it-in-trading-strategies.html

[^45]: https://www.numberanalytics.com/blog/overfitting-quant-finance-prevention-strategies

[^46]: https://onemoneyway.com/en/dictionary/overfitting/

[^47]: https://bowtiedraptor.substack.com/p/look-ahead-bias-and-how-to-prevent

[^48]: https://codesignal.com/learn/courses/preparing-financial-data-for-machine-learning/lessons/addressing-data-leakage-in-time-series?identifier=2340%2Caddressing-data-leakage-in-time-series

[^49]: https://quant.fish/wiki/data-snooping-in-algorithmic-trading/

[^50]: https://milvus.io/ai-quick-reference/what-are-the-limitations-of-reinforcement-learning

[^51]: https://www.quantifiedstrategies.com/reinforcement-learning-in-trading/

[^52]: https://journaldesseniors.20minutes.fr/actualite/10-erreurs-fatales-en-day-trading-guide-du-debutant-pour-les-eviter/

[^53]: https://themoneyoutlook.com/machine-learning-for-quantitative-finance-use-cases-and-challenges/

[^54]: https://ijrpr.com/uploads/V6ISSUE1/IJRPR38184.pdf

[^55]: https://research.aimultiple.com/deep-learning-in-finance/

[^56]: https://www.youtube.com/watch?v=ffE4uFFq1XU

[^57]: https://www.sciencedirect.com/science/article/abs/pii/S1568494625005526

[^58]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729146

[^59]: https://www.atlantis-press.com/article/125999624.pdf

[^60]: https://www.iguazio.com/blog/best-13-free-financial-datasets-for-machine-learning/

[^61]: https://digitaldefynd.com/IQ/deep-learning-case-studies/

[^62]: https://github.com/Leefinance/Quantitative-finance-papers-using-deep-learning

[^63]: https://www.tigerdata.com/learn/time-series-analysis-and-forecasting-with-python

[^64]: https://bfi.uchicago.edu/wp-content/uploads/2023/07/BFI_WP_2023-100.pdf

[^65]: https://www.geeksforgeeks.org/machine-learning/stock-price-prediction-using-machine-learning-in-python/

[^66]: https://arxiv.org/html/2505.22678v1

[^67]: https://consensus.app/search/what-are-the-limitations-of-deep-learning-algorith/m-Jw3etlRQmTtKEmIl_Wxg/

[^68]: https://consensus.app/search/what-are-the-limitations-of-using-deep-learning-fo/ReQAM_jrRrOyVE0MtJQROw/

[^69]: https://papers.ssrn.com/sol3/Delivery.cfm/5022165.pdf?abstractid=5022165\&mirid=1

[^70]: https://arxiv.org/html/2503.21422v1

[^71]: https://arxiv.org/html/2504.02281v3

[^72]: https://findingtheta.com/blog/using-reinforcement-learning-for-stock-trading-with-finrl

[^73]: https://arxiv.org/html/2405.10119v1

[^74]: https://finrl.readthedocs.io/en/latest/index.html

[^75]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/d6955ee87b3126f91a0603b86eb2303e/8c4099b5-1a9d-4fe6-8410-ebf705dedc59/0d08dcdd.csv

[^76]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/d6955ee87b3126f91a0603b86eb2303e/8c4099b5-1a9d-4fe6-8410-ebf705dedc59/26dfa9f2.csv

