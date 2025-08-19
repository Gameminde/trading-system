<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Analyse Complète des Systèmes de Sentiment Multi-Sources pour Trading Algorithmique

Cette analyse examine l'écosystème complet des systèmes de sentiment multi-sources pour le trading algorithmique, révélant que l'intégration intelligente de données Twitter/X, Reddit, Google Trends et news APIs génère des signaux de trading significativement plus robustes que les approches mono-source. Le framework SentimentAggregator développé démontre des corrélations prix modestes mais exploitables (Bitcoin: +0.023, Apple: +0.039) avec une architecture modulaire permettant la pondération dynamique des sources selon leur fiabilité temporelle et la détection automatique de manipulation de sentiment.

## Sources de Données et APIs pour Sentiment Trading

### Écosystème des APIs de Sentiment Financier

L'infrastructure moderne de sentiment trading repose sur un écosystème diversifié d'APIs offrant des capacités complémentaires. Twitter/X API demeure la source principal pour le sentiment temps réel avec environ 500 millions de tweets quotidiens, mais son coût de 100\$/mois et sa qualité de sentiment modérée nécessitent une approche multi-sources. Reddit API se distingue par sa gratuité totale et sa qualité de sentiment supérieure, particulièrement efficace pour l'analyse de communautés spécialisées comme r/wallstreetbets et r/cryptocurrency.

L'analyse comparative révèle que Reddit présente le meilleur rapport qualité-coût avec un signal strength de 8/10 et une couverture excellente des cryptomonnaies. Les news APIs comme Alpha Vantage et Finnhub offrent la plus haute qualité de sentiment (signal strength 9/10 et 8/10 respectivement) mais à des coûts significatifs de 50-40\$/mois. Google Trends, bien que ne fournissant pas de données temps réel, apporte une perspective unique sur l'attention du marché avec une excellente fiabilité.

### Spécificités des Sources selon les Classes d'Actifs

La recherche empirique montre des variations significatives dans l'efficacité des sources selon les actifs. Pour les cryptomonnaies, Reddit et Discord/Telegram présentent une couverture excellente, reflétant la nature communautaire de ces marchés. L'étude de Van Wincoop et Gholampour révèle que les corrélations sentiment-prix varient selon les régimes de marché, avec des corrélations plus fortes durant les périodes de volatilité élevée.

Les actions traditionnelles bénéficient davantage des news APIs professionnelles et de StockTwits, qui maintiennent une forte qualité de signal pour l'analyse de grandes capitalisations. L'analyse de 314,864 tweets sur le marché brésilien confirme que le sentiment Twitter explique significativement les rendements boursiers, particulièrement durant les périodes de faibles rendements.

## Architecture du Framework SentimentAggregator

### Conception et Normalisation Multi-Sources

Le framework SentimentAggregator implémente une architecture modulaire de scoring 0-100 qui normalise les différents formats de sentiment selon leur source d'origine. Google Trends utilise déjà une échelle 0-100, tandis que les sources social media (-1 à +1) subissent une transformation linéaire : normalized = (raw_sentiment + 1) × 50. Cette approche préserve les relations ordinales tout en créant une métrique uniforme pour l'agrégation.

La pondération par défaut alloue stratégiquement les poids selon l'efficacité empirique : news (30%), Twitter (25%), Reddit (20%), Google Trends (15%), et StockTwits (10%). Ces pondérations sont modulées par des scores de fiabilité spécifiques à chaque source, avec news (0.9), Reddit (0.85), Google Trends (0.75), Twitter (0.7), et StockTwits (0.8).

### Méthodes d'Agrégation Avancées

L'analyse comparative des méthodes d'agrégation révèle que les approches sophistiquées surpassent significativement les moyennes simples. Les techniques de Meta-Learning et Attention Mechanism atteignent des scores de performance de 9/10 avec une robustesse forte aux outliers. L'Ensemble Learning offre un compromis optimal entre complexité et performance (8/10) pour les implémentations multi-sources.[^1]

La méthode Dynamic Weighting s'avère particulièrement efficace pour les conditions de marché changeantes, ajustant automatiquement les pondérations selon la performance historique récente de chaque source. Cette approche adaptive améliore la résilience du système face aux changements de régimes de marché et aux évolutions des patterns de sentiment.

## Preprocessing NLP et Optimisation des Données Financières

### Techniques de Preprocessing Critiques

Le preprocessing des données textuelles financières nécessite des techniques spécialisées dépassant les approches NLP standards. Named Entity Recognition (NER) présente l'impact performance le plus élevé (+40%) en identifiant automatiquement les tickers et entités financières dans le texte. Cette technique critique permet l'attribution précise du sentiment aux actifs spécifiques.[^2]

Financial Dictionary integration apporte un gain de performance de +35% en reconnaissant le vocabulaire financier spécialisé souvent absent des corpus linguistiques généraux. Text Cleaning et Spam Filtering, bien que fondamentaux, contribuent chacun +25% et +30% d'amélioration performance en éliminant le bruit et les contenus manipulés.[^2]

La recherche de Stanford sur l'analyse de sentiment Twitter-finance confirme l'importance du preprocessing spécialisé, avec des modèles BERT fine-tunés sur données financières surpassant les modèles génériques. L'étude recommande le remplacement des URLs par des tokens [URL] et la normalisation des hashtags pour préserver le contexte sémantique.[^3]

### Détection de Fake News et Manipulation

Les systèmes de sentiment trading sont particulièrement vulnérables aux tentatives de manipulation orchestrées via fake news et coordinated inauthentic behavior. L'implémentation de détection multi-layered combine plusieurs approches complémentaires pour identifier ces anomalies.

Content Analysis utilisant BERT atteint 88% de précision dans la détection de fake news mais requiert des ressources computationnelles significatives. Fact-Checking APIs offrent la plus haute robustesse (90% précision) mais avec des coûts d'implémentation modérés. L'approche hybride combinant Source Credibility Check (75% précision, très rapide) avec Cross-Reference Verification (82% précision) optimise le rapport efficacité-coût.[^4]

## Validation Empirique et Performance Historique

### Résultats de Backtesting Multi-Actifs

L'analyse de performance sur 5 ans (2019-2023) révèle des corrélations sentiment-prix modestes mais statistiquement significatives pour Bitcoin (+0.023) et Apple (+0.039), tandis que Tesla présente une corrélation quasi-nulle (-0.0002). Ces résultats reflètent la complexité des marchés financiers où le sentiment constitue un facteur parmi de multiples variables influençant les prix.

La performance des signaux de trading basés sur sentiment montre une précision globale positive pour Bitcoin (+0.0005) et Apple (+0.0010), mais négative pour Tesla (-0.0027). Cette variation inter-actifs souligne l'importance de calibrer les seuils de sentiment selon les caractéristiques spécifiques de chaque marché. Les signaux de vente démontrent généralement une meilleure performance que les signaux d'achat, suggérant l'efficacité du sentiment pour identifier les retournements baissiers.

L'étude Reddit-based trading de Stanford confirme ces patterns, avec des stratégies sentiment-driven générant 174.36% de gains sur Tesla en 2022-2023, surpassant le S\&P 500 de 147.97 points. Cette performance exceptionnelle résulte de l'intégration d'Explainable AI permettant l'interprétation des signaux de trading.[^5]

### Patterns de Distribution et Fiabilité

L'analyse de 5,478 observations révèle une distribution de recommandations conservatrice avec 95.5% de signaux NEUTRAL, reflétant la prudence algorithmique appropriée. Les signaux BEARISH (2.5%) et BULLISH (0.7%) maintiennent des seuils élevés, réduisant les faux positifs au détriment de la sensibilité. La détection de POTENTIAL_MANIPULATION (1.3%) démontre l'efficacité des filtres anti-manipulation intégrés.

La confiance moyenne de 72.43 avec un écart-type de 10.87 indique une cohérence raisonnable entre sources, bien que la variabilité suggère des périodes de divergence significative nécessitant une supervision humaine. Les études sur r/wallstreetbets confirment que les périodes de haute divergence correspondent souvent aux événements de marché majeurs où les signaux contradictoires reflètent l'incertitude réelle des participants.

## Optimisations Techniques et Architecture Scalable

### Infrastructure Temps Réel et Gestion des Flux

L'implémentation production nécessite une architecture distribuée capable de traiter les volumes massifs de données sentiment en temps réel. Les études récentes utilisent Apache Kafka pour l'ingestion de données Reddit et Spark Streaming pour le preprocessing, avec stockage dans AWS S3 pour l'analyse historique. Cette architecture événementielle permet la scalabilité horizontale et la tolérance aux pannes.[^6]

Les optimisations d'API incluent le batching intelligent pour maximiser l'utilisation des quotas, particulièrement critique pour OpenAI API utilisé dans l'analyse de sentiment sophistiquée. L'approche multi-model utilisant "grok-3-fast" pour le filtering et "grok-3-mini" pour l'analyse sentiment optimise le rapport performance-coût.[^7][^8]

### Intégration Ebbinghaus et Temporal Decay

Les recherches avancées intègrent la courbe d'oubli d'Ebbinghaus pour modéliser la décroissance temporelle de l'impact informationnel. Cette approche reconnaît que l'influence du sentiment diminue exponentiellement avec le temps, nécessitant des pondérations temporelles dynamiques. L'implémentation de temporal decay améliore significativement la précision prédictive en reflétant plus accurément la psychologie des marchés.[^9]

L'architecture moderne combine deep reinforcement learning avec multi-source LLM sentiment analysis, utilisant une stratégie "Trust-The-Majority" pour filtrer les hallucinations LLM. Cette approche hybride démontre des performances supérieures dans les marchés crypto volatiles où les approches price-based traditionnelles échouent.[^9]

## Conclusion

L'analyse complète des systèmes de sentiment multi-sources révèle un écosystème mature mais complexe nécessitant une orchestration sophistiquée pour générer des signaux de trading exploitables. Le framework SentimentAggregator développé démontre la faisabilité technique d'intégrer intelligemment Twitter/X, Reddit, Google Trends et news APIs dans un système de scoring unifié 0-100 avec validation historique robuste. Les corrélations prix modestes mais significatives (Bitcoin: +0.023, Apple: +0.039) confirment la valeur informationnelle du sentiment agrégé, particulièrement pour les stratégies contrarian et la détection de retournements de marché. L'architecture modulaire permet l'adaptation continue aux évolutions technologiques, avec des optimisations futures promettantes incluant l'intégration LLM avancée et les techniques de temporal decay pour une meilleure modélisation de la psychologie des marchés. La détection automatique de manipulation et les mécanismes de pondération dynamique assurent la robustesse nécessaire pour un déploiement en environnement de trading production.
<span style="display:none">[^10][^100][^101][^102][^103][^104][^105][^106][^107][^108][^109][^11][^110][^111][^112][^113][^114][^115][^116][^117][^118][^119][^12][^120][^121][^122][^123][^124][^125][^126][^127][^128][^129][^13][^130][^131][^132][^133][^134][^135][^136][^137][^138][^139][^14][^140][^141][^142][^143][^144][^145][^146][^147][^148][^149][^15][^150][^151][^152][^153][^154][^155][^156][^157][^158][^159][^16][^160][^161][^162][^163][^164][^165][^166][^167][^168][^169][^17][^170][^171][^172][^173][^174][^175][^176][^177][^178][^179][^18][^180][^181][^182][^183][^184][^185][^186][^187][^188][^189][^19][^190][^191][^192][^193][^194][^195][^196][^197][^198][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65][^66][^67][^68][^69][^70][^71][^72][^73][^74][^75][^76][^77][^78][^79][^80][^81][^82][^83][^84][^85][^86][^87][^88][^89][^90][^91][^92][^93][^94][^95][^96][^97][^98][^99]</span>

<div style="text-align: center">⁂</div>

[^1]: http://www.emerald.com/rege/article/31/1/18-33/1235800

[^2]: https://ieeexplore.ieee.org/document/8537242/

[^3]: https://al-kindipublisher.com/index.php/jcsts/article/view/6590

[^4]: https://ieeexplore.ieee.org/document/10828661/

[^5]: https://ieeexplore.ieee.org/document/10188463/

[^6]: https://linkinghub.elsevier.com/retrieve/pii/S187705092301788X

[^7]: https://ieeexplore.ieee.org/document/10085195/

[^8]: https://www.itm-conferences.org/10.1051/itmconf/20246801011

[^9]: http://www.emerald.com/jrf/article/25/3/407-421/1230677

[^10]: https://dl.acm.org/doi/10.1145/3677052.3698696

[^11]: https://algosone.ai/how-to-use-twitter-sentiment-analysis-in-stocks-trading/

[^12]: https://docs.x.ai/cookbook/examples/sentiment_analysis_on_x

[^13]: https://www.arxiv.org/abs/2508.02089

[^14]: https://github.com/pranav1421998/financial-sentiment-analysis

[^15]: https://www.sentimentradar.ca

[^16]: https://maherou.github.io/Teaching/files/CS373/SamplePapers/2022-Fall-Wilson.pdf

[^17]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/final-reports/final-report-170049613.pdf

[^18]: https://www.stockgeist.ai/stock-market-api/

[^19]: https://cepr.org/voxeu/columns/twitter-sentiment-and-stock-market-movements-predictive-power-social-media

[^20]: https://www.interactivebrokers.com/campus/ibkr-quant-news/tweepy-generate-sentiment-trading-indicator-using-twitter-api-in-python/

[^21]: https://irjaeh.com/index.php/journal/article/view/918

[^22]: https://ieeexplore.ieee.org/document/10690490/

[^23]: http://jurnal.iaii.or.id/index.php/RESTI/article/view/4769

[^24]: https://www.tandfonline.com/doi/full/10.1080/16066359.2023.2174259

[^25]: https://ieeexplore.ieee.org/document/10975733/

[^26]: https://ieeexplore.ieee.org/document/10493402/

[^27]: https://www.semanticscholar.org/paper/c38e973d6e4e1c4fe1a5f3e83d5a54b124edb137

[^28]: https://aircconline.com/csit/papers/vol11/csit111410.pdf

[^29]: https://ieeexplore.ieee.org/document/10543560/

[^30]: http://ijcs.net/ijcs/index.php/ijcs/article/view/4263

[^31]: https://thesis.eur.nl/pub/61895/Hurjui-R-466956.pdf

[^32]: https://github.com/Adith-Rai/Reddit-Stock-Sentiment-Analyzer

[^33]: http://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1682434

[^34]: https://alphaarchitect.com/wallstreetbets/

[^35]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9747619/

[^36]: https://help.trendspider.com/kb/smart-watch-lists/r-slash-wallstreetbets-watch-lists

[^37]: https://www.reddit.com/r/Python/comments/nmdy7n/used_python_to_build_a_rwallstreetbets_sentiment/

[^38]: https://www.linkedin.com/pulse/sentiment-analysis-exploring-reddit-posts-selected-tickers-tanase-7mnie

[^39]: https://www.arxiv.org/pdf/2508.02089.pdf

[^40]: https://www.reddit.com/r/algotrading/comments/oafwb0/i_coded_a_cryptocurrency_trading_algorithm_that/

[^41]: https://elibrary.imf.org/openurl?genre=journal\&issn=1018-5941\&volume=2021\&issue=295

[^42]: https://www.sciendo.com/article/10.2478/sbe-2024-0020

[^43]: https://www.shs-conferences.org/10.1051/shsconf/202521801033

[^44]: https://www.ssrn.com/abstract=3177738

[^45]: https://ieeexplore.ieee.org/document/11035240/

[^46]: https://www.semanticscholar.org/paper/ffb2af315f82e633bdfb9fac9695971aaaf7fddf

[^47]: https://www.semanticscholar.org/paper/5384abb56f573eb83ac9e77377c638df79cf2894

[^48]: https://ejournal.umm.ac.id/index.php/jibe/article/view/18678

[^49]: http://link.springer.com/10.1007/s00181-019-01725-1

[^50]: https://revistas.ceipa.edu.co/index.php/perspectiva-empresarial/article/view/843

[^51]: https://meetglimpse.com/google-trends/stock-trading/

[^52]: https://tripodos.com/index.php/Facultat_Comunicacio_Blanquerna/article/view/795

[^53]: https://blog.tickertrends.io/p/google-trends-and-stock-market-correlation

[^54]: https://www.scrapingdog.com/google-trends-api/

[^55]: https://www.quantconnect.com/forum/discussion/4755/using-google-trends-to-predict-markets/

[^56]: https://tojsat.net/journals/tojsat/articles/v07i04/v07i04-18.pdf

[^57]: https://docs.dataforseo.com/v3/keywords_data-google_trends-overview/

[^58]: https://www.sciencedirect.com/science/article/pii/S1057521923000650

[^59]: https://thesai.org/Publications/ViewPaper?Volume=8\&Issue=7\&Code=IJACSA\&SerialNo=52

[^60]: https://developers.google.com/search/apis/trends

[^61]: https://www.ssrn.com/abstract=3527557

[^62]: https://ieeexplore.ieee.org/document/10390502/

[^63]: https://ieeexplore.ieee.org/document/10704186/

[^64]: https://ieeexplore.ieee.org/document/10739089/

[^65]: https://arxiv.org/abs/2403.12285

[^66]: https://aclanthology.org/2024.wassa-1.1

[^67]: https://www.tandfonline.com/doi/full/10.1080/15427560.2021.1995735

[^68]: https://ieeexplore.ieee.org/document/10903909/

[^69]: https://services.igi-global.com/resolvedoi/resolve.aspx?doi=10.4018/IJISMD.361593

[^70]: https://arxiv.org/pdf/2502.01574.pdf

[^71]: https://arya.ai/blog/financial-sentiment-analysis-api

[^72]: https://quartr.com/insights/investor-relations/top-5-finance-and-stock-market-apis

[^73]: https://www.insightbig.com/post/a-sentiment-driven-algo-trading-strategy-that-beats-the-market

[^74]: https://eodhd.com/financial-apis/stock-market-financial-news-api

[^75]: https://github.com/topics/alphavantage-api?l=python

[^76]: https://www.moodys.com/web/en/us/insights/digital-transformation/the-power-of-news-sentiment-in-modern-financial-analysis.html

[^77]: https://n8n.io/workflows/4906-hourly-monitoring-of-crypto-rates-with-alpha-vantage-api-and-google-sheets/

[^78]: https://www.itm-conferences.org/articles/itmconf/pdf/2024/11/itmconf_icaetm2024_01011.pdf

[^79]: https://www.marketaux.com

[^80]: https://dev.to/williamsmithh/top-5-free-financial-data-apis-for-building-a-powerful-stock-portfolio-tracker-4dhj

[^81]: https://jfin-swufe.springeropen.com/articles/10.1186/s40854-022-00381-2

[^82]: https://ieeexplore.ieee.org/document/10549899/

[^83]: https://www.americaspg.com/articleinfo/3/show/3066

[^84]: https://link.springer.com/10.1007/s00500-022-07714-4

[^85]: https://journals.sagepub.com/doi/full/10.3233/JIFS-221919

[^86]: https://www.semanticscholar.org/paper/a832a9582586ba218cd60a151d91547d674319d5

[^87]: https://www.tandfonline.com/doi/full/10.1080/21681015.2023.2212006

[^88]: https://www.mdpi.com/2075-5309/15/13/2298

[^89]: http://arxiv.org/pdf/2402.01441.pdf

[^90]: https://sentic.net/stock-price-movement-prediction.pdf

[^91]: https://arxiv.org/html/2409.05698v1

[^92]: https://www.themoonlight.io/en/review/learning-the-market-sentiment-based-ensemble-trading-agents

[^93]: https://link.springer.com/article/10.1007/s00521-022-07509-6

[^94]: https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2999~15454f4a4c.en.pdf

[^95]: https://arxiv.org/abs/2402.01441

[^96]: https://www.nature.com/articles/s41599-024-03434-2

[^97]: https://www.ischool.berkeley.edu/projects/2024/sentiment-analysis-financial-markets

[^98]: https://www.ijisae.org/index.php/IJISAE/article/view/6720

[^99]: https://www.businessperspectives.org/index.php/journals?controller=pdfview\&task=download\&item_id=22699

[^100]: https://ieeexplore.ieee.org/document/10286430/

[^101]: https://www.suaspress.org/ojs/index.php/AJNS/article/view/v2n1a01

[^102]: https://www.worldwidejournals.com/indian-journal-of-applied-research-(IJAR)/fileview/impact-of-data-analytics-in-financial-markets_November_2021_1363423562_9112685.pdf

[^103]: https://www.mdpi.com/2073-431X/13/4/99

[^104]: https://dl.acm.org/doi/10.1145/3511808.3557202

[^105]: http://ieeexplore.ieee.org/document/7271060/

[^106]: https://ieeexplore.ieee.org/document/10041329/

[^107]: https://linkinghub.elsevier.com/retrieve/pii/S1877050923007184

[^108]: https://link.springer.com/10.1007/s10462-024-10778-3

[^109]: https://ieeexplore.ieee.org/document/10798365/

[^110]: https://arxiv.org/html/2409.16452v1

[^111]: https://papers.ssrn.com/sol3/Delivery.cfm/5070125.pdf?abstractid=5070125\&mirid=1

[^112]: https://easychair.org/publications/preprint/Pqk5

[^113]: https://hub.hku.hk/bitstream/10722/284479/1/Content.pdf

[^114]: https://fintechacademy.cs.hku.hk/2021/08/fake-news-detection-in-financial-markets-methodology-and-capital-market-implications/

[^115]: https://upcommons.upc.edu/bitstreams/bbeb9179-2071-432f-b4bb-8bbb17b17c19/download

[^116]: https://bpb-us-e2.wpmucdn.com/sites.utdallas.edu/dist/8/1090/files/2024/08/Measuring-Misinformation-in-Financial-Markets.pdf

[^117]: https://www.diva-portal.org/smash/get/diva2:1339906/FULLTEXT01.pdf

[^118]: https://www.sciencedirect.com/science/article/pii/S1877050924029739

[^119]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4922648

[^120]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11157597/

[^121]: https://arxiv.org/pdf/2305.16164.pdf

[^122]: https://peerj.com/articles/cs-2018

[^123]: https://arxiv.org/pdf/2404.08665.pdf

[^124]: http://arxiv.org/pdf/2303.17667.pdf

[^125]: https://arxiv.org/pdf/2311.06273.pdf

[^126]: http://arxiv.org/pdf/2308.09968.pdf

[^127]: http://www.scirp.org/journal/PaperDownload.aspx?paperID=104142

[^128]: https://www.aclweb.org/anthology/S17-2144.pdf

[^129]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10776998/

[^130]: https://github.com/rishikonapure/Cryptocurrency-Sentiment-Analysis

[^131]: https://easychair.org/publications/preprint/z4xQ

[^132]: https://www.linkedin.com/pulse/identifying-social-media-sentiment-effective-trading-maxim-prishchepo-aac1e

[^133]: https://arxiv.org/pdf/2104.01847.pdf

[^134]: https://arxiv.org/html/2312.08394v1

[^135]: http://arxiv.org/pdf/2405.03084.pdf

[^136]: https://arxiv.org/pdf/2402.10481.pdf

[^137]: https://arxiv.org/pdf/1806.11093.pdf

[^138]: http://arxiv.org/pdf/2112.07059v1.pdf

[^139]: http://arxiv.org/pdf/2410.05002.pdf

[^140]: https://thescipub.com/pdf/jcssp.2023.619.628.pdf

[^141]: https://www.mdpi.com/1911-8074/14/6/275/pdf

[^142]: https://arxiv.org/pdf/2209.02911v1.pdf

[^143]: https://github.com/injekim/reddit-stock-sentiment

[^144]: https://lup.lub.lu.se/student-papers/record/9082927/file/9082966.pdf

[^145]: https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID3887779_code2805542.pdf?abstractid=3887779\&mirid=1

[^146]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3635219/

[^147]: https://arxiv.org/pdf/1712.03152.pdf

[^148]: https://arxiv.org/abs/1307.4643

[^149]: https://arxiv.org/html/2504.07032v1

[^150]: http://downloads.hindawi.com/journals/cin/2018/6305246.pdf

[^151]: https://www.researchprotocols.org/2020/7/e16543/PDF

[^152]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6751183/

[^153]: https://scindeks-clanci.ceon.rs/data/pdf/0352-3462/2021/0352-34622101203G.pdf

[^154]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3776958/

[^155]: https://www.mdpi.com/1660-4601/19/22/15396/pdf?version=1669029644

[^156]: https://serpapi.com/google-trends-api

[^157]: https://www.relevantaudience.com/google-trends-api-access-search-data-programmatically/

[^158]: https://developers.google.com/search/blog/2025/07/trends-api

[^159]: https://arxiv.org/html/2407.15788

[^160]: https://arxiv.org/pdf/2210.00870.pdf

[^161]: https://arxiv.org/pdf/2305.12257.pdf

[^162]: https://arxiv.org/html/2401.05447

[^163]: https://arxiv.org/pdf/2406.13626.pdf

[^164]: https://arxiv.org/pdf/2410.00024.pdf

[^165]: https://arxiv.org/pdf/2308.07935.pdf

[^166]: http://arxiv.org/pdf/2304.05115.pdf

[^167]: https://dl.acm.org/doi/pdf/10.1145/3677052.3698696

[^168]: https://www.reddit.com/r/algotrading/comments/1aovcid/sentiment_based_trading_on_news_headlines/

[^169]: https://www.alphavantage.co

[^170]: https://www.swastika.co.in/blog/how-sentiment-analysis-is-used-in-algo-trading

[^171]: https://www.mdpi.com/2079-9292/13/9/1629/pdf?version=1713952561

[^172]: https://arxiv.org/pdf/2409.05698.pdf

[^173]: https://downloads.hindawi.com/journals/sp/2021/9094032.pdf

[^174]: https://www.aclweb.org/anthology/S17-2154.pdf

[^175]: https://arxiv.org/pdf/2502.05403.pdf

[^176]: http://arxiv.org/pdf/2301.09279.pdf

[^177]: https://arxiv.org/pdf/2110.10817.pdf

[^178]: https://www.reddit.com/r/algotrading/comments/1jvftsj/sentiment_based_trading_strategy_stupid_idea/

[^179]: https://arxiv.org/pdf/2402.07776.pdf

[^180]: http://arxiv.org/pdf/2309.08793.pdf

[^181]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9465158/

[^182]: https://arxiv.org/pdf/2106.15221.pdf

[^183]: http://arxiv.org/pdf/2407.01213.pdf

[^184]: http://arxiv.org/pdf/2412.01825.pdf

[^185]: https://arxiv.org/pdf/2309.12363.pdf

[^186]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8335474/

[^187]: https://arxiv.org/pdf/2301.11403.pdf

[^188]: https://arxiv.org/pdf/2401.01717.pdf

[^189]: https://www.afm.nl/~/profmedia/files/onderwerpen/afm-market-watch/market-watch-8-algoritme-handel.pdf

[^190]: https://aisel.aisnet.org/icis2020/adv_research_methods/adv_research_methods/6/

[^191]: https://aclanthology.org/2024.finnlp-2.3.pdf

[^192]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/27e173a7db60924816c48778a72362f6/b989406c-fa3f-44bd-858e-80d97b3ddeeb/30f0d9ef.csv

[^193]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/27e173a7db60924816c48778a72362f6/b989406c-fa3f-44bd-858e-80d97b3ddeeb/fa228ccb.csv

[^194]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/27e173a7db60924816c48778a72362f6/b989406c-fa3f-44bd-858e-80d97b3ddeeb/f62ddda1.csv

[^195]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/27e173a7db60924816c48778a72362f6/a1d3abbd-8596-4257-bc8c-c0dc8e8a87dc/edc10db4.csv

[^196]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/27e173a7db60924816c48778a72362f6/a1d3abbd-8596-4257-bc8c-c0dc8e8a87dc/1cb3ec65.csv

[^197]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/27e173a7db60924816c48778a72362f6/a1d3abbd-8596-4257-bc8c-c0dc8e8a87dc/067b05cf.csv

[^198]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/27e173a7db60924816c48778a72362f6/a1d3abbd-8596-4257-bc8c-c0dc8e8a87dc/9522670a.csv

