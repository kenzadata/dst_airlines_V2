# DST Airlines — Tracking de vols & Prédiction de retards

PFE — Formation Data Engineering, Liora

## 1. Contexte et objectif

DST Airlines vise à s'approcher du niveau de service de Flightradar24 en agrégeant des API de compagnies/données aériennes pour du suivi de vols en temps réel, puis à aller plus loin en prédisant le **retard** d'un vol grâce à un modèle de Machine Learning.

**Périmètre du MVP** : 2 à 3 aéroports pilotes, horizon de quelques jours, pipeline complet collecte → stockage → ML → restitution → API.

Documents de cadrage déjà produits (à lire avant de coder) :
- `Cahier_des_charges_DST_Airlines.docx` — Discovery, veille technologique, SWOT
- `MVP_DST_Airlines.docx` — priorisation MoSCoW, architecture cible, risques
- `Roadmap_DST_Airlines.docx` — sprints, livrables, KPIs, stratégie de tests

## 2. Stack technique

| Brique | Choix retenu | Alternative envisageable |
|---|---|---|
| Langage | Python 3.11+ | — |
| Collecte de données | `requests` / `httpx` vers API OpenSky Network + API météo (Open-Meteo) | AviationStack, Aviation Edge (si quotas OpenSky insuffisants) |
| Stockage brut | PostgreSQL (ou SQLite en dev local) | Data lake léger (fichiers Parquet) |
| Pipeline ETL | Python (pandas) + orchestration simple (cron ou Airflow léger) | dbt si le volume de transformations grossit |
| Modélisation ML | scikit-learn / XGBoost | Réseaux de neurones (si le temps le permet) |
| Suivi d'expériences | MLflow | — |
| Restitution | Dash (Plotly) ou Power BI | Streamlit pour un prototypage rapide |
| Exposition API | FastAPI | Flask |
| Conteneurisation | Docker + Docker Compose (plugin v2) | `docker-compose` legacy |
| CI/CD | GitLab CI ou Jenkins | GitHub Actions |
| Infra as Code | Terraform + Ansible | — |
| Monitoring | Prometheus + Grafana | — |
| Cloud (si déploiement) | AWS | — |

## 3. Architecture cible

```
                 ┌──────────────────┐
API OpenSky ───▶ │   Collecteur      │
API Météo   ───▶ │  (scripts Python) │
                 └────────┬──────────┘
                          ▼
                 ┌──────────────────┐
                 │  Stockage brut    │  (PostgreSQL / fichiers)
                 └────────┬──────────┘
                          ▼
                 ┌──────────────────┐
                 │  Pipeline ETL     │  nettoyage, jointure vols+météo,
                 │                   │  construction des features
                 └────────┬──────────┘
                          ▼
                 ┌──────────────────┐
                 │  Modèle ML        │  entraînement hors-ligne,
                 │  (retard prédit)  │  modèle sérialisé (pickle/MLflow)
                 └────────┬──────────┘
                          ▼
        ┌─────────────────┴─────────────────┐
        ▼                                    ▼
┌──────────────────┐               ┌──────────────────┐
│  API FastAPI      │               │  Dashboard         │
│  /predict          │◀──────────── │  (Dash/Power BI)   │
└──────────────────┘               └──────────────────┘
```

## 4. Structure de projet proposée

```
dst-airlines/
├── README.md
├── .env.example
├── docker-compose.yml
├── requirements.txt
├── data/
│   ├── raw/                # données brutes collectées (non versionnées)
│   ├── processed/          # données nettoyées / features
│   └── output_scoring/     # résultats de src/ml/score.py (non versionnés)
├── src/
│   ├── collector/          # scripts d'appel aux API (vols, météo)
│   │   ├── flights_client.py       # OpenSky, positions temps réel
│   │   ├── weather_client.py       # Open-Meteo, prévisions
│   │   ├── bts_client.py           # vols historiques BTS + table des compagnies
│   │   ├── openflights_client.py   # référentiel aéroports (dim_airport)
│   │   ├── meteostat_client.py     # météo horaire historique (dim_weather)
│   │   └── openweather_client.py   # météo temps réel pour l'inférence
│   ├── etl/                # nettoyage, jointures, construction des features
│   │   ├── pipeline.py             # temps réel OpenSky -> flights/weather/aircraft_positions
│   │   ├── bts.py                  # transform des vols historiques BTS
│   │   ├── dimensions.py           # transform des dimensions + clés UTC
│   │   ├── warehouse.py            # load du schéma en étoile (PostgreSQL)
│   │   └── sql/schema.sql
│   ├── ml/                 # entraînement, évaluation, sérialisation du modèle
│   │   ├── dataset.py              # jeu d'entraînement SQL depuis l'entrepôt
│   │   ├── model.py                # features + modèles (gradient boosting)
│   │   ├── train.py                # sélection, optimisation, MLflow
│   │   ├── evaluate.py             # analyses et figures du rapport de modélisation
│   │   └── score.py                # scoring de 100 vols via l'API -> data/output_scoring/
│   ├── api/                # service FastAPI exposant /predict
│   │   ├── main.py
│   │   └── schemas.py
│   └── dashboard/          # app Dash de visualisation
│       ├── app.py                  # 3 onglets : carte, historique, prédiction
│       └── queries.py              # lectures SQL (agrégations côté base)
├── tests/                  # unitaires + intégration PostgreSQL (section 13)
├── docs/                   # rapport de modélisation, résultats de scoring, figures
├── notebooks/               # exploration, EDA, prototypage modèle
└── infra/
    ├── Dockerfile.api
    ├── Dockerfile.collector
    └── terraform/
```

## 5. Modèle de données (à affiner — Livrable n°4)

**Table `flights`**
| Colonne | Type | Description |
|---|---|---|
| flight_id | string (PK) | identifiant unique du vol |
| airline | string | code compagnie |
| origin_airport | string | code IATA/ICAO aéroport de départ |
| destination_airport | string | code IATA/ICAO aéroport d'arrivée |
| scheduled_departure | timestamp | horaire prévu |
| actual_departure | timestamp (nullable) | horaire réel |
| scheduled_arrival | timestamp | horaire prévu |
| actual_arrival | timestamp (nullable) | horaire réel |
| status | string | en vol / atterri / retardé / annulé |

**Table `weather`**
| Colonne | Type | Description |
|---|---|---|
| airport | string | code aéroport |
| observed_at | timestamp | horodatage de la mesure |
| temperature | float | °C |
| wind_speed | float | km/h |
| precipitation | float | mm |
| visibility | float | m |

**Table `predictions`**
| Colonne | Type | Description |
|---|---|---|
| flight_id | string (FK) | vol concerné |
| predicted_at | timestamp | horodatage de la prédiction |
| delay_probability | float | probabilité de retard > seuil |
| predicted_delay_minutes | float | estimation du retard |
| model_version | string | version du modèle utilisé |

### Entrepôt historique (schéma en étoile, `src/etl/sql/schema.sql`)

Source d'entraînement du modèle : les vols BTS donnent les horaires programmés/réels et les causes de retard, que l'API OpenSky ne fournit pas.

| Table | Source | Clé |
|---|---|---|
| `fact_flights` | BTS On-Time Performance (ZIP manuel, `data/raw/bts/`) | `fl_date, op_carrier, op_carrier_fl_num, origin, dest` |
| `dim_airport` | OpenFlights `airports.dat` | `iata_code` |
| `dim_airline` | BTS `L_UNIQUE_CARRIERS.csv` (manuel, `data/reference/`) | `iata_code` |
| `dim_aircraft` | Registre FAA `MASTER.txt` + `ACFTREF.txt` (manuel, `data/raw/faa/`) | `tail_num` |
| `dim_weather` | Meteostat, horaire, UTC | `airport_iata_code, weather_time_utc` |

`fact_flights.fl_time_utc_hour` (départ programmé converti en UTC, tronqué à l'heure) joint `dim_weather` sur `(origin, fl_time_utc_hour)`. Le chargement est idempotent (`ON CONFLICT DO NOTHING`).

## 6. Contraintes techniques par étape (référentiel officiel)

Ce tableau reprend le référentiel pédagogique tel quel — il fait foi sur les modules à mobiliser et les conditions de validation attendues à chaque étape. Le découpage en sprints de la section 9 s'y aligne.

### Étape 1 — Récolte des données

- **Description** : passer par différentes API pour récupérer des données sur les vols. Il peut être nécessaire d'aller puiser d'autres informations complémentaires, comme les codes IATA des aéroports/compagnies.
- **Objectif** : comprendre les données récupérables et faire un choix argumenté des routes API à utiliser — cette étape de compréhension est déterminante pour la suite.
- **Modules / Masterclass / Templates** : librairie `requests` ou outil Postman (pour tester les appels) ; techniques de webscraping (pour les données non disponibles via API, ex. référentiels de codes IATA).
- **Conditions de validation** : pas de livrable numéroté à ce stade — l'étape sert de base à l'étape 2.

### Étape 2 — Organisation de la donnée et Pipeline ETL

- **Description** : cette étape d'architecture découle directement de la compréhension du problème et du format des données récoltées en étape 1.
- **Objectif** : proposer un schéma d'organisation de la donnée (une ou plusieurs bases de données adaptées) et mettre en place une pipeline qui extrait, transforme et charge (ETL) la donnée dans ce nouveau système.
- **Modules / Masterclass / Templates** : module 142 - SQL, Elasticsearch, MongoDB, Neo4j, Snowflake, Big Query, DBT, Spark.
- **Conditions de validation** : **Livrable n°4** — schéma de la base de données détaillé et début de la création du pipeline.

### Étape 3 — Consommation de la donnée

- **Description** : deux volets en parallèle — un dashboard interactif représentant les vols des différents avions, et un algorithme de prédiction du retard potentiel d'un vol.
- **Objectif** : livrer un dashboard exploitable et un modèle prédictif fonctionnel ; mettre en place MLflow pour le versionning des modèles.
- **Modules / Masterclass / Templates** : modules DE120, 151, 123/124, Dash, Power BI.
- **Conditions de validation** : **Livrable n°5** — algorithme de Machine/Deep Learning fonctionnel et optimisé + Dashboard.

### Étape 4 — Déploiement

- **Description** : exposer les bases de données via une API, puis conteneuriser chaque composant du projet.
- **Objectif** : créer une API pour requêter les différentes bases de données ; faire un conteneur Docker de chaque composant (BDD, API) et un `docker-compose` fonctionnel bout en bout.
- **Modules / Masterclass / Templates** : FastAPI, Docker.
- **Conditions de validation** : rattaché au **Livrable n°6** (API + dockerisation des microservices).

### Étape 5 — Automatisation de déploiement d'infrastructure

- **Description** : déployer le code sur un environnement de production ou de test à l'aide d'outils automatisés, avec intégration dans un environnement de développement centralisé.
- **Objectif** :
  - Limiter les interruptions lors du déploiement/livraison continue (stratégie blue/green).
  - Collecter les KPI DevOps pour mesurer l'efficacité des pratiques mises en place (Deployment Cycle Time, Deployment Frequency, Deployment Success Rate).
  - Configurer des tableaux de bord de surveillance pour suivre les métriques clés et des alertes pour notifier les problèmes potentiels.
  - Définir des valeurs seuils et créer des alarmes pertinentes selon les métriques mesurées.
  - Définir et créer des actions automatiques en réponse à ces alarmes.
- **Modules / Masterclass / Templates** : Ansible, Jenkins, Terraform, Gitlab, AWS, Prometheus, Grafana.
- **Conditions de validation** : **Livrable n°7** — script d'automatisation du déploiement de l'infrastructure.

### Étape 6 — Soutenance

- **Description / Objectif** : démonstration de l'application et explication du raisonnement suivi tout au long du projet.
- **Modules / Masterclass / Templates** : aucun (X).
- **Conditions de validation** : soutenance orale + rapport.

## 7. Variables d'environnement (`.env`)

```
# API données de vols
OPENSKY_USERNAME=
OPENSKY_PASSWORD=
AVIATIONSTACK_API_KEY=          # optionnel, fallback payant

# API météo
WEATHER_API_URL=https://api.open-meteo.com/v1/forecast
OPENWEATHER_API_KEY=            # météo temps réel (inférence)

# Base de données
DATABASE_URL=postgresql://user:password@localhost:5432/dst_airlines
WAREHOUSE_DATABASE_URL=postgresql://user:password@localhost:5432/dst_airlines  # schéma en étoile (PostgreSQL ou Neon)

# Aéroports pilotes
PILOT_AIRPORTS=ATL,JFK

# API applicative
API_HOST=0.0.0.0
API_PORT=8000
API_URL=http://localhost:8000   # API appelée par src/ml/score.py

# MLflow (optionnel)
MLFLOW_TRACKING_URI=
```

## 8. Instructions de setup

### Option 1 — Setup manuel

```bash
# 1. Cloner le repo et se placer dedans
cd dst-airlines

# 2. Créer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Copier le fichier d'exemple et renseigner les clés
cp .env.example .env

# 5. Lancer les services (DB + API) en local
# Docker Compose v2 (recommandé):
docker compose up -d
# Si tu utilises l'ancienne commande legacy :
# docker-compose up -d

# 6. Lancer le collecteur (première ingestion)
python -m src.collector.flights_client

# 7. Lancer l'API en local
uvicorn src.api.main:app --reload --port 8000

# 8. Lancer le dashboard
python -m src.dashboard.app
```

### Entrepôt historique (extract → transform → load)

```bash
# Extract (~700 Mo téléchargés pour 24 mois, ~1 h)
python -m src.collector.bts_client --start 2024-01 --end 2025-12   # data/raw/bts/*.zip + table des compagnies
python -m src.collector.openflights_client                          # data/raw/airports.dat
python -m src.collector.meteostat_client --start 2024-01-01 --end 2025-12-31

# Transform
python -m src.etl.bts --delete-source               # data/processed/bts/bts_YYYY_MM.parquet (ZIPs supprimés)
python -m src.etl.dimensions all                    # data/processed/dim_*.{csv,parquet}

# Load
python -m src.etl.warehouse all                     # schéma + chargement PostgreSQL (idempotent)

# ou en une commande : just warehouse   (just warehouse 2025-01 2025-06 pour une autre période)
```

Registre des avions FAA (optionnel, ajoute l'âge et le modèle de l'avion) : registry.faa.gov refuse les téléchargements scriptés. Télécharger `ReleasableAircraft.zip` depuis un navigateur (https://registry.faa.gov/database/ReleasableAircraft.zip), le déposer dans `data/raw/faa/`, puis relancer `python -m src.etl.dimensions aircraft && python -m src.etl.warehouse aircraft`.

### Modèle de prédiction et MLflow (étape 3)

```bash
python -m src.ml.train          # ou : just train   (~2 min sur 24 mois, ~1,8 Go de RAM)
python -m src.ml.evaluate       # ou : just evaluate — figures et chiffres de docs/figures/
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # ou : just mlflow-ui
```

Démarche, résultats et limites détaillés : **[rapport de modélisation](docs/rapport_modelisation.md)**.

### Scoring d'un échantillon de vols via l'API (`src/ml/score.py`)

`score.py` vérifie le modèle **tel qu'il est servi en production** : il passe par l'API, alors que `evaluate.py` évalue le modèle chargé en mémoire. Il :

1. vérifie que l'API répond (`GET /health`) et qu'un modèle est chargé ;
2. tire au hasard **100 vols opérés** dans l'entrepôt, parmi les **3 derniers mois** (la période de test que le modèle n'a jamais vue). Le tirage est reproductible : même graine, mêmes vols ;
3. envoie chaque vol à `POST /predict` avec les seules informations connues avant le départ (vol programmé + météo à l'heure prévue). Le retard réel n'est jamais envoyé ;
4. confronte chaque prédiction au retard réellement observé, affiche le résultat sous forme de DataFrame avec un résumé, et l'enregistre dans `data/output_scoring/scoring_<version du modèle>_<horodatage>.csv`.

```bash
docker compose up -d db api      # l'API doit tourner et servir un modèle entraîné
python -m src.ml.score           # ou : just score
python -m src.ml.score --n 500 --months 1 --seed 7 --api-url http://localhost:8000
python -m src.ml.score --docs    # ou : just score-docs — publie aussi le résultat dans docs/
```

| Option | Défaut | Rôle |
|---|---|---|
| `--n` | 100 | Nombre de vols scorés |
| `--months` | 3 | Tirage parmi les N derniers mois de l'entrepôt |
| `--seed` | 42 | Graine du tirage (même graine = mêmes vols) |
| `--api-url` | `$API_URL` ou `http://localhost:8000` | API appelée |
| `--docs` | désactivé | Publie le résultat dans [`docs/resultats_scoring.md`](docs/resultats_scoring.md) (contexte, résumé, matrice de confusion, tableau des vols trié par probabilité) et copie le CSV dans `docs/scoring/` |

Colonnes du fichier produit : les informations du vol (`flight_id`, `fl_date`, `crs_dep_time`, `origin`, `dest`, `op_carrier`, `tail_num`, `distance`, météo `temp`, `prcp`, `wspd`, `pres`, `coco`…), la réponse de l'API (`delay_probability`, `predicted_delayed`, `predicted_delay_minutes`, `model_version`), la réalité (`actual_delayed` = retard ≥ 15 min, `actual_delay_minutes`) et `error` si l'appel a échoué pour ce vol. Un échec isolé n'interrompt pas le lot. Le résumé affiché donne le taux de vols prédits en retard et réellement en retard, l'exactitude, la précision, le rappel, la probabilité moyenne des vols réellement en retard et à l'heure, et l'erreur moyenne en minutes.

Exemple (modèle `20261020225807`, graine 42, publié dans [`docs/resultats_scoring.md`](docs/resultats_scoring.md)) : 100 vols scorés, 0 erreur ; 24 % prédits en retard pour 19 % réellement en retard ; exactitude 73 %, rappel 42 %, précision 33 % ; probabilité moyenne de 24 % pour les vols réellement en retard contre 16 % pour les autres. Sur 100 vols, ces chiffres fluctuent ; les résultats de référence, sur 103 583 vols, sont dans le rapport de modélisation.

Chaque appel est aussi enregistré par l'API dans sa table `predictions`, comme toute prédiction.

- **Cible** : retard au départ ≥ 15 min (classification, seuil officiel du DOT) et minutes de retard attendues (régression), sur les vols non annulés.
- **Features**, toutes connues avant le départ : aéroport de départ, destination, compagnie, heure programmée, jour de la semaine, mois, distance, durée programmée, âge de l'avion, météo à l'heure prévue (température, précipitations, vent, pression, code condition). Les heures réelles et les causes de retard sont exclues (fuite de la cible).
- **Validation** : les 3 derniers mois sont mis de côté comme jeu de test ; sélection de modèle et hyperparamètres par validation croisée **temporelle** (`TimeSeriesSplit`) sur la période d'entraînement.
- **Optimisation** : comparaison d'une référence naïve, d'une régression logistique et d'un gradient boosting (`HistGradientBoosting`) avec recherche aléatoire d'hyperparamètres ; seuil de décision choisi pour maximiser le F1.
- **Interprétabilité** : importance par permutation de chaque feature, exposée par l'API et le dashboard.
- **MLflow** : un run par entraînement (paramètres, métriques de test, importances), un run enfant par modèle candidat, classifieur et régresseur journalisés et inscrits au Model Registry (`dst-airlines-delay-classifier`, `dst-airlines-delay-regressor`). L'API et le dashboard chargent le dernier `models/delay_model_*.pkl`.

### Dashboard (étape 3)

`python -m src.dashboard.app` puis http://localhost:8050 :

| Onglet | Contenu | Source |
|---|---|---|
| Carte temps réel | Dernière position des avions autour de l'aéroport choisi (au sol / en vol, altitude, vitesse), rafraîchie chaque minute | `aircraft_positions` (OpenSky) |
| Retards historiques & avions | Indicateurs (vols, taux de retard, retard moyen, annulations), retard par heure, jour, mois et compagnie ; tableau des avions (filtrable, triable) et historique des vols de l'avion sélectionné | Entrepôt BTS |
| Prédiction | Formulaire vol + météo → probabilité de retard, verdict, minutes attendues ; métriques de test du modèle et facteurs principaux | Dernier modèle entraîné |

### Option 2 — Workflow rapide avec `just`

Le projet inclut un `Justfile` pour gérer le cycle de vie des environnements et des services Docker.

```bash
# Lister les commandes disponibles
just --list

# Initialiser le projet complet (venv + dépendances + .env + Docker)
just init

# Démarrer les services Docker
just start

# Redémarrer proprement l'application
just restart

# Arrêter les services Docker
just stop

# Voir les logs
just logs

# Rebuild complet des conteneurs
just build

# Nettoyer conteneurs et volumes
just clean

# Entrepôt, modèle, tests
just warehouse          # extract -> transform -> load (2024-01 -> 2025-12)
just train              # entraînement + MLflow
just evaluate           # figures du rapport de modélisation
just score              # scoring de 100 vols via l'API -> data/output_scoring/
just score-docs         # idem + publication dans docs/resultats_scoring.md
just mlflow-ui          # http://localhost:5000
just test               # lint + tests unitaires (comme la CI)
just test-integration   # + test d'intégration PostgreSQL
```

### Vérification rapide

```bash
# API FastAPI
http://localhost:8000/docs

# Dashboard Dash
http://localhost:8050

# Prometheus
http://localhost:9090

# Grafana
http://localhost:3000
```

## 9. Roadmap de développement (sprints)

| Sprint | Objectif | Livrable |
|---|---|---|
| S1 | Connecteur API vols + météo fonctionnel | Data brute collectée pour 2-3 aéroports |
| S2 | Schéma BDD + pipeline ETL | Livrable n°4 |
| S3-S4 | Modèle ML de prédiction + dashboard | Livrable n°5 |
| S5 | API FastAPI + dockerisation | Livrable n°6 |
| S6 | Automatisation du déploiement (CI/CD) | Livrable n°7 |
| S7 | Soutenance | Rapport final + démo |

## 10. Instructions pour Claude Code

Quand tu (Claude Code) interviens sur ce repo :

1. **Toujours commencer par lire ce README** et les 3 docs de cadrage cités en section 1 avant de générer du code — ils contiennent les décisions déjà prises (périmètre, priorités MoSCoW, risques identifiés).
2. **Respecter la structure de dossiers** de la section 4 ; ne pas créer de nouveaux dossiers à la racine sans raison.
3. **Un module = une responsabilité** : le collecteur ne fait que collecter, l'ETL ne fait que transformer, le modèle ne fait qu'entraîner/prédire, l'API ne fait qu'exposer.
4. **Tests obligatoires** pour toute nouvelle fonction dans `src/etl/`, `src/ml/`, `src/api/` — ajouter le test correspondant dans `tests/` dans le même commit.
5. **Ne jamais committer** `.env`, les données brutes (`data/raw/`), ni les modèles sérialisés lourds — les lister dans `.gitignore`.
6. **Suivre le découpage en sprints** (section 9) : proposer les tâches dans l'ordre S1 → S6 sauf demande explicite contraire, en respectant les conditions de validation officielles de chaque étape (section 6).
7. **Documenter les choix non triviaux** (ex. pourquoi tel algorithme ML) en commentaire ou dans `notebooks/`, pas seulement dans le code.
8. **Interprétabilité du modèle** : toute fonctionnalité de prédiction doit pouvoir exposer les facteurs principaux (feature importance / SHAP), conformément à l'exigence éthique du MVP.
9. **Rate limiting / clés API** : toujours prévoir une gestion d'erreur propre en cas de quota dépassé sur les API externes (OpenSky, météo) plutôt qu'un crash silencieux.

## 11. Ressources externes utiles

- OpenSky Network — documentation API : https://openskynetwork.github.io/opensky-api/
- Open-Meteo — documentation API : https://open-meteo.com/en/docs
- FastAPI — documentation : https://fastapi.tiangolo.com/
- Dash (Plotly) — documentation : https://dash.plotly.com/
- MLflow — documentation : https://mlflow.org/docs/latest/index.html

## 12. Évaluation de conformité au cahier des charges

*État au 21/10/2026. Évaluation faite à partir du code, des tests (77 tests dont 1 d'intégration PostgreSQL), de l'entrepôt chargé avec les vols BTS réels 2024-2025, d'un entraînement complet du modèle et de la stack Docker relancée (API, dashboard, collecteur, Prometheus). Étapes 1, 2, 4 et 5 : évaluation du 20/10/2026, mise à jour seulement là où l'étape 3 les a fait évoluer.*

### Synthèse

| Étape | Livrable | Statut | En une phrase |
|---|---|---|---|
| 1. Récolte des données | — | ✅ Largement conforme | 6 sources API/fichiers collectées via `requests`, BTS automatisé, référentiels IATA en place ; ni webscraping ni Postman. |
| 2. Organisation des données + ETL | n°4 | ✅ Conforme | Schéma en étoile chargé avec 884 451 vols réels (ATL/JFK, 2024-2025) et 35 088 relevés météo ; deux modèles de données coexistent (temps réel / historique). |
| 3. Consommation (dashboard + ML + MLflow) | n°5 | ✅ Conforme | Modèle entraîné sur les vols réels, comparé et optimisé en validation temporelle, versionné dans le Model Registry MLflow ; dashboard à 3 onglets dont carte et vue par avion. Voir le [rapport de modélisation](docs/rapport_modelisation.md). |
| 4. Déploiement (API + Docker) | n°6 | 🟡 Partiel | Conteneurs et `docker-compose` en place, mais l'API ne fait que prédire : elle ne permet pas de requêter les bases. |
| 5. Automatisation de l'infrastructure | n°7 | ❌ Non conforme | Seule la CI (lint + tests) existe ; pas de CD, de blue/green, de KPI DevOps, d'alerting ni d'IaC appliquée. |
| 6. Soutenance | — | ⏳ À venir | — |

### Étape 1 — Récolte des données : ✅ largement conforme

**En place**
- Vols temps réel : API OpenSky `/states/all` ([flights_client.py](src/collector/flights_client.py)), avec retries et gestion du quota (HTTP 429).
- Vols historiques : BTS On-Time Performance, téléchargé automatiquement mois par mois ([bts_client.py](src/collector/bts_client.py)) : la seule source gratuite qui contient les horaires réels et les retards observés.
- Météo : Open-Meteo (prévisions), Meteostat (historique horaire), OpenWeather (temps réel pour l'inférence).
- Référentiels : codes IATA/ICAO des aéroports (OpenFlights + [airports.json](data/reference/airports.json)), codes compagnies (BTS `L_UNIQUE_CARRIERS`), avions (registre FAA).
- Les choix de routes sont argumentés dans les docstrings. Par exemple, OpenSky gratuit ne fournit pas d'horaires, d'où le recours à BTS pour l'entraînement.

**Écarts**
- Aucune technique de webscraping : les référentiels viennent de fichiers ou d'API.
- Pas de collection Postman pour documenter et tester les appels.
- Le registre FAA des avions reste un téléchargement manuel : registry.faa.gov refuse les clients scriptés. Il est optionnel (sans lui, l'âge et le modèle de l'avion sont absents).

### Étape 2 — Organisation des données et pipeline ETL (Livrable n°4) : ✅ conforme, avec réserves

**En place**
- Schéma en étoile PostgreSQL détaillé : `fact_flights` + `dim_airport`, `dim_airline`, `dim_aircraft`, `dim_weather` ([schema.sql](src/etl/sql/schema.sql), section 5), avec clés primaires, clés étrangères, index et une clé de jointure vol ↔ météo en UTC.
- Pipeline ETL complet, découpé en extract ([src/collector/](src/collector/)), transform ([bts.py](src/etl/bts.py), [dimensions.py](src/etl/dimensions.py)) et load ([warehouse.py](src/etl/warehouse.py)). Le chargement est idempotent et vérifié de bout en bout sur PostgreSQL.
- Le pipeline temps réel OpenSky → `flights`/`weather` est opérationnel ([pipeline.py](src/etl/pipeline.py)).
- Toutes les fonctions ETL sont couvertes par des tests.

**Réserves**
- **Deux modèles de données coexistent** : le modèle temps réel `flights`/`weather`/`predictions` (SQLAlchemy) et l'entrepôt en étoile. Leur articulation n'est pas encore documentée, et la section 5 présente encore surtout le premier.
- Le registre FAA doit être déposé manuellement pour alimenter `dim_aircraft` (fait : 5 261 avions, 97,8 % des vols rattachés à leur avion).
- Pas d'orchestration (cron, Airflow) : les étapes se lancent à la main ou via `just warehouse`.
- Une seule technologie de stockage (SQL) : c'est acceptable (« 1 ou plusieurs bases »), mais aucune des autres briques proposées (MongoDB, Elasticsearch, Spark, DBT…) n'est utilisée.

### Étape 3 — Consommation des données (Livrable n°5) : ✅ conforme

Détail de la démarche, des résultats et des limites : **[docs/rapport_modelisation.md](docs/rapport_modelisation.md)**.

**Algorithme de prédiction ([src/ml/](src/ml/))**
- Entraîné sur les **872 096 vols réels opérés** au départ d'ATL et JFK (2024-2025), joints à la météo horaire observée à l'heure de départ prévue ([dataset.py](src/ml/dataset.py)). Le jeu synthétique a été supprimé.
- Une seule fonction de features pour l'entraînement, l'API et le dashboard : pas d'écart entre entraînement et production ; uniquement des informations connues avant le départ (pas de fuite de la cible).
- **Optimisé** : référence naïve, régression logistique et gradient boosting comparés en validation croisée temporelle, recherche aléatoire d'hyperparamètres, seuil de décision choisi sur une période distincte, plafonnement des minutes prédites.
- Résultats sur les 3 derniers mois (103 583 vols jamais vus) : ROC AUC 0,68 ; PR AUC 0,37 contre 0,22 pour le hasard ; erreur moyenne sur le retard de 19,1 min contre 22,8 min pour la référence. L'ajout du registre FAA (âge de l'avion) ne change pas ces résultats : l'effet de l'avion est déjà porté par la compagnie (section 2.4 du rapport).
- Interprétabilité : importance par permutation, exposée par l'API et le dashboard.

**MLflow** : un run par entraînement (paramètres, métriques, importances), un run enfant par candidat, classifieur et régresseur inscrits au Model Registry (`dst-airlines-delay-classifier`, `dst-airlines-delay-regressor`).

**Dashboard ([src/dashboard/](src/dashboard/))** : carte temps réel des avions autour des aéroports, retards historiques filtrables (heure, jour, mois, compagnie), tableau par avion et historique des vols de l'avion sélectionné, simulateur de prédiction.

**Limites restantes**
- Performance modeste mais réaliste : sans information sur les retards en cascade (avion arrivant en retard) ni sur la congestion, le modèle n'explique qu'une partie des retards. Voir les pistes du rapport.
- Pas de tableau de bord Power BI (Dash seul, ce qui est autorisé par le référentiel).
- MLflow tourne en local (`mlflow ui`), pas dans le `docker-compose`.

### Étape 4 — Déploiement (Livrable n°6) : 🟡 partiel

**En place**
- API FastAPI ([main.py](src/api/main.py)) : `/predict` (prédiction + facteurs explicatifs, prédiction sauvegardée en base), `/health`, `/metrics` (Prometheus).
- Un Dockerfile par composant applicatif (API, dashboard, collecteur) et un `docker-compose.yml` valide avec 6 services : PostgreSQL, API, dashboard, collecteur, Prometheus, Grafana.

**Écarts**
- **L'API ne permet pas de « requêter les différentes bases de données »** : aucun endpoint pour lire les vols, la météo, les aéroports, les compagnies ou l'historique des prédictions.
- L'entrepôt (`src/etl/warehouse.py`) et MLflow ne sont pas dans le `docker-compose`.
- Le conteneur collecteur s'exécute une fois puis s'arrête (pas de planification).
- Images `prom/prometheus:latest` et `grafana/grafana:latest` non épinglées.

### Étape 5 — Automatisation du déploiement (Livrable n°7) : ❌ non conforme

**En place**
- Code centralisé sur GitHub, avec une CI GitHub Actions ([ci.yml](.github/workflows/ci.yml)) qui lance ruff + 77 tests (dont l'intégration sur un service PostgreSQL) à chaque push et pull request.
- Prometheus collecte les métriques de l'API ; un conteneur Grafana est présent.
- Squelette Terraform AWS ([main.tf](infra/terraform/main.tf)) : provider et variables seulement, **aucune ressource**, jamais appliqué.

**Manquants**
- Aucun déploiement automatisé (CD) vers un environnement de test ou de production, et aucune stratégie blue/green.
- Aucun KPI DevOps collecté (Deployment Cycle Time, Deployment Frequency, Deployment Success Rate).
- Grafana sans source de données ni tableau de bord provisionné ; aucune règle d'alerte, pas d'Alertmanager, pas de seuils définis.
- Aucune action automatique en réponse aux alertes.
- Pas d'Ansible ; Jenkins et GitLab sont remplacés par GitHub Actions (choix défendable, à justifier en soutenance).
- **Le livrable n°7 (script d'automatisation du déploiement) n'existe pas encore.**

### Plan d'action priorisé

| Priorité | Action | Débloque |
|---|---|---|
| 1 | API : endpoints de lecture (`/flights`, `/airports`, `/airlines`, `/weather`, `/predictions`) ; ajouter MLflow au `docker-compose` | Livrable n°6 |
| 2 | Terraform : ressources AWS réelles (ECR, ECS, RDS) ; pipeline CD GitHub Actions avec déploiement blue/green | Livrable n°7 |
| 3 | Grafana provisionné (datasource, dashboards, alertes, seuils) ; KPI DevOps calculés depuis l'historique des déploiements ; actions automatiques sur alerte | Livrable n°7 |
| 4 | Modèle : features de retard en cascade et de congestion (cf. rapport de modélisation), rotation de l'avion dans la journée | Amélioration du livrable n°5 |
| 5 | Planifier la collecte (cron ou Airflow) et le réentraînement | Étapes 2-3 en continu |

## 13. Tests d'exécution

Deux niveaux de vérification : les **tests automatisés** (rapides, sans réseau, lancés aussi par la CI), puis une **recette d'exécution de bout en bout** qui fait tourner chaque brique sur de vraies données.

### 13.1 Tests automatisés

```bash
source .venv/bin/activate

# Lint + tests unitaires (identique à la CI GitHub Actions)
ruff check src tests && pytest -q                       # ou : just test

# + test d'intégration sur un vrai PostgreSQL (base docker-compose démarrée)
docker compose up -d db
WAREHOUSE_TEST_DATABASE_URL=postgresql://dst:dst@localhost:5432/dst_airlines pytest -q   # ou : just test-integration

# Un seul fichier, un seul test, en mode verbeux
pytest tests/test_model.py -v
pytest tests/test_dashboard.py -k aircraft -v
```

Résultat attendu : `ruff` affiche `All checks passed!` et `pytest` n'affiche aucun échec. Sans `WAREHOUSE_TEST_DATABASE_URL`, le test d'intégration est signalé `skipped` (normal). Le test d'intégration travaille dans un schéma PostgreSQL temporaire (`it_xxxxxxxx`) supprimé à la fin : il ne modifie pas les tables de la base.

| Fichier | Ce qui est vérifié |
|---|---|
| `test_collector.py` | Clients API (OpenSky, Open-Meteo, OpenWeather, OpenFlights, Meteostat, BTS) : parsing, sauvegarde brute, reprise sur quota 429 et erreurs réseau sans crash (réseau simulé) |
| `test_etl.py` | Pipeline temps réel : vols, météo, positions des avions, jointure, écriture idempotente en base |
| `test_bts.py` | Transformation BTS : formats d'export et mensuel, filtrage ATL/JFK, typage, dédoublonnage, calcul du retard au passage de minuit |
| `test_dimensions.py` | Dimensions aéroports, compagnies, avions (FAA), météo ; conversion de l'heure locale en UTC (heure d'été/hiver) |
| `test_warehouse.py` | Chargement : URL de connexion, conversion des valeurs manquantes, requêtes idempotentes, schéma rejouable |
| `test_model.py` | Features, cibles, découpage temporel, seuil de décision, plafonnement des minutes, rejet d'un modèle d'ancien format, entraînement complet avec MLflow (le modèle doit battre la référence naïve) |
| `test_evaluate.py` | Génération des figures et des chiffres du rapport de modélisation |
| `test_score.py` | Scoring via l'API (HTTP simulé) : format des requêtes, cible jamais envoyée, comparaison au réel, erreurs isolées, API injoignable, fichier CSV |
| `test_api.py` | `/health`, `/predict` (avec et sans modèle), validation des entrées (HTTP 422) |
| `test_dashboard.py` | Callbacks des 3 onglets (carte, historique, avion sélectionné, prédiction) et messages en cas de source indisponible |
| `test_integration_warehouse.py` | Chaîne réelle sur PostgreSQL : ZIP BTS → transformation → chargement ×2 (sans doublon) → jeu d'entraînement → requêtes du dashboard |

### 13.2 Recette d'exécution de bout en bout

À dérouler dans l'ordre depuis la racine du projet, venv activé, `.env` renseigné (`WAREHOUSE_DATABASE_URL`, `PILOT_AIRPORTS=ATL,JFK`).

**0. Base de données**

```bash
docker compose up -d db
docker compose exec db pg_isready -U dst -d dst_airlines
```
Attendu : `accepting connections`.

**1. Collecte temps réel (étape 1) et pipeline temps réel (étape 2)**

```bash
python -m src.collector.flights_client
python -m src.collector.weather_client
python -m src.etl.pipeline
ls data/raw/ | tail -4
```
Attendu : des fichiers `flights_ATL_*.json`, `flights_JFK_*.json`, `weather_*_*.json` horodatés, et un `data/processed/flights_weather_*.csv`. En cas de quota OpenSky dépassé, un avertissement est logué et le script se termine sans erreur.

**2. Entrepôt historique (étape 2)**

```bash
just warehouse        # ou les commandes de la section 8, ~1 h pour 24 mois
```
Contrôle :
```bash
docker compose exec db psql -U dst -d dst_airlines -c "
  SELECT 'fact_flights' AS t, COUNT(*) FROM fact_flights UNION ALL
  SELECT 'dim_airport', COUNT(*) FROM dim_airport UNION ALL
  SELECT 'dim_airline', COUNT(*) FROM dim_airline UNION ALL
  SELECT 'dim_weather', COUNT(*) FROM dim_weather UNION ALL
  SELECT 'vols sans météo', COUNT(*) FROM fact_flights f
    LEFT JOIN dim_weather w ON w.airport_iata_code = f.origin AND w.weather_time_utc = f.fl_time_utc_hour
    WHERE w.temp IS NULL;"
```
Attendu (2024-2025) : 884 451 vols, 2 aéroports, 1 778 compagnies, 35 088 relevés météo horaires, et environ 200 vols seulement sans météo (trous de relevé des stations). Relancer `python -m src.etl.warehouse all` ne doit **pas** changer ces nombres (chargement idempotent).

**3. Entraînement et MLflow (étape 3)**

```bash
python -m src.ml.train
python -m src.ml.evaluate
ls models/ docs/figures/
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Attendu : les logs affichent le score de validation croisée de chaque candidat, puis les métriques de test ; un nouveau `models/delay_model_<version>.pkl` est créé, et `src.ml.evaluate` régénère les 7 figures et `evaluation.json` de `docs/figures/` (chiffres cités dans le [rapport de modélisation](docs/rapport_modelisation.md)). Dans l'interface MLflow (http://localhost:5000) : expérience `dst-airlines-delay-prediction`, un run avec 3 runs enfants (`candidate-*`), et les modèles `dst-airlines-delay-classifier` / `dst-airlines-delay-regressor` dans l'onglet **Models**. Le PR AUC de test doit être nettement supérieur au taux de base (`test_base_rate`) et la MAE inférieure à celle de la référence (`baseline_mae_minutes`).

**4. API (étapes 3-4)**

```bash
uvicorn src.api.main:app --port 8000      # ou via docker compose (service api)
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "flight_id": "DL1234_20260115", "fl_date": "2026-01-15", "crs_dep_time": 1745,
  "origin": "ATL", "dest": "LAX", "op_carrier": "DL", "distance": 1946,
  "temp": 2.0, "prcp": 4.0, "wspd": 35.0, "pres": 1004}'
curl -s localhost:8000/metrics | grep http_requests_total | head -3
python -m src.ml.score
```
Attendu : `/health` renvoie la version du modèle chargé ; `/predict` renvoie `delay_probability`, `predicted_delayed`, `predicted_delay_minutes` et les facteurs principaux ; une heure invalide (`"crs_dep_time": 2500`) renvoie HTTP 422 ; `/metrics` expose les compteurs Prometheus ; `src.ml.score` affiche le tableau des 100 vols scorés avec `errors: 0` dans le résumé et écrit un CSV dans `data/output_scoring/`. L'API charge le modèle au démarrage : la redémarrer après un nouvel entraînement (`docker compose restart api`).

**5. Dashboard (étape 3)**

```bash
python -m src.dashboard.app               # ou via docker compose (service dashboard)
```
Ouvrir http://localhost:8050 et vérifier :
- **Carte temps réel** : des points autour d'ATL, puis de JFK après changement dans la liste ; l'heure de dernière observation correspond à l'étape 1.
- **Retards historiques & avions** : les indicateurs et les 4 graphiques changent quand on filtre sur un aéroport ou une compagnie ; cliquer une ligne du tableau des avions affiche le nuage de ses vols (retard par date, seuil de 15 min en pointillé).
- **Prédiction** : les métriques du modèle s'affichent ; augmenter la pluie et le vent ou décaler l'heure vers le soir doit augmenter la probabilité de retard.

**6. Stack Docker complète (étape 4)**

```bash
docker compose up -d --build
docker compose ps
```
Attendu : `db`, `api`, `dashboard`, `prometheus`, `grafana` à l'état `running` (le `collector` s'exécute une fois puis s'arrête) ; http://localhost:8000/docs, http://localhost:8050 et http://localhost:9090/targets (cible `dst-airlines-api` à l'état UP) répondent.
