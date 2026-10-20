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
│   └── processed/          # données nettoyées / features
├── src/
│   ├── collector/          # scripts d'appel aux API (vols, météo)
│   │   ├── flights_client.py       # OpenSky, positions temps réel
│   │   ├── weather_client.py       # Open-Meteo, prévisions
│   │   ├── openflights_client.py   # référentiel aéroports (dim_airport)
│   │   ├── meteostat_client.py     # météo horaire historique (dim_weather)
│   │   └── openweather_client.py   # météo temps réel pour l'inférence
│   ├── etl/                # nettoyage, jointures, construction des features
│   │   ├── pipeline.py             # temps réel OpenSky -> flights/weather
│   │   ├── bts.py                  # transform des vols historiques BTS
│   │   ├── dimensions.py           # transform des dimensions + clés UTC
│   │   ├── warehouse.py            # load du schéma en étoile (PostgreSQL)
│   │   └── sql/schema.sql
│   ├── ml/                 # entraînement, évaluation, sérialisation du modèle
│   │   ├── train.py
│   │   └── model.py
│   ├── api/                # service FastAPI exposant /predict
│   │   ├── main.py
│   │   └── schemas.py
│   └── dashboard/          # app Dash de visualisation
│       └── app.py
├── tests/
│   ├── test_etl.py
│   ├── test_api.py
│   └── test_model.py
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
# Prérequis manuels : ZIPs BTS dans data/raw/bts/, MASTER.txt + ACFTREF.txt (FAA)
# dans data/raw/faa/, L_UNIQUE_CARRIERS.csv (BTS) dans data/reference/.
python -m src.collector.openflights_client          # data/raw/airports.dat
python -m src.collector.meteostat_client            # data/raw/weather_history_*.parquet
python -m src.etl.bts                               # data/processed/bts/bts_YYYY_MM.parquet
python -m src.etl.dimensions all                    # data/processed/dim_*.{csv,parquet}
python -m src.etl.warehouse all                     # schéma + chargement PostgreSQL
# ou en une commande : just warehouse
```

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
