set shell := ["bash", "-cu"]

default:
    @just --list

help:
    @just --list

init:
    @echo "==> Création de l'environnement virtuel"
    python3 -m venv .venv
    . .venv/bin/activate && pip install -r requirements.txt
    @if [ -f .env.example ] && [ ! -f .env ]; then cp .env.example .env; fi
    @echo "==> Démarrage des services Docker"
    docker compose up -d

start:
    @echo "==> Démarrage des services"
    docker compose up -d

restart:
    @echo "==> Redémarrage des services"
    docker compose down --remove-orphans
    docker compose up -d

stop:
    @echo "==> Arrêt des services"
    docker compose down --remove-orphans

logs:
    docker compose logs -f

build:
    @echo "==> Build des conteneurs"
    docker compose build

clean:
    @echo "==> Nettoyage complet des conteneurs et volumes"
    docker compose down -v --remove-orphans

# Entrepôt historique (BTS + dimensions) : collecte -> transformation -> chargement.
# Seul prérequis manuel (optionnel) : ReleasableAircraft.zip de la FAA dans data/raw/faa/.

warehouse-extract start="2024-01" end="2025-12":
    @echo "==> Collecte : vols BTS, compagnies, aéroports, météo historique"
    . .venv/bin/activate && python -m src.collector.bts_client --start {{start}} --end {{end}}
    . .venv/bin/activate && python -m src.collector.openflights_client
    . .venv/bin/activate && python -m src.collector.meteostat_client --start {{start}}-01 --end "$(date -d '{{end}}-01 +1 month -1 day' +%F)"

warehouse-transform:
    @echo "==> Nettoyage BTS (ZIPs supprimés après traitement) et construction des dimensions"
    . .venv/bin/activate && python -m src.etl.bts --delete-source && python -m src.etl.dimensions all

warehouse-load:
    @echo "==> Chargement dans PostgreSQL (WAREHOUSE_DATABASE_URL)"
    . .venv/bin/activate && python -m src.etl.warehouse all

warehouse start="2024-01" end="2025-12": (warehouse-extract start end) warehouse-transform warehouse-load

# Entraînement du modèle sur l'entrepôt, suivi dans MLflow
train:
    . .venv/bin/activate && python -m src.ml.train

# Figures et chiffres du rapport de modélisation (docs/figures/)
evaluate:
    . .venv/bin/activate && python -m src.ml.evaluate

# Scoring de 100 vols via l'API (l'API doit tourner) -> data/output_scoring/
score n="100":
    . .venv/bin/activate && python -m src.ml.score --n {{n}}

# Idem, et publication du résultat dans docs/resultats_scoring.md (+ CSV dans docs/scoring/)
score-docs n="100":
    . .venv/bin/activate && python -m src.ml.score --n {{n}} --docs

# Interface MLflow (expériences, runs, Model Registry) : http://localhost:5000
mlflow-ui:
    . .venv/bin/activate && mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Tests unitaires + lint (identiques à la CI)
test:
    . .venv/bin/activate && ruff check src tests && pytest -q

# Tests unitaires + intégration PostgreSQL (schéma temporaire, la base n'est pas modifiée)
test-integration:
    . .venv/bin/activate && WAREHOUSE_TEST_DATABASE_URL=postgresql://dst:dst@localhost:5432/dst_airlines pytest -q

venv:
    @echo "==> Activation de l'environnement virtuel"
    @echo "source .venv/bin/activate"
