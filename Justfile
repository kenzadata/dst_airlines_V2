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
# Prérequis manuels : ZIPs BTS dans data/raw/bts/, MASTER.txt + ACFTREF.txt FAA dans
# data/raw/faa/, L_UNIQUE_CARRIERS.csv dans data/reference/.

warehouse-extract:
    @echo "==> Collecte des référentiels et de la météo historique"
    . .venv/bin/activate && python -m src.collector.openflights_client && python -m src.collector.meteostat_client

warehouse-transform:
    @echo "==> Nettoyage BTS et construction des dimensions"
    . .venv/bin/activate && python -m src.etl.bts && python -m src.etl.dimensions all

warehouse-load:
    @echo "==> Chargement dans PostgreSQL (WAREHOUSE_DATABASE_URL)"
    . .venv/bin/activate && python -m src.etl.warehouse all

warehouse: warehouse-extract warehouse-transform warehouse-load

venv:
    @echo "==> Activation de l'environnement virtuel"
    @echo "source .venv/bin/activate"
