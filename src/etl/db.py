"""Schéma de base de données (README section 5) et helpers de connexion.

Un seul schéma sert à la fois SQLite (dev local, valeur par défaut) et
PostgreSQL (via docker-compose) : le choix se fait uniquement via
`DATABASE_URL`, aucune logique spécifique à un moteur n'est nécessaire ici.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, DateTime, Float, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()


class Base(DeclarativeBase):
    pass


class Flight(Base):
    __tablename__ = "flights"

    flight_id = Column(String, primary_key=True)
    airline = Column(String, nullable=True)
    origin_airport = Column(String, nullable=False)
    destination_airport = Column(String, nullable=True)
    scheduled_departure = Column(DateTime, nullable=True)
    actual_departure = Column(DateTime, nullable=True)
    scheduled_arrival = Column(DateTime, nullable=True)
    actual_arrival = Column(DateTime, nullable=True)
    status = Column(String, nullable=True)


class AircraftPosition(Base):
    """Position d'un aéronef à un instant (OpenSky), pour la carte temps réel."""

    __tablename__ = "aircraft_positions"

    id = Column(String, primary_key=True)  # icao24 + horodatage du dernier contact
    icao24 = Column(String, nullable=False, index=True)
    callsign = Column(String, nullable=True)
    airport = Column(String, nullable=False)
    observed_at = Column(DateTime, nullable=False, index=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    altitude_m = Column(Float, nullable=True)
    velocity_ms = Column(Float, nullable=True)
    heading = Column(Float, nullable=True)
    on_ground = Column(Boolean, nullable=True)


class Weather(Base):
    __tablename__ = "weather"

    id = Column(String, primary_key=True)
    airport = Column(String, nullable=False)
    observed_at = Column(DateTime, nullable=False)
    temperature = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)
    precipitation = Column(Float, nullable=True)
    visibility = Column(Float, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True)
    flight_id = Column(String, nullable=False)
    predicted_at = Column(DateTime, nullable=False)
    delay_probability = Column(Float, nullable=False)
    predicted_delay_minutes = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)


def sqlalchemy_url(url: str) -> str:
    """Rend le pilote PostgreSQL explicite (`postgresql+psycopg2://`).

    Depuis SQLAlchemy 2.1, `postgresql://` désigne psycopg 3, que le projet
    n'installe pas : sans pilote explicite, l'API plante dans les conteneurs.
    """
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg2://" + url.removeprefix(prefix)
    return url


def get_engine(database_url: str | None = None):
    url = database_url or os.environ.get(
        "DATABASE_URL", "sqlite:///./data/processed/dst_airlines.db"
    )
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(sqlalchemy_url(url), connect_args=connect_args)


def init_db(engine=None) -> None:
    engine = engine or get_engine()
    Base.metadata.create_all(engine)


def get_session_factory(engine=None) -> sessionmaker[Session]:
    engine = engine or get_engine()
    return sessionmaker(bind=engine)
