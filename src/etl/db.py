"""Schéma de base de données (README section 5) et helpers de connexion.

Un seul schéma sert à la fois SQLite (dev local, valeur par défaut) et
PostgreSQL (via docker-compose) : le choix se fait uniquement via
`DATABASE_URL`, aucune logique spécifique à un moteur n'est nécessaire ici.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, String, create_engine
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


def get_engine(database_url: str | None = None):
    url = database_url or os.environ.get(
        "DATABASE_URL", "sqlite:///./data/processed/dst_airlines.db"
    )
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def init_db(engine=None) -> None:
    engine = engine or get_engine()
    Base.metadata.create_all(engine)


def get_session_factory(engine=None) -> sessionmaker[Session]:
    engine = engine or get_engine()
    return sessionmaker(bind=engine)
