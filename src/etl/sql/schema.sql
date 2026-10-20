-- DST Airlines — schéma en étoile de l'entrepôt historique (PostgreSQL)
-- Ordre : dimensions d'abord, puis fact_flights (FK vers les dimensions).
-- Idempotent : peut être rejoué sans perte de données (CREATE ... IF NOT EXISTS).
-- =====================================================================

-- ---------------------------------------------------------------------
-- DIM_AIRPORT  (source : OpenFlights airports.dat, filtré sur les aéroports pilotes)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_airport (
    iata_code  CHAR(3)           PRIMARY KEY,
    name       TEXT              NOT NULL,
    city       TEXT,
    latitude   DOUBLE PRECISION,
    longitude  DOUBLE PRECISION,
    timezone   TEXT
);

-- ---------------------------------------------------------------------
-- DIM_AIRLINE  (source : BTS Carrier Decode / L_UNIQUE_CARRIERS)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_airline (
    iata_code    VARCHAR(10) PRIMARY KEY,
    airline_name TEXT NOT NULL
);

-- ---------------------------------------------------------------------
-- DIM_AIRCRAFT (source : FAA Aircraft Registry, MASTER + ACFTREF)
-- clé = tail number normalisé (préfixe N conservé, comme BTS TAIL_NUM)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_aircraft (
    tail_num      VARCHAR(10) PRIMARY KEY,
    manufacturer  TEXT,
    model         TEXT,
    year_mfr      SMALLINT,
    aircraft_age  SMALLINT
);

-- ---------------------------------------------------------------------
-- DIM_WEATHER (source : Meteostat, météo horaire historique)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_weather (
    airport_iata_code CHAR(3)      NOT NULL,
    weather_time_utc  TIMESTAMPTZ  NOT NULL,
    temp              DOUBLE PRECISION,
    prcp              DOUBLE PRECISION,
    wspd              DOUBLE PRECISION,
    pres              DOUBLE PRECISION,
    coco              SMALLINT,
    PRIMARY KEY (airport_iata_code, weather_time_utc)
);

-- ---------------------------------------------------------------------
-- FACT_FLIGHTS  (source : BTS Reporting Carrier On-Time Performance)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_flights (
      fl_date               DATE         NOT NULL
    , op_unique_carrier     VARCHAR(7)   NOT NULL
    , op_carrier            VARCHAR(7)   NOT NULL
    , tail_num              VARCHAR(10)
    , op_carrier_fl_num     INTEGER      NOT NULL
    , origin_airport_id     INTEGER      NOT NULL
    , origin                CHAR(3)      NOT NULL
    , dest_airport_id       INTEGER      NOT NULL
    , dest                  CHAR(3)      NOT NULL
    , crs_dep_time          SMALLINT
    , dep_time              SMALLINT
    , arr_time              SMALLINT
    , cancelled             SMALLINT
    , diverted              SMALLINT
    , crs_elapsed_time      REAL
    , actual_elapsed_time   REAL
    , air_time              REAL
    , flights               INTEGER
    , distance              REAL
    , carrier_delay         REAL
    , weather_delay         REAL
    , nas_delay             REAL
    , security_delay        REAL
    , late_aircraft_delay   REAL
    , fl_time_utc           TIMESTAMPTZ
    , fl_time_utc_hour      TIMESTAMPTZ
    , CONSTRAINT pk_fact_flights PRIMARY KEY (fl_date, op_carrier, op_carrier_fl_num, origin, dest)
    , CONSTRAINT fk_origin  FOREIGN KEY (origin)     REFERENCES dim_airport (iata_code)
    , CONSTRAINT fk_carrier FOREIGN KEY (op_carrier) REFERENCES dim_airline (iata_code)
    -- Pas de FK sur dest (seuls les aéroports pilotes sont dans dim_airport),
    -- tail_num (avions non-US absents du registre FAA) ni sur la météo
    -- (heures sans relevé) : ces jointures se font en LEFT JOIN.
);

CREATE INDEX IF NOT EXISTS idx_fact_flights_fl_date  ON fact_flights (fl_date);
CREATE INDEX IF NOT EXISTS idx_fact_flights_origin   ON fact_flights (origin);
CREATE INDEX IF NOT EXISTS idx_fact_flights_dest     ON fact_flights (dest);
CREATE INDEX IF NOT EXISTS idx_fact_flights_carrier  ON fact_flights (op_carrier);
CREATE INDEX IF NOT EXISTS idx_fact_flights_tail_num ON fact_flights (tail_num);
CREATE INDEX IF NOT EXISTS idx_fact_flights_weather  ON fact_flights (origin, fl_time_utc_hour);
