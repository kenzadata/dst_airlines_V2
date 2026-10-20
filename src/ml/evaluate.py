"""Évaluation détaillée du dernier modèle entraîné, pour le rapport de modélisation.

Recharge le modèle de `models/`, rejoue le même découpage temporel que
l'entraînement, calcule les analyses (courbes, calibration, performance par
aéroport et par mois, erreurs du régresseur) et produit les figures et un
fichier de chiffres dans `docs/figures/`. Le rapport
(`docs/rapport_modelisation.md`) cite ces chiffres.

Usage :
    python -m src.ml.evaluate [--test-months 3]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.ml.model import DelayModel, build_features, build_targets, latest_model_path
from src.ml.train import split_train_test

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parents[2] / "docs" / "figures"

# Palette de référence (skill dataviz) : surface claire, encres, 3 premiers slots catégoriels
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FEATURE_LABELS = {
    "dep_hour": "Heure de départ", "month": "Mois", "op_carrier": "Compagnie", "dest": "Destination",
    "day_of_week": "Jour de la semaine", "coco": "Condition météo", "temp": "Température",
    "pres": "Pression", "prcp": "Précipitations", "wspd": "Vent", "origin": "Aéroport de départ",
    "distance": "Distance", "crs_elapsed_time": "Durée programmée", "aircraft_age": "Âge de l'avion",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "font.size": 10,
        "legend.frameon": False, "lines.linewidth": 2,
    })


def _save(fig, name: str, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / name, dpi=150)
    plt.close(fig)
    return name


def _fr(value: float, digits: int = 2) -> str:
    """Nombre au format français (virgule décimale) pour les légendes."""
    return f"{value:.{digits}f}".replace(".", ",")


def _percent_axis(ax, axis: str = "y") -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def delay_rate_profiles(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Taux de retard observé par heure prévue et par mois, pour chaque aéroport."""
    delayed, _ = build_targets(df)
    data = df.assign(delayed=delayed, dep_hour=(pd.to_numeric(df["crs_dep_time"]) % 2400) // 100,
                     month=pd.to_datetime(df["fl_date"]).dt.to_period("M").dt.to_timestamp())
    return {
        "hour": data.groupby(["dep_hour", "origin"])["delayed"].mean().unstack(),
        "month": data.groupby(["month", "origin"])["delayed"].mean().unstack(),
    }


def aircraft_profiles(df: pd.DataFrame, min_flights: int = 2000) -> dict:
    """Couverture du registre FAA et taux de retard par âge, constructeur et modèle d'avion."""
    delayed, _ = build_targets(df)
    data = df.assign(delayed=delayed)
    if "aircraft_age" not in data or data["aircraft_age"].isna().all():
        return {"coverage": 0.0, "by_age": [], "by_manufacturer": [], "by_model": []}

    age = pd.to_numeric(data["aircraft_age"], errors="coerce")
    data["age_bucket"] = pd.cut(age, [-1, 5, 10, 15, 20, 25, np.inf],
                                labels=["0-5", "6-10", "11-15", "16-20", "21-25", "> 25"])

    def profile(column: str) -> list[dict]:
        grouped = data.groupby(column, observed=True)["delayed"].agg(flights="size", delay_rate="mean")
        grouped = grouped[grouped["flights"] >= (1 if column == "age_bucket" else min_flights)]
        if column != "age_bucket":
            grouped = grouped.sort_values("flights", ascending=False).head(10)
        return grouped.round(4).reset_index().astype({column: str}).to_dict("records")

    return {
        "coverage": float(data["manufacturer"].notna().mean()) if "manufacturer" in data else float("nan"),
        "age_known": float(age.notna().mean()),
        "median_age": float(age.median()),
        "delay_rate_age_unknown": float(data.loc[age.isna(), "delayed"].mean()),
        "by_age": profile("age_bucket"),
        "by_manufacturer": profile("manufacturer") if "manufacturer" in data else [],
        "by_model": profile("aircraft_model") if "aircraft_model" in data else [],
    }


def calibration_table(y: pd.Series, proba: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Probabilité moyenne prédite vs taux observé, par décile de probabilité."""
    deciles = pd.qcut(proba, q=bins, labels=False, duplicates="drop")
    return (pd.DataFrame({"predicted": proba, "observed": y.to_numpy(), "decile": deciles})
            .groupby("decile").agg(predicted=("predicted", "mean"), observed=("observed", "mean"),
                                   flights=("observed", "size")))


def segment_metrics(test_df: pd.DataFrame, y: pd.Series, proba: np.ndarray, by: pd.Series) -> pd.DataFrame:
    rows = []
    for key, idx in pd.Series(range(len(by)), index=by.to_numpy()).groupby(level=0):
        positions = idx.to_numpy()
        y_seg, p_seg = y.iloc[positions], proba[positions]
        rows.append({"segment": str(key), "flights": len(positions), "base_rate": float(y_seg.mean()),
                     "roc_auc": float(roc_auc_score(y_seg, p_seg)),
                     "pr_auc": float(average_precision_score(y_seg, p_seg))})
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, model: DelayModel, *, test_months: int = 3, output_dir: Path = FIGURES_DIR) -> dict:
    _style()
    df = df.sort_values("fl_time_utc").reset_index(drop=True)
    train_df, test_df = split_train_test(df, test_months)
    X_test = build_features(test_df)
    y_test, minutes_test = build_targets(test_df)
    proba, minutes_pred = model.predict(test_df)
    predicted = model.is_delayed(proba)

    results: dict = {"model_version": model.model_version, "threshold": model.threshold,
                     "max_delay_minutes": model.max_delay_minutes, "figures": []}

    # --- Données ---
    _, minutes_all = build_targets(df)
    delayed_all, _ = build_targets(df)
    results["data"] = {
        "flights": len(df), "train_flights": len(train_df), "test_flights": len(test_df),
        "period": [str(df["fl_date"].min().date()), str(df["fl_date"].max().date())],
        "test_period": [str(test_df["fl_date"].min().date()), str(test_df["fl_date"].max().date())],
        "delay_rate": float(delayed_all.mean()),
        "delay_rate_by_origin": df.assign(d=delayed_all).groupby("origin")["d"].mean().round(4).to_dict(),
        "flights_by_origin": df["origin"].value_counts().to_dict(),
        "delay_minutes_quantiles": {f"p{int(q * 100)}": float(minutes_all.quantile(q)) for q in (0.5, 0.75, 0.9, 0.99)},
        "mean_delay_minutes": float(minutes_all.mean()),
        "missing_rate": build_features(df).isna().mean().round(4).to_dict(),
    }

    profiles = delay_rate_profiles(df)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for origin, color in zip(profiles["hour"].columns, (BLUE, ORANGE)):
        series = profiles["hour"][origin].dropna()
        ax.plot(series.index, series.to_numpy(), color=color, marker="o", markersize=4, label=origin)
    ax.set(title="Taux de retard ≥ 15 min selon l'heure de départ prévue (2024-2025)",
           xlabel="Heure locale de départ prévue", ylabel="Vols en retard")
    ax.set_xticks(range(0, 24, 2))
    _percent_axis(ax)
    ax.legend(loc="upper left")
    results["figures"].append(_save(fig, "taux_retard_heure.png", output_dir))

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for origin, color in zip(profiles["month"].columns, (BLUE, ORANGE)):
        series = profiles["month"][origin]
        ax.plot(series.index, series.to_numpy(), color=color, marker="o", markersize=4, label=origin)
    ax.axvspan(pd.Timestamp(results["data"]["test_period"][0]), series.index.max() + pd.offsets.MonthEnd(0),
               color=GRID, alpha=0.6, lw=0)
    ax.text(pd.Timestamp(results["data"]["test_period"][0]), ax.get_ylim()[1], " période de test",
            va="top", color=INK_MUTED, fontsize=9)
    ax.set(title="Taux de retard ≥ 15 min par mois", ylabel="Vols en retard")
    _percent_axis(ax)
    ax.legend(loc="upper left")
    results["figures"].append(_save(fig, "taux_retard_mois.png", output_dir))

    # --- Avions (registre FAA) ---
    results["aircraft"] = aircraft_profiles(df)
    if results["aircraft"]["by_age"]:
        by_age = pd.DataFrame(results["aircraft"]["by_age"])
        fig, ax = plt.subplots(figsize=(7, 3.6))
        bars = ax.bar(by_age["age_bucket"], by_age["delay_rate"], color=BLUE, width=0.6)
        for bar, n in zip(bars, by_age["flights"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{n:,} vols".replace(",", " "),
                    ha="center", va="bottom", fontsize=8, color=INK_MUTED)
        ax.set_ylim(0, by_age["delay_rate"].max() * 1.15)
        ax.set(title="Taux de retard ≥ 15 min selon l'âge de l'avion (registre FAA)",
               xlabel="Âge de l'avion (années)", ylabel="Vols en retard")
        _percent_axis(ax)
        ax.grid(axis="x", visible=False)
        results["figures"].append(_save(fig, "taux_retard_age_avion.png", output_dir))

    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    order = np.argsort(-proba)
    top_decile = order[: len(order) // 10]
    results["classifier"] = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "base_rate": float(y_test.mean()),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(tp / (tp + fp)), "recall": float(tp / (tp + fn)),
        "flagged_share": float(predicted.mean()),
        "mean_predicted_probability": float(proba.mean()),
        "top_decile_delay_rate": float(y_test.iloc[top_decile].mean()),
        "lift_top_decile": float(y_test.iloc[top_decile].mean() / y_test.mean()),
    }
    calibration = calibration_table(y_test, proba)
    results["calibration"] = calibration.round(4).reset_index().to_dict("records")
    results["by_origin"] = segment_metrics(test_df, y_test, proba, test_df["origin"]).round(4).to_dict("records")
    results["by_month"] = segment_metrics(
        test_df, y_test, proba, pd.to_datetime(test_df["fl_date"]).dt.strftime("%Y-%m")).round(4).to_dict("records")

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(9, 3.8))
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax_roc.plot(fpr, tpr, color=BLUE, label=f"Modèle (AUC {_fr(results['classifier']['roc_auc'])})")
    ax_roc.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls="--", label="Hasard (AUC 0,50)")
    ax_roc.set(title="Courbe ROC", xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs")
    ax_roc.legend(loc="lower right")
    precision, recall, _ = precision_recall_curve(y_test, proba)
    # Les tout premiers points (une poignée de vols) ne sont que du bruit visuel
    visible = recall >= 0.01
    ax_pr.plot(recall[visible], precision[visible], color=BLUE,
               label=f"Modèle (PR AUC {_fr(results['classifier']['pr_auc'])})")
    ax_pr.axhline(y_test.mean(), color=INK_MUTED, lw=1, ls="--", label=f"Hasard ({_fr(y_test.mean())})")
    ax_pr.scatter([results["classifier"]["recall"]], [results["classifier"]["precision"]], s=40, color=ORANGE,
                  zorder=3, edgecolor=SURFACE, linewidth=2, label=f"Seuil retenu ({_fr(model.threshold)})")
    ax_pr.set(title="Précision / rappel", xlabel="Rappel (retards détectés)", ylabel="Précision")
    ax_pr.set_ylim(0, 1)
    ax_pr.legend(loc="upper right")
    results["figures"].append(_save(fig, "courbes_roc_pr.png", output_dir))

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    upper = max(calibration["predicted"].max(), calibration["observed"].max()) * 1.05
    ax.plot([0, upper], [0, upper], color=INK_MUTED, lw=1, ls="--", label="Calibration parfaite")
    ax.plot(calibration["predicted"], calibration["observed"], color=BLUE, marker="o", markersize=6,
            label="Modèle (déciles)")
    ax.set(title="Calibration des probabilités", xlabel="Probabilité moyenne prédite", ylabel="Taux de retard observé",
           xlim=(0, upper), ylim=(0, upper))
    _percent_axis(ax, "x")
    _percent_axis(ax, "y")
    ax.legend(loc="upper left")
    results["figures"].append(_save(fig, "calibration.png", output_dir))

    importances = pd.Series(model.importances).sort_values()
    importances = importances[importances > 0]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([FEATURE_LABELS.get(i, i) for i in importances.index], importances.to_numpy(), color=BLUE, height=0.6)
    ax.set(title="Importance des features (permutation, jeu de test)",
           xlabel="Perte de PR AUC quand la feature est brouillée")
    ax.grid(axis="y", visible=False)
    results["importances"] = {k: round(float(v), 4) for k, v in importances.sort_values(ascending=False).items()}
    results["figures"].append(_save(fig, "importances.png", output_dir))

    # --- Régresseur ---
    buckets = pd.cut(minutes_test, [-0.1, 0, 15, 60, 180, np.inf], labels=["0", "1-15", "16-60", "61-180", "> 180"])
    by_bucket = pd.DataFrame({"observed": minutes_test, "predicted": minutes_pred, "bucket": buckets}).groupby(
        "bucket", observed=True).agg(flights=("observed", "size"), observed=("observed", "mean"),
                                     predicted=("predicted", "mean"))
    baseline = np.full(len(minutes_test), build_targets(train_df)[1].mean())
    results["regressor"] = {
        "mae": float(mean_absolute_error(minutes_test, minutes_pred)),
        "baseline_mae": float(mean_absolute_error(minutes_test, baseline)),
        "mean_observed": float(minutes_test.mean()), "mean_predicted": float(np.mean(minutes_pred)),
        "correlation": float(np.corrcoef(minutes_test, minutes_pred)[0, 1]),
        "capped_share": float(np.mean(minutes_pred >= model.max_delay_minutes)),
        "by_bucket": by_bucket.round(1).reset_index().astype({"bucket": str}).to_dict("records"),
    }

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(by_bucket))
    ax.bar(x - 0.2, by_bucket["observed"], width=0.38, color=BLUE, label="Retard observé (moyenne)")
    ax.bar(x + 0.2, by_bucket["predicted"], width=0.38, color=ORANGE, label="Retard prédit (moyenne)")
    ax.set_xticks(x, [f"{b} min\n({n:,} vols)".replace(",", " ") for b, n in zip(by_bucket.index, by_bucket["flights"])])
    ax.set(title="Régresseur : retard prédit selon le retard réellement observé", ylabel="Minutes")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    results["figures"].append(_save(fig, "regresseur_par_tranche.png", output_dir))

    del X_test
    (output_dir / "evaluation.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                                               encoding="utf-8")
    logger.info("Évaluation écrite dans %s (%s figures)", output_dir, len(results["figures"]))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.etl.warehouse import get_engine
    from src.ml.dataset import load_training_data

    parser = argparse.ArgumentParser(description="Évalue le dernier modèle et produit les figures du rapport")
    parser.add_argument("--test-months", type=int, default=3)
    args = parser.parse_args()
    evaluate(load_training_data(get_engine()), DelayModel.load(latest_model_path()), test_months=args.test_months)
