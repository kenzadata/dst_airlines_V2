"""Entraînement du modèle de retard au départ, avec suivi MLflow
(README section 6, étape 3 : algorithme "fonctionnel et optimisé",
MLflow pour le versionning des modèles).

Démarche :
1. Jeu de données : vols BTS de l'entrepôt joints à la météo et aux avions
   (`dataset.py`), triés chronologiquement.
2. Découpage temporel : les `test_months` derniers mois forment le jeu de
   test, jamais vu pendant l'entraînement ni la sélection de modèle.
3. Sélection de modèle : référence naïve, régression logistique et gradient
   boosting (recherche aléatoire d'hyperparamètres), comparés en validation
   croisée temporelle (`TimeSeriesSplit`) sur la précision moyenne (PR AUC),
   adaptée à une classe minoritaire (~20 % de vols en retard).
4. Seuil de décision : celui qui maximise le F1 sur la fin de la période
   d'entraînement.
5. Interprétabilité : importance par permutation sur le jeu de test.
6. MLflow : paramètres, métriques, comparaison des candidats (runs
   enfants), modèles journalisés et inscrits au Model Registry.

Usage :
    python -m src.ml.train [--test-months 3] [--n-iter 15] [--search-sample 200000]
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score

from src.ml.model import (
    FEATURE_COLUMNS,
    DelayModel,
    active_features,
    build_features,
    build_targets,
    make_hgb_classifier,
    make_hgb_regressor,
    make_logistic_classifier,
)

load_dotenv()

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "dst-airlines-delay-prediction"
REGISTERED_CLASSIFIER = "dst-airlines-delay-classifier"
REGISTERED_REGRESSOR = "dst-airlines-delay-regressor"

# MLflow 3 sérialise les modèles scikit-learn avec skops, qui refuse tout type
# non déclaré. Types internes de nos pipelines (dtype de l'OrdinalEncoder,
# validation d'entrée du gradient boosting), sans code arbitraire.
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "functools.partial", "sklearn.utils.validation.check_array"]

CLASSIFIER_SEARCH_SPACE = {
    "estimator__learning_rate": loguniform(0.02, 0.3),
    "estimator__max_iter": randint(150, 600),
    "estimator__max_leaf_nodes": randint(15, 128),
    "estimator__min_samples_leaf": randint(20, 400),
    "estimator__l2_regularization": loguniform(1e-3, 10),
    "estimator__max_features": uniform(0.5, 0.5),
}
REGRESSOR_SEARCH_SPACE = {
    # Poisson (effets multiplicatifs) ou quadratique (effets additifs, extrapole moins)
    "estimator__loss": ["poisson", "squared_error"],
    "estimator__learning_rate": loguniform(0.02, 0.3),
    "estimator__max_iter": randint(150, 600),
    "estimator__max_leaf_nodes": randint(15, 128),
    "estimator__min_samples_leaf": randint(20, 400),
    "estimator__l2_regularization": loguniform(1e-3, 10),
}


def split_train_test(df: pd.DataFrame, test_months: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpage temporel : les `test_months` derniers mois calendaires forment le test."""
    months = pd.to_datetime(df["fl_date"]).dt.to_period("M")
    test_start = months.max() - test_months + 1
    is_test = months >= test_start
    if is_test.all() or not is_test.any():
        raise ValueError(f"Période trop courte pour isoler {test_months} mois de test")
    return df[~is_test], df[is_test]


def _time_ordered_sample(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    """Sous-échantillon aléatoire qui conserve l'ordre chronologique (pour TimeSeriesSplit)."""
    if len(df) <= n:
        return df
    rng = np.random.default_rng(random_state)
    return df.iloc[np.sort(rng.choice(len(df), size=n, replace=False))]


def best_f1_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def classification_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = proba >= threshold
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted)),
        "brier": float(brier_score_loss(y_true, proba)),
        "base_rate": float(y_true.mean()),
    }


def select_classifier(
    X: pd.DataFrame, y: pd.Series, *, features: tuple[list[str], list[str]],
    n_iter: int, n_splits: int, random_state: int,
):
    """Compare les candidats en validation croisée temporelle ; renvoie (nom, pipeline non entraîné, résultats)."""
    cv = TimeSeriesSplit(n_splits=n_splits)
    results: dict[str, dict] = {}

    for name, estimator in [
        ("baseline_prior", DummyClassifier(strategy="prior")),
        ("logistic_regression", make_logistic_classifier(*features)),
    ]:
        scores = cross_val_score(estimator, X, y, cv=cv, scoring="average_precision")
        results[name] = {"cv_pr_auc": float(scores.mean()), "params": {}, "estimator": estimator}
        logger.info("%s : PR AUC CV = %.4f", name, scores.mean())

    search = RandomizedSearchCV(
        make_hgb_classifier(random_state, *features), CLASSIFIER_SEARCH_SPACE, n_iter=n_iter, cv=cv,
        scoring="average_precision", random_state=random_state, n_jobs=-1,
    )
    search.fit(X, y)
    results["hist_gradient_boosting"] = {
        "cv_pr_auc": float(search.best_score_),
        "params": {k.removeprefix("estimator__"): v for k, v in search.best_params_.items()},
        "estimator": clone(search.best_estimator_),
    }
    logger.info("hist_gradient_boosting : PR AUC CV = %.4f (%s)", search.best_score_, search.best_params_)

    best_name = max(results, key=lambda name: results[name]["cv_pr_auc"])
    return best_name, results[best_name]["estimator"], results


def tune_regressor(
    X: pd.DataFrame, y: pd.Series, *, features: tuple[list[str], list[str]],
    n_iter: int, n_splits: int, random_state: int,
):
    search = RandomizedSearchCV(
        make_hgb_regressor(random_state, *features), REGRESSOR_SEARCH_SPACE, n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=n_splits), scoring="neg_mean_absolute_error",
        random_state=random_state, n_jobs=-1,
    )
    search.fit(X, y)
    logger.info("Régresseur : MAE CV = %.2f min (%s)", -search.best_score_, search.best_params_)
    return clone(search.best_estimator_), {k.removeprefix("estimator__"): v for k, v in search.best_params_.items()}


def _permutation_importances(model: DelayModel, X: pd.DataFrame, y: pd.Series, random_state: int) -> dict[str, float]:
    sample = X.sample(n=min(len(X), 20_000), random_state=random_state)
    result = permutation_importance(
        model.classifier, sample, y.loc[sample.index], scoring="average_precision",
        n_repeats=5, random_state=random_state, n_jobs=-1,
    )
    return {name: max(float(score), 0.0) for name, score in zip(FEATURE_COLUMNS, result.importances_mean)}


def train(
    df: pd.DataFrame,
    *,
    model_version: str | None = None,
    test_months: int = 3,
    n_iter: int = 15,
    n_splits: int = 3,
    search_sample: int = 200_000,
    random_state: int = 42,
    register: bool = True,
) -> tuple[DelayModel, dict[str, float]]:
    model_version = model_version or datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    df = df.sort_values("fl_time_utc").reset_index(drop=True) if "fl_time_utc" in df else df
    train_df, test_df = split_train_test(df, test_months)
    X_train, X_test = build_features(train_df), build_features(test_df)
    (y_train, minutes_train), (y_test, minutes_test) = build_targets(train_df), build_targets(test_df)
    logger.info("Entraînement : %s vols, test : %s vols (à partir du %s)",
                len(train_df), len(test_df), test_df["fl_date"].min().date())
    features = active_features(X_train)
    dropped = sorted(set(FEATURE_COLUMNS) - set(features[0]) - set(features[1]))
    if dropped:
        logger.warning("Features entièrement vides, écartées du modèle : %s", dropped)

    # 1. Sélection et optimisation sur un échantillon chronologique de l'entraînement
    search_idx = _time_ordered_sample(X_train, search_sample, random_state).index
    best_name, best_classifier, candidates = select_classifier(
        X_train.loc[search_idx], y_train.loc[search_idx], features=features,
        n_iter=n_iter, n_splits=n_splits, random_state=random_state,
    )
    regressor, regressor_params = tune_regressor(
        X_train.loc[search_idx], minutes_train.loc[search_idx], features=features,
        n_iter=max(n_iter // 2, 1), n_splits=n_splits, random_state=random_state,
    )

    # 2. Seuil de décision : entraîné sur 90 % de la période, choisi sur les 10 % suivants
    cut = int(len(X_train) * 0.9)
    threshold_model = clone(best_classifier).fit(X_train.iloc[:cut], y_train.iloc[:cut])
    threshold = best_f1_threshold(y_train.iloc[cut:], threshold_model.predict_proba(X_train.iloc[cut:])[:, 1])

    # 3. Modèles finaux sur toute la période d'entraînement, évalués sur le test
    model = DelayModel(
        model_version=model_version,
        classifier=best_classifier.fit(X_train, y_train),
        regressor=regressor.fit(X_train, minutes_train),
        threshold=threshold,
        max_delay_minutes=float(np.quantile(minutes_train, 0.99)),
    )
    proba_test = model.classifier.predict_proba(X_test)[:, 1]
    minutes_pred = model.predict_minutes(X_test)
    baseline_minutes = DummyRegressor(strategy="mean").fit(X_train, minutes_train).predict(X_test)

    metrics = {f"test_{k}": v for k, v in classification_metrics(y_test, proba_test, threshold).items()}
    metrics.update({
        "test_mae_minutes": float(mean_absolute_error(minutes_test, minutes_pred)),
        "test_rmse_minutes": float(root_mean_squared_error(minutes_test, minutes_pred)),
        "baseline_mae_minutes": float(mean_absolute_error(minutes_test, baseline_minutes)),
        "baseline_rmse_minutes": float(root_mean_squared_error(minutes_test, baseline_minutes)),
        "threshold": threshold,
        "max_delay_minutes": model.max_delay_minutes,
    })
    model.metrics = metrics
    model.importances = _permutation_importances(model, X_test, y_test, random_state)

    saved_path = model.save()
    _log_to_mlflow(model, candidates, best_name, regressor_params, train_df, test_df, X_test, register, dropped)
    logger.info("Modèle %s (%s) sauvegardé dans %s", model_version, best_name, saved_path)
    logger.info("Métriques de test : %s", {k: round(v, 4) for k, v in metrics.items()})
    return model, metrics


def _log_to_mlflow(model, candidates, best_name, regressor_params, train_df, test_df, X_test, register, dropped) -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=model.model_version):
        mlflow.log_params({
            "model_version": model.model_version,
            "selected_classifier": best_name,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "train_period": f"{train_df['fl_date'].min().date()} -> {train_df['fl_date'].max().date()}",
            "test_period": f"{test_df['fl_date'].min().date()} -> {test_df['fl_date'].max().date()}",
            "features": ",".join(c for c in FEATURE_COLUMNS if c not in dropped),
            "dropped_empty_features": ",".join(dropped) or "aucune",
        })
        mlflow.log_params({f"clf_{k}": v for k, v in candidates[best_name]["params"].items()})
        mlflow.log_params({f"reg_{k}": v for k, v in regressor_params.items()})
        mlflow.log_metrics(model.metrics)
        mlflow.log_dict(model.importances, "feature_importances.json")

        for name, result in candidates.items():
            with mlflow.start_run(run_name=f"candidate-{name}", nested=True):
                mlflow.log_params({"candidate": name, **result["params"]})
                mlflow.log_metric("cv_pr_auc", result["cv_pr_auc"])

        example = X_test.head(5)
        for estimator, name, registered in [
            (model.classifier, "classifier", REGISTERED_CLASSIFIER),
            (model.regressor, "regressor", REGISTERED_REGRESSOR),
        ]:
            output = estimator.predict_proba(example) if name == "classifier" else estimator.predict(example)
            mlflow.sklearn.log_model(
                estimator, name=name, input_example=example,
                signature=infer_signature(example, output),
                registered_model_name=registered if register else None,
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.etl.warehouse import get_engine
    from src.ml.dataset import load_training_data

    parser = argparse.ArgumentParser(description="Entraîne le modèle de retard sur l'entrepôt historique")
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=15, help="Itérations de la recherche aléatoire")
    parser.add_argument("--search-sample", type=int, default=200_000,
                        help="Taille max de l'échantillon pour la sélection de modèle")
    args = parser.parse_args()

    data = load_training_data(get_engine())
    logger.info("%s vols chargés depuis l'entrepôt", len(data))
    train(data, test_months=args.test_months, n_iter=args.n_iter, search_sample=args.search_sample)
