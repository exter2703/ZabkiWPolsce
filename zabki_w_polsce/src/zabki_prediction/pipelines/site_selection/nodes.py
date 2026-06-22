"""Nodes for predicting attractive new Zabka locations."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

EARTH_RADIUS_KM = 6371.0088


def parse_zabka_geojson(raw_geojson: dict[str, Any]):
    """Convert Overpass GeoJSON into a flat table of store points."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for feature in raw_geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Point":
            continue
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) < 2:
            continue
        lon, lat = coordinates[:2]
        properties = feature.get("properties") or {}
        rows.append(
            {
                "store_id": feature.get("id") or properties.get("@id"),
                "lon": float(lon),
                "lat": float(lat),
                "city": properties.get("addr:city") or "unknown",
                "street": properties.get("addr:street"),
                "postcode": properties.get("addr:postcode"),
                "source": "osm_overpass",
            }
        )

    if not rows:
        raise ValueError("No point features were found in the Zabka GeoJSON file.")

    frame = pd.DataFrame(rows).drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
    return frame


def build_model_tables(store_points, city_population, params: dict[str, Any]):
    """Create supervised training rows and unlabeled candidate locations."""
    import numpy as np
    import pandas as pd

    bbox = params["poland_bbox"]
    random_state = int(params.get("random_state", 42))
    step = float(params.get("candidate_grid_step_degrees", 0.06))
    min_distance = float(params.get("min_candidate_distance_km", 0.35))

    lons = np.arange(bbox["min_lon"], bbox["max_lon"] + step, step)
    lats = np.arange(bbox["min_lat"], bbox["max_lat"] + step, step)
    candidate_grid = pd.DataFrame(
        [(float(lon), float(lat)) for lat in lats for lon in lons],
        columns=["lon", "lat"],
    )
    candidate_grid["candidate_id"] = [
        f"grid_{idx:06d}" for idx in range(len(candidate_grid))
    ]

    candidate_features = add_spatial_features(
        candidate_grid,
        store_points,
        city_population,
        exclude_self=False,
    )
    candidate_features = candidate_features[
        candidate_features["nearest_store_km"] >= min_distance
    ].reset_index(drop=True)

    negative_ratio = float(params.get("negative_ratio", 2.0))
    n_negatives = min(len(candidate_features), int(len(store_points) * negative_ratio))
    negatives = (
        candidate_features.sample(n=n_negatives, random_state=random_state)
        .assign(label=0)
        .reset_index(drop=True)
    )

    positives = store_points[["lon", "lat"]].copy()
    positives["candidate_id"] = store_points["store_id"].astype(str)
    positives = add_spatial_features(
        positives,
        store_points,
        city_population,
        exclude_self=True,
    )
    positives["label"] = 1

    training_table = pd.concat([positives, negatives], ignore_index=True)
    training_table = training_table.sample(frac=1.0, random_state=random_state)
    training_table = training_table.reset_index(drop=True)

    return training_table, candidate_features


def add_spatial_features(points, store_points, city_population=None, exclude_self: bool = False):
    """Add density and distance features based on the current store network."""
    import numpy as np

    result = points.copy().reset_index(drop=True)
    point_lon = result["lon"].to_numpy(dtype=float)
    point_lat = result["lat"].to_numpy(dtype=float)
    store_lon = store_points["lon"].to_numpy(dtype=float)
    store_lat = store_points["lat"].to_numpy(dtype=float)

    nearest, counts = _distance_features(
        point_lon=point_lon,
        point_lat=point_lat,
        store_lon=store_lon,
        store_lat=store_lat,
        radii_km=(1.0, 3.0, 5.0, 10.0, 25.0),
        exclude_self=exclude_self,
    )

    result["nearest_store_km"] = nearest
    result["stores_1km"] = counts[1.0]
    result["stores_3km"] = counts[3.0]
    result["stores_5km"] = counts[5.0]
    result["stores_10km"] = counts[10.0]
    result["stores_25km"] = counts[25.0]

    city_features = _nearest_city_features(result, store_points, city_population)
    result["nearest_city"] = city_features["nearest_city"]
    result["nearest_city_center_km"] = city_features["nearest_city_center_km"]
    result["nearest_city_store_count"] = city_features["nearest_city_store_count"]
    result["city_population"] = city_features["city_population"]
    result["log_city_population"] = np.log1p(result["city_population"].to_numpy(dtype=float))
    result["stores_per_10k_residents"] = (
        result["nearest_city_store_count"].to_numpy(dtype=float)
        / result["city_population"].clip(lower=1).to_numpy(dtype=float)
        * 10_000.0
    )
    result["residents_per_store"] = (
        result["city_population"].to_numpy(dtype=float)
        / result["nearest_city_store_count"].clip(lower=1).to_numpy(dtype=float)
    )
    result["population_missing"] = city_features["population_missing"]

    result["lon_sin"] = np.sin(np.radians(result["lon"].to_numpy(dtype=float)))
    result["lon_cos"] = np.cos(np.radians(result["lon"].to_numpy(dtype=float)))
    result["lat_sin"] = np.sin(np.radians(result["lat"].to_numpy(dtype=float)))
    result["lat_cos"] = np.cos(np.radians(result["lat"].to_numpy(dtype=float)))

    return result


def train_autogluon_and_rank_candidates(
    training_table,
    candidate_table,
    modeling_params: dict[str, Any],
    mlflow_params: dict[str, Any],
):
    """Train AutoGluon with MLflow tracking and rank candidate locations."""
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise RuntimeError(
            "AutoGluon is required for the final training pipeline. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    import mlflow
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    feature_columns = modeling_params["feature_columns"]
    label_column = "label"
    random_state = int(modeling_params.get("random_state", 42))
    model_path = Path(modeling_params["model_path"])
    top_n = int(modeling_params.get("top_n_candidates", 250))

    train_df, valid_df = train_test_split(
        training_table[feature_columns + [label_column]],
        test_size=0.2,
        random_state=random_state,
        stratify=training_table[label_column],
    )

    if model_path.exists():
        shutil.rmtree(model_path)

    mlflow.set_tracking_uri(mlflow_params.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_params.get("experiment_name", "zabka_location_prediction"))

    with mlflow.start_run(run_name="autogluon_site_selection"):
        mlflow.log_params(
            {
                "model_type": "AutoGluon TabularPredictor",
                "train_rows": len(train_df),
                "valid_rows": len(valid_df),
                "candidate_rows": len(candidate_table),
                "feature_count": len(feature_columns),
                "presets": modeling_params.get("autogluon_presets", "medium_quality"),
            }
        )

        predictor = TabularPredictor(
            label=label_column,
            path=str(model_path),
            problem_type="binary",
            eval_metric="roc_auc",
        ).fit(
            train_data=train_df,
            presets=modeling_params.get("autogluon_presets", "medium_quality"),
            time_limit=int(modeling_params.get("autogluon_time_limit_seconds", 600)),
        )

        valid_scores = _positive_class_scores(predictor.predict_proba(valid_df[feature_columns]))
        valid_pred = (valid_scores >= 0.5).astype(int)
        metrics = {
            "roc_auc": float(roc_auc_score(valid_df[label_column], valid_scores)),
            "average_precision": float(
                average_precision_score(valid_df[label_column], valid_scores)
            ),
            "accuracy_at_0_5": float(accuracy_score(valid_df[label_column], valid_pred)),
        }
        mlflow.log_metrics(metrics)

        leaderboard = predictor.leaderboard(valid_df, silent=True)
        leaderboard_path = Path("data/08_reporting/autogluon_leaderboard.csv")
        leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(leaderboard_path, index=False)
        mlflow.log_artifact(str(leaderboard_path), artifact_path="reports")
        mlflow.log_artifacts(str(model_path), artifact_path="autogluon_model")

        registered_name = mlflow_params.get("registered_model_name")
        if registered_name:
            metrics["registered_model_name"] = registered_name

    candidate_scores = _positive_class_scores(
        predictor.predict_proba(candidate_table[feature_columns])
    )
    scored_candidates = candidate_table.copy()
    scored_candidates["score"] = candidate_scores
    scored_candidates = scored_candidates.sort_values("score", ascending=False)
    scored_candidates = scored_candidates.reset_index(drop=True)

    top_candidates = scored_candidates.copy()
    top_candidates = (
        top_candidates.sort_values("score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    city_scores = build_city_scores(scored_candidates)
    monitoring_profile = _training_profile(training_table, feature_columns)
    return metrics, top_candidates, scored_candidates, city_scores, monitoring_profile


def build_city_scores(scored_candidates):
    """Aggregate candidate-level model scores into city-level presentation scores."""
    import pandas as pd

    if "nearest_city" not in scored_candidates.columns:
        return pd.DataFrame()

    grouped = (
        scored_candidates.groupby("nearest_city", dropna=False)
        .agg(
            score=("score", "max"),
            mean_top_score=("score", "mean"),
            candidate_count=("candidate_id", "count"),
            best_lon=("lon", "first"),
            best_lat=("lat", "first"),
            nearest_store_km=("nearest_store_km", "first"),
            stores_3km=("stores_3km", "first"),
            stores_10km=("stores_10km", "first"),
            nearest_city_store_count=("nearest_city_store_count", "first"),
            city_population=("city_population", "first"),
            stores_per_10k_residents=("stores_per_10k_residents", "first"),
            residents_per_store=("residents_per_store", "first"),
            population_missing=("population_missing", "first"),
        )
        .reset_index()
        .rename(columns={"nearest_city": "city"})
    )
    return grouped.sort_values("score", ascending=False).reset_index(drop=True)


def score_candidate_dataframe(points, store_points, model_path: str, city_population=None):
    """Score ad-hoc points using a trained AutoGluon model."""
    from autogluon.tabular import TabularPredictor

    features = add_spatial_features(points, store_points, city_population, exclude_self=False)
    predictor = TabularPredictor.load(model_path)
    scores = _positive_class_scores(predictor.predict_proba(features))
    result = features.copy()
    result["score"] = scores
    return result


def _distance_features(
    point_lon,
    point_lat,
    store_lon,
    store_lat,
    radii_km: tuple[float, ...],
    exclude_self: bool,
    batch_size: int = 512,
):
    import numpy as np

    nearest = np.empty(len(point_lon), dtype=float)
    counts = {radius: np.empty(len(point_lon), dtype=int) for radius in radii_km}

    store_lon_rad = np.radians(store_lon)
    store_lat_rad = np.radians(store_lat)
    cos_store_lat = np.cos(store_lat_rad)

    for start in range(0, len(point_lon), batch_size):
        end = min(start + batch_size, len(point_lon))
        lon_rad = np.radians(point_lon[start:end])[:, None]
        lat_rad = np.radians(point_lat[start:end])[:, None]

        dlon = store_lon_rad[None, :] - lon_rad
        dlat = store_lat_rad[None, :] - lat_rad
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat_rad) * cos_store_lat[None, :] * np.sin(dlon / 2.0) ** 2
        )
        distances = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        if exclude_self:
            distances[distances < 0.001] = np.inf

        nearest[start:end] = np.min(distances, axis=1)
        for radius in radii_km:
            counts[radius][start:end] = np.sum(distances <= radius, axis=1)

    return nearest, counts


def _nearest_city_features(points, store_points, city_population=None):
    import numpy as np
    import pandas as pd

    cities = (
        store_points.groupby("city", dropna=False)
        .agg(lon=("lon", "mean"), lat=("lat", "mean"), store_count=("store_id", "count"))
        .reset_index()
    )
    city_lon = cities["lon"].to_numpy(dtype=float)
    city_lat = cities["lat"].to_numpy(dtype=float)
    city_counts = cities["store_count"].to_numpy(dtype=float)
    population_by_city, default_population = _population_lookup(city_population)

    nearest_city = []
    nearest_distance = []
    nearest_count = []
    nearest_population = []
    population_missing = []
    for row in points[["lon", "lat"]].itertuples(index=False):
        distances = _haversine_vector(row.lon, row.lat, city_lon, city_lat)
        idx = int(np.argmin(distances))
        city_name = str(cities.loc[idx, "city"])
        normalized_city = _normalize_city_name(city_name)
        population = population_by_city.get(normalized_city)
        missing = population is None
        if missing:
            population = default_population

        nearest_city.append(city_name)
        nearest_distance.append(float(distances[idx]))
        nearest_count.append(float(city_counts[idx]))
        nearest_population.append(float(population))
        population_missing.append(int(missing))

    return pd.DataFrame(
        {
            "nearest_city": nearest_city,
            "nearest_city_center_km": nearest_distance,
            "nearest_city_store_count": nearest_count,
            "city_population": nearest_population,
            "population_missing": population_missing,
        }
    )


def _population_lookup(city_population) -> tuple[dict[str, float], float]:
    if city_population is None or len(city_population) == 0:
        return {}, 50_000.0

    frame = city_population.copy()
    frame["city_norm"] = frame["city"].map(_normalize_city_name)
    frame = frame.dropna(subset=["city_norm", "population"])
    default_population = float(frame["population"].median()) if len(frame) else 50_000.0
    population_lookup = dict(
        zip(frame["city_norm"], frame["population"].astype(float), strict=False)
    )
    return population_lookup, default_population


def _normalize_city_name(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().casefold()


def _haversine_vector(lon: float, lat: float, other_lon, other_lat):
    import numpy as np

    lon_rad = math.radians(lon)
    lat_rad = math.radians(lat)
    other_lon_rad = np.radians(other_lon)
    other_lat_rad = np.radians(other_lat)
    dlon = other_lon_rad - lon_rad
    dlat = other_lat_rad - lat_rad
    a = np.sin(dlat / 2.0) ** 2 + math.cos(lat_rad) * np.cos(other_lat_rad) * np.sin(
        dlon / 2.0
    ) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _positive_class_scores(probabilities):
    if hasattr(probabilities, "columns"):
        if 1 in probabilities.columns:
            return probabilities[1].to_numpy()
        if "1" in probabilities.columns:
            return probabilities["1"].to_numpy()
        return probabilities.iloc[:, -1].to_numpy()
    return probabilities[:, -1]


def _training_profile(training_table, feature_columns: list[str]) -> dict[str, Any]:
    profile: dict[str, Any] = {"feature_columns": feature_columns, "rows": int(len(training_table))}
    for column in feature_columns:
        series = training_table[column]
        profile[column] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return json.loads(json.dumps(profile))
