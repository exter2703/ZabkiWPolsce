"""Local production API for candidate scoring and monitoring."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from zabki_prediction.pipelines.site_selection.nodes import score_candidate_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STORE_POINTS_PATH = PROJECT_ROOT / "data/02_intermediate/store_points.csv"
CITY_POPULATION_PATH = PROJECT_ROOT / "data/01_raw/city_population.csv"
TOP_CANDIDATES_PATH = PROJECT_ROOT / "data/07_model_output/top_candidate_locations.csv"
CITY_SCORES_PATH = PROJECT_ROOT / "data/07_model_output/city_location_scores.csv"
MODEL_PATH = PROJECT_ROOT / "data/06_models/autogluon_site_selector"
PREDICTION_LOG_PATH = PROJECT_ROOT / "data/08_reporting/prediction_log.jsonl"
TRAINING_PROFILE_PATH = PROJECT_ROOT / "data/08_reporting/training_profile.json"

app = FastAPI(title="Zabka Location Prediction API", version="0.1.0")


class LocationRequest(BaseModel):
    lon: float = Field(..., ge=14.0, le=24.5)
    lat: float = Field(..., ge=48.8, le=55.0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict_location(request: LocationRequest) -> dict[str, float]:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model is not trained yet. Run: kedro run")
    if not STORE_POINTS_PATH.exists():
        raise HTTPException(status_code=503, detail="Store features are missing. Run: kedro run")

    store_points = pd.read_csv(STORE_POINTS_PATH)
    city_population = _load_city_population()
    points = pd.DataFrame([{"lon": request.lon, "lat": request.lat}])
    scored = score_candidate_dataframe(
        points,
        store_points,
        str(MODEL_PATH),
        city_population,
    ).iloc[0]
    response = {
        "lon": float(request.lon),
        "lat": float(request.lat),
        "score": float(scored["score"]),
        "nearest_store_km": float(scored["nearest_store_km"]),
        "stores_3km": float(scored["stores_3km"]),
        "stores_10km": float(scored["stores_10km"]),
        "city_population": float(scored["city_population"]),
        "stores_per_10k_residents": float(scored["stores_per_10k_residents"]),
    }
    _log_prediction(response)
    return response


@app.get("/candidates")
def top_candidates(limit: int = 100) -> list[dict[str, Any]]:
    if not TOP_CANDIDATES_PATH.exists():
        raise HTTPException(status_code=503, detail="Candidate ranking is missing. Run: kedro run")
    frame = pd.read_csv(TOP_CANDIDATES_PATH).head(max(1, min(limit, 500)))
    return _records_for_json(frame)


@app.get("/cities")
def city_scores(limit: int = 100) -> list[dict[str, Any]]:
    if not CITY_SCORES_PATH.exists():
        raise HTTPException(status_code=503, detail="City ranking is missing. Run: kedro run")
    frame = pd.read_csv(CITY_SCORES_PATH).sort_values("score", ascending=False)
    return _records_for_json(frame.head(max(1, min(limit, 500))))


@app.get("/monitoring/drift")
def monitoring_drift() -> dict[str, object]:
    if not PREDICTION_LOG_PATH.exists():
        return {"status": "no_predictions_logged", "prediction_count": 0}

    rows = [json.loads(line) for line in PREDICTION_LOG_PATH.read_text().splitlines() if line]
    if not rows:
        return {"status": "no_predictions_logged", "prediction_count": 0}

    scores = [row["score"] for row in rows if "score" in row]
    mean_score = sum(scores) / len(scores)
    status = "ok"
    if TRAINING_PROFILE_PATH.exists() and (mean_score < 0.05 or mean_score > 0.95):
        status = "check_score_distribution"

    return {
        "status": status,
        "prediction_count": len(rows),
        "mean_logged_score": mean_score,
        "last_prediction_at": rows[-1].get("timestamp"),
    }


@app.get("/map", response_class=HTMLResponse)
def candidate_map() -> str:
    return """
<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Zabka candidate map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map { height: 100%; margin: 0; }
    .legend {
      position: absolute; z-index: 500; right: 12px; top: 12px;
      background: white; padding: 10px 12px; border-radius: 6px;
      font: 14px/1.35 system-ui, sans-serif; box-shadow: 0 2px 12px #0002;
    }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="legend">Top predicted Zabka locations</div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const map = L.map('map').setView([52.1, 19.2], 6);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);
    fetch('/candidates?limit=250')
      .then(response => response.json())
      .then(rows => {
        rows.forEach(row => {
          const radius = 4 + 10 * Number(row.score || 0);
          L.circleMarker([row.lat, row.lon], {
            radius, color: '#0f766e', fillColor: '#14b8a6', fillOpacity: 0.65, weight: 1
          }).bindPopup(
            `Score: ${Number(row.score).toFixed(3)}<br>` +
            `Nearest store: ${Number(row.nearest_store_km).toFixed(2)} km`
          )
            .addTo(map);
        });
      });
  </script>
</body>
</html>
"""


def _log_prediction(payload: dict[str, float]) -> None:
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(UTC).isoformat(), **payload}
    with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _load_city_population() -> pd.DataFrame | None:
    if not CITY_POPULATION_PATH.exists():
        return None
    return pd.read_csv(CITY_POPULATION_PATH)


def _records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.to_dict(orient="records")
    return [{key: _json_value(value) for key, value in row.items()} for row in records]


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def run() -> None:
    import uvicorn

    uvicorn.run("zabki_prediction.api.main:app", host="0.0.0.0", port=8000, reload=False)

