"""Streamlit GUI for presenting city and point-level predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

EARTH_RADIUS_KM = 6371.0088
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CITY_SCORES_PATH = PROJECT_ROOT / "data/07_model_output/city_location_scores.csv"
TOP_CANDIDATES_PATH = PROJECT_ROOT / "data/07_model_output/top_candidate_locations.csv"
ALL_CANDIDATES_PATH = PROJECT_ROOT / "data/07_model_output/all_candidate_locations.csv"
EXTRA_CITY_CENTERS_PATH = PROJECT_ROOT / "data/01_raw/extra_city_centers.csv"
STORE_POINTS_PATH = PROJECT_ROOT / "data/02_intermediate/store_points.csv"


def main() -> None:
    st.set_page_config(page_title="Predykcja lokalizacji Zabek", layout="wide")
    st.title("Predykcja nowych lokalizacji Zabek")

    city_scores = load_csv(CITY_SCORES_PATH)
    extra_cities = load_csv(EXTRA_CITY_CENTERS_PATH)
    store_points = load_csv(STORE_POINTS_PATH)
    candidates = load_csv(ALL_CANDIDATES_PATH)
    if candidates.empty:
        candidates = load_csv(TOP_CANDIDATES_PATH)

    if city_scores.empty or candidates.empty:
        st.warning("Brakuje wynikow modelu. Najpierw uruchom: kedro run --pipeline site_selection")
        st.stop()

    city_scores = city_scores.sort_values("city")
    city_options = build_city_options(city_scores, extra_cities)
    selected_city = st.selectbox("Miasto", city_options)

    if selected_city == "Wszystkie":
        city_row = global_summary(city_scores)
        urban_candidates = urban_candidate_filter(candidates, max_nearest_store_km=1.6)
        city_candidates = select_representative_locations(
            urban_candidates,
            limit=80,
            min_distance_km=5.0,
            min_score=0.3,
        )
    elif selected_city in set(city_scores["city"].astype(str)):
        city_row = city_scores[city_scores["city"].astype(str) == selected_city].iloc[0]
        city_candidates = candidates[candidates["nearest_city"].astype(str) == selected_city]
        city_candidates = urban_candidate_filter(city_candidates, max_nearest_store_km=1.6)
        if city_candidates.empty:
            st.info("Brak dobrych punktow kandydackich dla tego miasta w aktualnej siatce.")
            city_candidates = candidates.head(0)
        else:
            city_candidates = select_representative_locations(
                city_candidates,
                limit=20,
                min_distance_km=0.6,
                min_score=0.25,
            )
    else:
        city_row = extra_city_summary(selected_city, extra_cities, candidates)
        city_candidates = candidates_near_extra_city(selected_city, extra_cities, candidates)
        if city_candidates.empty:
            st.info("Brak ocenionych punktow w okolicy tego miasta w aktualnej siatce.")
        else:
            city_candidates = select_representative_locations(
                city_candidates,
                limit=20,
                min_distance_km=0.6,
                min_score=0.15,
            )

    probability = float(city_row["score"])
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Wynik", f"{probability:.1%}")
    col_b.metric("Pokazane punkty", len(city_candidates))
    col_c.metric("Mieszkancy", f"{int(city_row['city_population']):,}".replace(",", " "))
    col_d.metric("Zabki / 10 tys.", f"{float(city_row['stores_per_10k_residents']):.2f}")

    st.caption(
        "Wynik jest scorem modelu dla atrakcyjnosci lokalizacji, a nie gwarancja realnego "
        "otwarcia sklepu. Model bazuje na obecnych lokalizacjach i cechach przestrzennych."
    )

    show_map(city_candidates, store_points, selected_city)

    if selected_city == "Wszystkie":
        st.subheader("Najlepsze proponowane lokalizacje w Polsce")
    else:
        st.subheader("Proponowane lokalizacje w wybranym miescie")
    columns = [
        "candidate_id",
        "nearest_city",
        "score",
        "lon",
        "lat",
        "nearest_store_km",
        "stores_3km",
        "stores_10km",
        "city_population",
        "stores_per_10k_residents",
        "residents_per_store",
    ]
    st.dataframe(candidate_display(city_candidates[columns]), use_container_width=True)

    with st.expander("Ranking miast wedlug modelu"):
        st.dataframe(
            city_display(city_scores.sort_values("score", ascending=False).head(50)),
            use_container_width=True,
        )


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_city_options(city_scores: pd.DataFrame, extra_cities: pd.DataFrame) -> list[str]:
    cities = set(city_scores["city"].dropna().astype(str))
    if not extra_cities.empty:
        cities.update(extra_cities["city"].dropna().astype(str))
    return ["Wszystkie"] + sorted(cities)


def candidate_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "ID punktu",
                "Miasto",
                "Atrakcyjnosc",
                "Dlugosc geogr.",
                "Szerokosc geogr.",
                "Najblizsza Zabka (km)",
                "Zabki w promieniu 3 km",
                "Zabki w promieniu 10 km",
                "Mieszkancy miasta",
                "Zabki na 10 tys. mieszkancow",
                "Mieszkancow na 1 Zabke",
            ]
        )

    display = frame.sort_values("score", ascending=False).copy()
    display["score"] = (display["score"] * 100).round(1).astype(str) + "%"
    display["lon"] = display["lon"].round(5)
    display["lat"] = display["lat"].round(5)
    display["nearest_store_km"] = display["nearest_store_km"].round(2)
    display["city_population"] = display["city_population"].round(0).astype(int)
    display["stores_per_10k_residents"] = display["stores_per_10k_residents"].round(2)
    display["residents_per_store"] = display["residents_per_store"].round(0).astype(int)
    return display.rename(
        columns={
            "candidate_id": "ID punktu",
            "nearest_city": "Miasto",
            "score": "Atrakcyjnosc",
            "lon": "Dlugosc geogr.",
            "lat": "Szerokosc geogr.",
            "nearest_store_km": "Najblizsza Zabka (km)",
            "stores_3km": "Zabki w promieniu 3 km",
            "stores_10km": "Zabki w promieniu 10 km",
            "city_population": "Mieszkancy miasta",
            "stores_per_10k_residents": "Zabki na 10 tys. mieszkancow",
            "residents_per_store": "Mieszkancow na 1 Zabke",
        }
    )


def show_map(candidate_frame: pd.DataFrame, store_points: pd.DataFrame, selected_city: str) -> None:
    st.markdown(
        """
        <div style="display:flex; gap:18px; align-items:center; margin: 0 0 8px 0;">
          <span><span style="color:#16a34a; font-size:22px;">●</span> Istniejace Zabki</span>
          <span>
            <span style="color:#dc2626; font-size:22px;">●</span>
            Przewidywane lokalizacje
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    layers = []
    if not store_points.empty:
        stores = stores_for_selection(store_points, candidate_frame, selected_city)
        if not stores.empty:
            stores = stores[["lat", "lon"]].copy()
            stores["size"] = 28
            stores["color"] = "#16a34a"
            layers.append(stores)

    if not candidate_frame.empty:
        predictions = candidate_frame[["lat", "lon", "score"]].copy()
        predictions["size"] = (predictions["score"] * 140).clip(lower=35)
        predictions["color"] = "#dc2626"
        layers.append(predictions[["lat", "lon", "size", "color"]])

    if not layers:
        return

    map_frame = pd.concat(layers, ignore_index=True)
    st.map(map_frame, latitude="lat", longitude="lon", size="size", color="color")


def stores_for_selection(
    store_points: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    selected_city: str,
) -> pd.DataFrame:
    if selected_city != "Wszystkie":
        city_stores = store_points[store_points["city"].astype(str) == selected_city]
        if not city_stores.empty:
            return city_stores

    if candidate_frame.empty:
        return store_points.head(1000)

    min_lon = candidate_frame["lon"].min() - 0.05
    max_lon = candidate_frame["lon"].max() + 0.05
    min_lat = candidate_frame["lat"].min() - 0.05
    max_lat = candidate_frame["lat"].max() + 0.05
    nearby = store_points[
        store_points["lon"].between(min_lon, max_lon)
        & store_points["lat"].between(min_lat, max_lat)
    ]
    if selected_city == "Wszystkie":
        return nearby.head(1200)
    return nearby


def city_display(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "city",
        "score",
        "city_population",
        "stores_per_10k_residents",
        "residents_per_store",
        "candidate_count",
        "best_lon",
        "best_lat",
    ]
    display = frame[columns].copy()
    display["score"] = (display["score"] * 100).round(1).astype(str) + "%"
    display["city_population"] = display["city_population"].round(0).astype(int)
    display["stores_per_10k_residents"] = display["stores_per_10k_residents"].round(2)
    display["residents_per_store"] = display["residents_per_store"].round(0).astype(int)
    display["best_lon"] = display["best_lon"].round(5)
    display["best_lat"] = display["best_lat"].round(5)
    return display.rename(
        columns={
            "city": "Miasto",
            "score": "Wynik modelu",
            "city_population": "Mieszkancy",
            "stores_per_10k_residents": "Zabki na 10 tys.",
            "residents_per_store": "Mieszkancow na 1 Zabke",
            "candidate_count": "Liczba sprawdzonych punktow",
            "best_lon": "Dlugosc najlepszego punktu",
            "best_lat": "Szerokosc najlepszego punktu",
        }
    )


def global_summary(city_scores: pd.DataFrame) -> pd.Series:
    top_score = float(city_scores["score"].max())
    population = float(city_scores["city_population"].sum())
    stores_per_10k = float(city_scores["stores_per_10k_residents"].mean())
    residents_per_store = float(city_scores["residents_per_store"].median())
    return pd.Series(
        {
            "score": top_score,
            "city_population": population,
            "stores_per_10k_residents": stores_per_10k,
            "residents_per_store": residents_per_store,
        }
    )


def extra_city_summary(
    selected_city: str,
    extra_cities: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.Series:
    row = extra_cities[extra_cities["city"].astype(str) == selected_city].iloc[0]
    city_candidates = candidates_near_extra_city(selected_city, extra_cities, candidates)
    top_score = float(city_candidates["score"].max()) if not city_candidates.empty else 0.0
    population = float(row["population"])
    store_count = 1.0
    if not city_candidates.empty:
        store_count = float(city_candidates["nearest_city_store_count"].max())
    return pd.Series(
        {
            "score": top_score,
            "city_population": population,
            "stores_per_10k_residents": store_count / max(population, 1.0) * 10_000.0,
            "residents_per_store": population / max(store_count, 1.0),
        }
    )


def candidates_near_extra_city(
    selected_city: str,
    extra_cities: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    row = extra_cities[extra_cities["city"].astype(str) == selected_city].iloc[0]
    center = (float(row["lon"]), float(row["lat"]))
    radius_km = float(row["radius_km"])
    result = candidates.copy()
    result["distance_to_city_center_km"] = result.apply(
        lambda candidate: _distance_km(center, (float(candidate["lon"]), float(candidate["lat"]))),
        axis=1,
    )
    result = result[result["distance_to_city_center_km"] <= radius_km].copy()
    result = urban_candidate_filter(result, max_nearest_store_km=2.2)
    if not result.empty:
        result["nearest_city"] = selected_city
        result["city_population"] = float(row["population"])
    return result.sort_values("score", ascending=False)


def select_representative_locations(
    frame: pd.DataFrame,
    limit: int = 20,
    min_distance_km: float = 1.2,
    min_score: float = 0.15,
) -> pd.DataFrame:
    ranked = frame[frame["score"] >= min_score].sort_values("score", ascending=False)
    if ranked.empty:
        ranked = frame.sort_values("score", ascending=False).head(limit)

    selected_rows = []
    selected_points: list[tuple[float, float]] = []
    for row in ranked.itertuples(index=False):
        point = (float(row.lon), float(row.lat))
        if all(_distance_km(point, selected) >= min_distance_km for selected in selected_points):
            selected_rows.append(row._asdict())
            selected_points.append(point)
        if len(selected_rows) >= limit:
            break

    return pd.DataFrame(selected_rows)


def urban_candidate_filter(frame: pd.DataFrame, max_nearest_store_km: float) -> pd.DataFrame:
    if frame.empty:
        return frame
    filtered = frame[
        (frame["nearest_store_km"] <= max_nearest_store_km)
        & (frame["stores_3km"] >= 2)
        & (frame["stores_10km"] >= 5)
    ].copy()
    if filtered.empty:
        filtered = frame[frame["nearest_store_km"] <= max_nearest_store_km].copy()
    return filtered


def _distance_km(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    import math

    lon_a, lat_a = point_a
    lon_b, lat_b = point_b
    lon_a_rad = math.radians(lon_a)
    lat_a_rad = math.radians(lat_a)
    lon_b_rad = math.radians(lon_b)
    lat_b_rad = math.radians(lat_b)
    dlon = lon_b_rad - lon_a_rad
    dlat = lat_b_rad - lat_a_rad
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat_a_rad) * math.cos(lat_b_rad) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, max(0.0, a))))


if __name__ == "__main__":
    main()
