# Predykcja nowych lokalizacji Zabek w Polsce

Projekt przewiduje atrakcyjne punkty dla nowych sklepow Zabka na podstawie obecnych lokalizacji z OpenStreetMap/Overpass (`zabka_locs.geojson`) oraz danych demograficznych GUS o liczbie mieszkancow miast. Rozwiazanie spelnia prostsza sciezke z wymagan: **Automatyzacja MLOps**.

## Wybrana sciezka wymagan

- Kedro: pipeline `site_selection` obejmuje wczytanie danych, preprocessing, trening i ewaluacje.
- MLflow: eksperymenty, parametry, metryki i artefakty modelu sa logowane do `mlruns/`.
- AutoGluon: finalny model to `TabularPredictor`.
- Baseline: notebook `notebooks/01_baseline.ipynb` zawiera EDA, preprocessing, model bazowy i metryki.
- Produkcja: FastAPI udostepnia predykcje, liste kandydatow, monitoring i mape.
- GUI: Streamlit pozwala wybrac miasto i pokazuje score pojawienia sie atrakcyjnej lokalizacji.
- Monitoring: predykcje sa zapisywane w `data/08_reporting/prediction_log.jsonl`, endpoint driftu jest pod `/monitoring/drift`.
- MLOps B: sa workflow CI, CD oraz Continuous Training w `.github/workflows/`.

## Jak dziala model

Poniewaz mamy tylko obecne lokalizacje sklepow, pipeline tworzy problem klasyfikacji przestrzennej:

1. obecne sklepy sa pozytywnymi przykladami,
2. punkty siatki nad Polska oddalone od istniejacych sklepow sa negatywnymi/kandydackimi przykladami,
3. cechy opisuja m.in. odleglosc do najblizszego sklepu, liczbe Zabek w promieniach 1/3/5/10/25 km, liczbe mieszkancow najblizszego miasta, Zabki na 10 tys. mieszkancow oraz odleglosc do najblizszego centrum miasta wywnioskowanego z danych OSM,
4. AutoGluon wybiera najlepszy model tabularny i ranking kandydatow trafia do `data/07_model_output/top_candidate_locations.csv`.

## Dane

- `zabka_locs.geojson` - obecne lokalizacje sklepow z OpenStreetMap/Overpass.
- `data/01_raw/city_population.csv` - snapshot danych GUS: "Ludnosc. Stan i struktura ludnosci oraz ruch naturalny w przekroju terytorialnym w 2025 r. Stan w dniu 30 czerwca", tabela 11. Zrodlo publikacji: `https://stat.gov.pl/obszary-tematyczne/ludnosc/ludnosc/ludnosc-stan-i-struktura-ludnosci-oraz-ruch-naturalny-w-przekroju-terytorialnym-w-2025-r-stan-w-dniu-30-czerwca,6,39.html`.

## Instalacja

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Uruchomienie pipeline

```powershell
kedro run --pipeline site_selection
```

## Uruchomienie API i mapy

```powershell
uvicorn zabki_prediction.api.main:app --reload
```

Po starcie:

- API: `http://127.0.0.1:8000/docs`
- mapa kandydatow: `http://127.0.0.1:8000/map`
- monitoring: `http://127.0.0.1:8000/monitoring/drift`

## Uruchomienie GUI Streamlit

```powershell
streamlit run src/zabki_prediction/ui/streamlit_app.py
```

GUI pokazuje ranking miast, wynik dla wybranego miasta, najlepsze punkty kandydackie i mape. Wynik jest scorem modelu dla atrakcyjnosci lokalizacji, a nie gwarancja realnego otwarcia sklepu.

## Docker

Najprostsze uruchomienie calego projektu:

```powershell
docker compose up --build
```

Po starcie:

- Streamlit GUI: `http://127.0.0.1:8501`
- FastAPI docs: `http://127.0.0.1:8000/docs`
- FastAPI mapa: `http://127.0.0.1:8000/map`

Zatrzymanie:

```powershell
docker compose down
```

Sam obraz Dockera podczas budowania uruchamia `kedro run --pipeline site_selection`, wiec model i rankingi sa tworzone wewnatrz kontenera.

Alternatywnie samo API:

```powershell
docker build -t zabki-location-app .
docker run -p 8000:8000 zabki-location-app
```

## Uwaga o danych mapowych

Projekt nie wymaga recznego wybierania punktow na mapie. Mapa sluzy do wizualizacji rankingu, a kandydaci sa generowani automatycznie z siatki geograficznej nad Polska. W kolejnej iteracji mozna podlaczyc dodatkowe zrodla, np. gestosc ludnosci, przystanki, konkurencje i lokale uslugowe z OSM.

## Autorzy: Oskar Wiktorowicz (s28038), Jakub Truszczyński (s28774)
