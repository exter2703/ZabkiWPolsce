FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV KEDRO_DISABLE_TELEMETRY=1

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY conf conf
COPY data/01_raw data/01_raw
COPY src src
COPY zabka_locs.geojson ./

RUN pip install --no-cache-dir -e .
RUN kedro run --pipeline site_selection

EXPOSE 8000 8501
CMD ["uvicorn", "zabki_prediction.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
