# Obraz serwujący model jakości wina (Flask + Feast + monitoring).
# Kontekst budowania = katalog główny repo (patrz docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app

# zależności systemowe (lekkie — dla scipy/pyarrow zwykle wystarczą wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# najpierw zależności (lepszy cache warstw)
COPY serving/requirements.txt /app/serving/requirements.txt
RUN pip install --no-cache-dir -r /app/serving/requirements.txt

# kod aplikacji
COPY serving/ /app/serving/
COPY feature_repo/ /app/feature_repo/
COPY data/ /app/data/
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app
EXPOSE 5000

ENTRYPOINT ["/app/entrypoint.sh"]
