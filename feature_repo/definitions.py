"""Definicje Feast dla cech wina (sekcja 6, Opcja A — Feature Store).

Encja: wine (join_key = wine_id). Feature view: wine_features (11 cech fizykochemicznych).
Źródło offline: data/wine_features.parquet (generowane przez generate_features.py).

Po edycji uruchom:  feast apply
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32
from feast.value_type import ValueType

# encja — pojedyncza próbka wina
wine = Entity(
    name="wine",
    join_keys=["wine_id"],
    value_type=ValueType.INT64,
    description="Identyfikator próbki wina (indeks wiersza w zbiorze).",
)

# źródło danych offline
wine_source = FileSource(
    name="wine_source",
    path="data/wine_features.parquet",
    timestamp_field="event_timestamp",
)

# lista 11 cech (kontrakt danych)
_FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

wine_features = FeatureView(
    name="wine_features",
    entities=[wine],
    ttl=timedelta(days=3650),  # szerokie okno — obejmuje stały event_timestamp danych demo
    schema=[Field(name=f, dtype=Float32) for f in _FEATURES],
    source=wine_source,
    online=True,
    description="Właściwości fizykochemiczne wina do predykcji jakości.",
)
