import pandas as pd

t = pd.read_csv(
    "competitions/geoai-aquaculture-pond-identification-challenge/data/processed/features_train.csv"
)
print("shape:", t.shape)
print("lon present:", "lon" in t.columns)
print("lat present:", "lat" in t.columns)
print("Longitude present:", "Longitude" in t.columns)
print("Latitude present:", "Latitude" in t.columns)
print("Columns:", list(t.columns[:10]), "...")
