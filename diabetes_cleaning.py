import polars as pl
import numpy as np

df = pl.read_csv('data/diabetes/diabetes_dataset.csv')

df = df.with_columns(
    (pl.col("Fasting_Blood_Glucose") > 125).cast(pl.Int8).alias("Outcome")
)
df = df.drop("")
labels = df["Outcome"]
features = df.drop("Outcome")

nary_cols = ['Smoking_Status', 'Alcohol_Consumption', 'Physical_Activity_Level', 'Ethnicity', 'Sex']


features = features.to_dummies(columns=nary_cols)

features = features.with_columns([
    ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c) for c in features.columns
])

x = features.to_numpy().astype(np.float32)
y = labels.to_numpy().astype(int)

test_percent = .20

