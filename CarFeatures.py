import pandas as pd


def clean_data(df):
    """Fix the text columns ("23.4 kmpl" -> 23.4) and drop bad rows."""
    df = df.drop_duplicates().drop(columns="torque")
    for col in ["mileage", "engine", "max_power"]:
        number = df[col].astype(str).str.extract(r"(\d+\.?\d*)", expand=False)
        df[col] = pd.to_numeric(number, errors="coerce")
    return df.dropna().reset_index(drop=True)


def add_features(df):
    """Brand from the car name, and car age instead of year."""
    df = df.copy()
    df["brand"] = df["name"].str.split().str[0]
    df["car_age"] = df["year"].max() - df["year"]
    return df.drop(columns=["name", "year"])


def encode(df):
    """Turn text columns into 0/1 columns."""
    return pd.get_dummies(df, drop_first=True, dtype=int)
