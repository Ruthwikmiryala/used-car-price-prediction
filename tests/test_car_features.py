import pandas as pd

from CarFeatures import add_features, clean_data, encode


def sample():
    return pd.DataFrame({
        "name": ["Maruti Swift VDI", "Hyundai i20 Asta", "Maruti Swift VDI"],
        "year": [2014, 2018, 2014],
        "selling_price": [450000, 700000, 450000],
        "km_driven": [145500, 40000, 145500],
        "fuel": ["Diesel", "Petrol", "Diesel"],
        "owner": ["First Owner", "First Owner", "First Owner"],
        "mileage": ["23.4 kmpl", "18.6 kmpl", "23.4 kmpl"],
        "engine": ["1248 CC", "1197 CC", "1248 CC"],
        "max_power": ["74 bhp", "81.8 bhp", "74 bhp"],
        "torque": ["190Nm", "114Nm", "190Nm"],
        "seats": [5.0, 5.0, 5.0],
    })


def test_clean_data():
    df = clean_data(sample())
    assert len(df) == 2                      # duplicate removed
    assert "torque" not in df.columns
    assert df["mileage"].tolist() == [23.4, 18.6]
    assert df["engine"].tolist() == [1248, 1197]


def test_add_features():
    df = add_features(clean_data(sample()))
    assert df["brand"].tolist() == ["Maruti", "Hyundai"]
    assert df["car_age"].tolist() == [4, 0]
    assert "name" not in df.columns and "year" not in df.columns


def test_encode_gives_numbers_only():
    data = encode(add_features(clean_data(sample())))
    assert all(pd.api.types.is_numeric_dtype(t) for t in data.dtypes)
