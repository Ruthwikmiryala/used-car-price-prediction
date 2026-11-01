# Used Car Price Prediction (XGBoost)

Predict the selling price (INR) of a used car.

- Dataset: [CarDekho vehicle dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) - put `Car details v3.csv` in this folder
- Model: XGBoost, compared with a Ridge regression baseline

## Files

| File | What it does |
|---|---|
| `CarFeatures.py` | `clean_data`, `add_features`, `encode` |
| `helper_utils.py` | `show_results`, `plot_actual_vs_predicted`, `plot_importance` |
| `Car_price_analysis.ipynb` | Data exploration |
| `Car_Price_Final.ipynb` | Model training and evaluation |
| `tests/test_car_features.py` | Unit tests |

## Run

```bash
pip install -r requirements.txt
pytest
jupyter lab        # open the notebooks from this folder
```

## Results

Fill in from your run of `Car_Price_Final.ipynb`:

| Model | RMSE | MAE | R2 | Median error % |
|---|---|---|---|---|
| Ridge | | | | |
| XGBoost | | | | |
