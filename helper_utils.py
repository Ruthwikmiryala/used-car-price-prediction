import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def show_results(y_true, y_pred, name="Model"):
    """Print the main regression metrics (in rupees)."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pct = np.median(np.abs(y_true - y_pred) / y_true) * 100
    print(f"{name}: RMSE {rmse:,.0f} | MAE {mae:,.0f} | R2 {r2:.3f} | median error {pct:.1f}%")


def plot_actual_vs_predicted(y_true, y_pred):
    """Points close to the line mean good predictions."""
    plt.scatter(y_true, y_pred, s=8, alpha=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("Actual vs predicted")
    plt.show()


def plot_importance(model, columns, top=10):
    """Bar chart of the most important features."""
    imp = pd.Series(model.feature_importances_, index=columns).nlargest(top)
    imp.sort_values().plot(kind="barh", title="Top features")
    plt.tight_layout()
    plt.show()
