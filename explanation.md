# Explanation

**Problem.** Predict `selling_price` (rupees) from the car's details. This is regression.

**Cleaning (`clean_data`).** Remove duplicate listings (so the same car is not in both train and test); drop `torque` (messy units); `mileage`, `engine` and `max_power` are text like "23.4 kmpl", so extract the number; drop rows with missing values.

**Features (`add_features`).** `brand` (first word of the name) and `car_age` (newest year minus the car's year) replace `name` and `year`.

**Encoding (`encode`).** `pd.get_dummies` turns text columns into 0/1 columns.

**Log of the price.** Prices are heavily right-skewed. The model learns `log(1 + price)` and predictions are converted back with `np.expm1`, so all reported errors are in rupees.

**Split.** 80% train / 20% test; the test set is used once, at the end.

**Models.** Ridge regression is the baseline. XGBoost builds small trees one after another, each correcting the mistakes of the previous ones. `GridSearchCV` tries a few settings with 5-fold cross-validation.

**Reading the results.** RMSE and MAE are in rupees; R2 is the share of price variation explained; median error % is the typical percentage miss. The actual-vs-predicted plot should hug the diagonal line.
