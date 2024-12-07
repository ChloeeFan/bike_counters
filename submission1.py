import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna

# Load train data
data = pd.read_parquet("/kaggle/input/msdb-2024/train.parquet")

# Feature Engineering for Date
def encode_dates(X):
    X = X.copy()
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["hour"] = X["date"].dt.hour
    X["weekday"] = X["date"].dt.weekday
    X["hour_weekday"] = X["hour"] + X["weekday"] * 24  # Interaction term
    # Add cyclic features
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    X["weekday_sin"] = np.sin(2 * np.pi * X["weekday"] / 7)
    X["weekday_cos"] = np.cos(2 * np.pi * X["weekday"] / 7)
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
    return X.drop(columns=["date"])

# Preprocess datetime columns
def preprocess_datetime(data):
    return encode_dates(data)

# Temporal train-test split
def train_test_split_temporal(X, y, delta_threshold="30 days"):
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = X["date"] <= cutoff_date
    X_train, X_valid = X[mask], X[~mask]
    y_train, y_valid = y[mask], y[~mask]
    return X_train, y_train, X_valid, y_valid

# Prepare data
target_column = "log_bike_count"
X = data.drop(columns=["bike_count", target_column])
y = data[target_column].values
X_train, y_train, X_valid, y_valid = train_test_split_temporal(X, y)

# Drop irrelevant columns
columns_to_drop = ["site_id", "counter_id", "counter_technical_id", "counter_installation_date", "coordinates"]
X_train = X_train.drop(columns=columns_to_drop)
X_valid = X_valid.drop(columns=columns_to_drop)

# Preprocess features
X_train_preprocessed = preprocess_datetime(X_train)
X_valid_preprocessed = preprocess_datetime(X_valid)

# Scale numeric features
scaler = StandardScaler()
numeric_features = ["latitude", "longitude"]
X_train_preprocessed[numeric_features] = scaler.fit_transform(X_train_preprocessed[numeric_features])
X_valid_preprocessed[numeric_features] = scaler.transform(X_valid_preprocessed[numeric_features])

# Define categorical features
categorical_columns = ["counter_name", "site_name"]
cat_features_indices = [X_train_preprocessed.columns.get_loc(col) for col in categorical_columns]

# Train CatBoost model
cat_model = CatBoostRegressor(
    cat_features=cat_features_indices, random_seed=42, verbose=100, iterations=1000
)
cat_model.fit(X_train_preprocessed, y_train, eval_set=(X_valid_preprocessed, y_valid), early_stopping_rounds=50)
cat_preds_valid = cat_model.predict(X_valid_preprocessed)
cat_rmse = mean_squared_error(y_valid, cat_preds_valid, squared=False)
print(f"CatBoost Validation RMSE: {cat_rmse:.5f}")

# Train XGBoost model
xg_model = XGBRegressor(random_state=42, enable_categorical=True, n_estimators=200, learning_rate=0.05)
xg_model.fit(X_train_preprocessed, y_train)
xgb_preds_valid = xg_model.predict(X_valid_preprocessed)
xgb_rmse = mean_squared_error(y_valid, xgb_preds_valid, squared=False)
print(f"XGBoost Validation RMSE: {xgb_rmse:.5f}")

# Train LightGBM model
lgbm_model = LGBMRegressor(random_state=42, n_estimators=200, learning_rate=0.05)
lgbm_model.fit(X_train_preprocessed, y_train)
lgbm_preds_valid = lgbm_model.predict(X_valid_preprocessed)
lgbm_rmse = mean_squared_error(y_valid, lgbm_preds_valid, squared=False)
print(f"LightGBM Validation RMSE: {lgbm_rmse:.5f}")

# Stacking model
stacked_features_valid = np.vstack((cat_preds_valid, xgb_preds_valid, lgbm_preds_valid)).T
meta_model = Ridge(alpha=1.0)
meta_model.fit(stacked_features_valid, y_valid)
final_preds_valid = meta_model.predict(stacked_features_valid)
stack_rmse = mean_squared_error(y_valid, final_preds_valid, squared=False)
print(f"Stacked Model - Validation RMSE: {stack_rmse:.5f}")

# Test data predictions
test_data = pd.read_parquet("/kaggle/input/msdb-2024/final_test.parquet")
test_data = test_data.drop(columns=columns_to_drop)
X_test_preprocessed = preprocess_datetime(test_data)
X_test_preprocessed[numeric_features] = scaler.transform(X_test_preprocessed[numeric_features])

cat_preds_test = cat_model.predict(Pool(data=X_test_preprocessed, cat_features=cat_features_indices))
xgb_preds_test = xg_model.predict(X_test_preprocessed)
lgbm_preds_test = lgbm_model.predict(X_test_preprocessed)

stacked_features_test = np.vstack((cat_preds_test, xgb_preds_test, lgbm_preds_test)).T
final_preds_test = meta_model.predict(stacked_features_test)

# Create submission file
submission = pd.DataFrame({"Id": np.arange(len(final_preds_test)), "log_bike_count": final_preds_test})
submission.to_csv("submission.csv", index=False)
