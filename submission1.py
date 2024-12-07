# %% [code]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

### Import data

data = pd.read_parquet("/kaggle/input/msdb-2024/train.parquet")
data.head()

### Data Preprocessing

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna

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

def preprocess_datetime(data):
    data = encode_dates(data)
    return data

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
# A type (class) which will be used to create wrapper objects for y_pred


def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


def get_train_data(path="/kaggle/input/msdb-2024/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

X,y = get_train_data()

def train_test_split_temporal(X, y, delta_threshold="30 days"):
    
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = (X["date"] <= cutoff_date)
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    return X_train, y_train, X_valid, y_valid

X_train, y_train, X_valid, y_valid = train_test_split_temporal(X, y)

print(
    f'Train: n_samples={X_train.shape[0]},  {X_train["date"].min()} to {X_train["date"].max()}'
)
print(
    f'Valid: n_samples={X_valid.shape[0]},  {X_valid["date"].min()} to {X_valid["date"].max()}'
)

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

target_column = "log_bike_count"

# Define features and target
X_train = X_train.drop(columns=["site_id","counter_id", "counter_technical_id", "counter_installation_date", "coordinates"])
X_valid = X_valid.drop(columns=["site_id","counter_id", "counter_technical_id", "counter_installation_date", "coordinates"])

columns_to_drop = ["site_id", "counter_id", "counter_technical_id", "counter_installation_date", "coordinates"]

X_train_preprocessed = preprocess_datetime(X_train)
X_valid_preprocessed = preprocess_datetime(X_valid)

# Scale numeric features
scaler = StandardScaler()
numeric_features = ["latitude", "longitude"]
X_train_preprocessed[numeric_features] = scaler.fit_transform(X_train_preprocessed[numeric_features])
X_valid_preprocessed[numeric_features] = scaler.transform(X_valid_preprocessed[numeric_features])

# Define categorical features for CatBoost
categorical_columns = ["counter_name", "site_name"]
cat_features_indices = [X_train_preprocessed.columns.get_loc(col) for col in categorical_columns]

# ------------------- CatBoost Tuning -------------------
def catboost_objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 1, 10),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
    }
    if params["bootstrap_type"] != "Bayesian":
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)

    model = CatBoostRegressor(
        cat_features=cat_features_indices,
        random_seed=42,
        verbose=0,
        **params
    )
    model.fit(X_train_preprocessed, y_train, eval_set=(X_valid_preprocessed, y_valid), early_stopping_rounds=50)
    preds = model.predict(X_valid_preprocessed)
    return mean_squared_error(y_valid, preds, squared=False)

catboost_study = optuna.create_study(direction="minimize")
catboost_study.optimize(catboost_objective, n_trials=10)
best_catboost_params = catboost_study.best_params

cat_model = CatBoostRegressor(cat_features=cat_features_indices, random_seed=42, verbose=100, **best_catboost_params)
cat_model.fit(X_train_preprocessed, y_train, eval_set=(X_valid_preprocessed, y_valid), early_stopping_rounds=50)
cat_preds_valid = cat_model.predict(X_valid_preprocessed)
cat_rmse = mean_squared_error(y_valid, cat_preds_valid, squared=False)
print(f"CatBoost Validation RMSE: {cat_rmse:.5f}")

# ------------------- XGBoost Tuning -------------------
def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
    }
    model = XGBRegressor(random_state=42, enable_categorical=True, **params)  # Enable categorical support
    model.fit(X_train_preprocessed, y_train)
    preds = model.predict(X_valid_preprocessed)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    print(f"XGBoost Trial {trial.number} RMSE: {rmse:.5f}")
    return rmse

xgb_study = optuna.create_study(direction="minimize")
xgb_study.optimize(xgb_objective, n_trials=10)
best_xgb_params = xgb_study.best_params

xg_model = XGBRegressor(random_state=42, enable_categorical=True, **best_xgb_params)  # Enable categorical support
xg_model.fit(X_train_preprocessed, y_train)
xgb_preds_valid = xg_model.predict(X_valid_preprocessed)

xgb_rmse = mean_squared_error(y_valid, xgb_preds_valid, squared=False)
print(f"XGBoost Validation RMSE: {xgb_rmse:.5f}")


# ------------------- LightGBM Tuning -------------------
def lgbm_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    model = LGBMRegressor(random_state=42, **params)
    model.fit(X_train_preprocessed, y_train)
    preds = model.predict(X_valid_preprocessed)
    return mean_squared_error(y_valid, preds, squared=False)

lgbm_study = optuna.create_study(direction="minimize")
lgbm_study.optimize(lgbm_objective, n_trials=10)
best_lgbm_params = lgbm_study.best_params

lgbm_model = LGBMRegressor(random_state=42, **best_lgbm_params)
lgbm_model.fit(X_train_preprocessed, y_train)
lgbm_preds_valid = lgbm_model.predict(X_valid_preprocessed)

lgbm_rmse = mean_squared_error(y_valid, lgbm_preds_valid, squared=False)
print(f"LightGBM Validation RMSE: {lgbm_rmse:.5f}")


# ------------------- Stacking Model -------------------
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Combine predictions from the base models for validation
stacked_features_valid = np.vstack((cat_preds_valid, xgb_preds_valid, lgbm_preds_valid)).T

# Train the meta-model (Ridge regression) using the validation predictions
meta_model = Ridge(alpha=1.0)
meta_model.fit(stacked_features_valid, y_valid)

# Predict using the meta-model (stacking) on validation
final_preds_valid = meta_model.predict(stacked_features_valid)

# Calculate RMSE for the stacked model
stack_rmse = mean_squared_error(y_valid, final_preds_valid, squared=False)
print(f"Stacked Model - Validation RMSE: {stack_rmse:.5f}")

# ------------------- Test Predictions and Submission -------------------
test_data = pd.read_parquet("/kaggle/input/msdb-2024/final_test.parquet")
test_data = test_data.drop(columns=columns_to_drop)

# Preprocess the test data
X_test_preprocessed = preprocess_datetime(test_data)
X_test_preprocessed[numeric_features] = scaler.transform(X_test_preprocessed[numeric_features])

# Generate predictions from the base models on the test set
cat_preds_test = cat_model.predict(Pool(data=X_test_preprocessed, cat_features=cat_features_indices))
xgb_preds_test = xg_model.predict(X_test_preprocessed)
lgbm_preds_test = lgbm_model.predict(X_test_preprocessed)

# Combine test predictions for the stacked model
stacked_features_test = np.vstack((cat_preds_test, xgb_preds_test, lgbm_preds_test)).T

# Generate final predictions using the meta-model (stacked)
final_preds_test = meta_model.predict(stacked_features_test)

# Create the submission file
submission = pd.DataFrame(
    {"Id": np.arange(len(final_preds_test)), "log_bike_count": final_preds_test}
)
submission.to_csv("submission.csv", index=False)