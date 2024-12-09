# %% [code]
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    TimeSeriesSplit,
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import optuna

# Visualization Imports (Not Used, but Kept for Future Use)
import matplotlib.pyplot as plt
import seaborn as sns

# File Path Handling
from pathlib import Path


### Import data

data = pd.read_parquet("/kaggle/input/msdb-2024/train.parquet")
external_data = pd.read_csv("/kaggle/input/msdb-2024/external_data.csv")


# Ensure 'date' columns are in datetime format
data["date"] = pd.to_datetime(data["date"])
external_data["date"] = pd.to_datetime(external_data["date"])

# Select relevant columns from the weather dataset
# Example: Keep 'date', 't' (temperature), 'u' (humidity), 'rr1' (rainfall in 1 hour)
weather_data = external_data[["date", "t", "u", "rr1"]]  # Adjust based on relevant columns

# Merge datasets on the 'date' column
merged_data = pd.merge(data, weather_data, on="date", how="left")
merged_data["t"].fillna(merged_data["t"].mean(), inplace=True)  # Fill missing temperature with the mean
merged_data["u"].fillna(merged_data["u"].median(), inplace=True)  # Fill missing humidity with the median
merged_data["rr1"].fillna(0, inplace=True)


data = merged_data


date_columns = ["date"]
categorical_columns = ["counter_name", "site_name"]
numeric_columns = ["latitude", "longitude","t", "u", "rr1"]

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


def get_train_data(data):
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

X,y = get_train_data(data)

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

### XGBoosting

#### Parameter Tuning

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.metrics import mean_squared_error
import numpy as np

import holidays


# Function to add holiday feature
def add_holiday_feature(X):
    X = X.copy()
    # Check if each date is a French holiday
    X["is_french_holiday"] = X["date"].apply(lambda x: 1 if x in french_holidays else 0)
    return X


def encode_dates(X):
    X = X.copy()
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["hour"] = X["date"].dt.hour
    X["weekday"] = X["date"].dt.weekday
    X["hour_weekday"] = X["hour"] + X["weekday"] * 24  # Interaction term
    X["is_french_holiday"] = X["date"].apply(lambda x: 1 if x in french_holidays else 0)
    return X.drop(columns=["date"])

# Initialize French holidays
french_holidays = holidays.France()

def preprocess_datetime(data):
    data = data.copy()
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["hour"] = data["date"].dt.hour
    data["weekday"] = data["date"].dt.weekday
    data["hour_weekday"] = data["hour"] + data["weekday"] * 24  # Interaction term
    data["is_french_holiday"] = data["date"].apply(lambda x: 1 if x in french_holidays else 0)
    return data.drop(columns=["date"])  # Drop the original datetime column

# Define preprocessors
date_encoder = FunctionTransformer(encode_dates)
categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ("date", date_encoder, ["date"]),
        ("cat", categorical_encoder, ["counter_name", "site_name"]),
    ],
    remainder="passthrough"
)

# Define the pipeline
pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(random_state=42, n_jobs=-1))
])



#### Hyperparameter Optimization

import optuna
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

def objective(trial):
    # Suggest hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
    }

    # Create a pipeline with the suggested parameters
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(random_state=42, **params))
    ])
    
    # Fit and evaluate
    pipe.fit(X_train, y_train)
    y_valid_pred = pipe.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_valid_pred, squared=False)
    return rmse

# Create and run the study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# Display the best parameters
print("Best parameters:", study.best_params)

# Train the model with the best parameters
best_params = study.best_params
final_xg_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(random_state=42, **best_params))
])


final_xg_model.fit(X_train, y_train)

# Evaluate Train and Validation RMSE
train_rmse = mean_squared_error(y_train, final_xg_model.predict(X_train), squared=False)
valid_rmse = mean_squared_error(y_valid, final_xg_model.predict(X_valid), squared=False)

print(f"Final XGboost Model - Train RMSE: {train_rmse:.5f}")
print(f"Final XGboost Model - Validation RMSE: {valid_rmse:.5f}")


### CatBoost

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# Function to preprocess date
def preprocess_datetime(data):
    data = data.copy()
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["hour"] = data["date"].dt.hour
    data["weekday"] = data["date"].dt.weekday
    data["hour_weekday"] = data["hour"] + data["weekday"] * 24  # Interaction term
    return data.drop(columns=["date"])  # Drop the original datetime column

# Apply preprocessing to the train and validation datasets
X_train_preprocessed = preprocess_datetime(X_train)
X_valid_preprocessed = preprocess_datetime(X_valid)

# Specify categorical feature indices (relative to preprocessed data)
categorical_features = ["counter_name", "site_name"]
cat_features_indices = [X_train_preprocessed.columns.get_loc(col) for col in categorical_features]


import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

def objective(trial):
    # Suggest hyperparameters for CatBoost
    params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 1, 10),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
    }

    # Only add `subsample` if bootstrap type is not Bayesian
    if params["bootstrap_type"] != "Bayesian":
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)

    # Add bagging temperature for Bayesian bootstrap
    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)

    # Create and train the CatBoost model
    model = CatBoostRegressor(
        cat_features=cat_features_indices,
        random_seed=42,
        verbose=0,
        **params
    )
    model.fit(X_train_preprocessed, y_train, eval_set=(X_valid_preprocessed, y_valid), early_stopping_rounds=50)

    # Calculate RMSE for validation set
    valid_rmse = mean_squared_error(y_valid, model.predict(X_valid_preprocessed), squared=False)
    return valid_rmse

# Create an Optuna study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10, timeout=3600)  # Run 50 trials or stop after 1 hour

# Display the best parameters and RMSE
print("Best parameters:", study.best_params)
print(f"Best Validation RMSE: {study.best_value:.5f}")

# Train final model with the best parameters
best_params = study.best_params
final_cat_model = CatBoostRegressor(
    cat_features=cat_features_indices,
    random_seed=42,
    verbose=100,
    **best_params
)
final_cat_model.fit(X_train_preprocessed, y_train, eval_set=(X_valid_preprocessed, y_valid), early_stopping_rounds=50)

# Evaluate Train and Validation RMSE
train_rmse = mean_squared_error(y_train, final_cat_model.predict(X_train_preprocessed), squared=False)
valid_rmse = mean_squared_error(y_valid, final_cat_model.predict(X_valid_preprocessed), squared=False)

print(f"Final CatBoost Model - Train RMSE: {train_rmse:.5f}")
print(f"Final CatBoost Model - Validation RMSE: {valid_rmse:.5f}")

import optuna
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# Define the objective function for Optuna
def lgb_objective(trial):
    # Define parameter space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "random_state": 42
    }

    # Initialize the LightGBM model
    lgb_model = lgb.LGBMRegressor(**params)
    
    # Fit the model with early stopping
    lgb_model.fit(
        X_train_preprocessed, y_train,
        eval_set=[(X_valid_preprocessed, y_valid)],
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Evaluate RMSE on the validation set
    preds = lgb_model.predict(X_valid_preprocessed)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    return rmse

# Create Optuna study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(lgb_objective, n_trials=30)

# Get the best parameters
best_lgb_params = study.best_params
print("Best LightGBM Parameters:", best_lgb_params)

# Train the final LightGBM model with the best parameters
final_lgb_model = lgb.LGBMRegressor(**best_lgb_params)
final_lgb_model.fit(
    X_train_preprocessed, y_train,
    eval_set=[(X_valid_preprocessed, y_valid)],
    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(10)]
)

# Validate the model
lgb_valid_preds = final_lgb_model.predict(X_valid_preprocessed)
lgb_rmse = mean_squared_error(y_valid, lgb_valid_preds, squared=False)
print(f"Final LightGBM Model - Validation RMSE: {lgb_rmse:.5f}")



### Stacking(Catboost and Xgboost)

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from catboost import Pool
from sklearn.model_selection import GridSearchCV


final_cat_model.fit(X_train_preprocessed, y_train, eval_set=(X_valid_preprocessed, y_valid), early_stopping_rounds=50)
final_cat_model.predict(X_valid_preprocessed)

final_xg_model.fit(X_train, y_train)
final_xg_model.predict(X_valid)

final_lgb_model.fit(X_train_preprocessed, y_train, 
                    eval_set=[(X_valid_preprocessed, y_valid)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])


lgb_preds = final_lgb_model.predict(X_valid_preprocessed)
cat_preds = final_cat_model.predict(X_valid_preprocessed)

xgb_preds = final_xg_model.predict(X_valid)

stacked_features_full = np.vstack((cat_preds, xgb_preds, lgb_preds)).T

# Split stacked features for meta-model training and validation
X_train_meta, X_valid_meta, y_train_meta, y_valid_meta = train_test_split(
    stacked_features_full, y_valid, test_size=0.2, random_state=42
)

# Define Ridge parameter grid
ridge_param_grid = {
    "alpha": [0.01, 0.1, 1, 10, 100]  # Regularization strength
}

# Perform GridSearchCV for Ridge tuning
ridge_grid_search = GridSearchCV(
    estimator=Ridge(),
    param_grid=ridge_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5,  # 5-fold cross-validation
    verbose=2,
    n_jobs=-1
)

# Fit Ridge with meta-training data
ridge_grid_search.fit(X_train_meta, y_train_meta)

# Best parameters and model
print("Best Ridge parameters:", ridge_grid_search.best_params_)
best_ridge_model = ridge_grid_search.best_estimator_

# Predict on validation meta-set
meta_preds = best_ridge_model.predict(X_valid_meta)

# Calculate and print the RMSE for the stacked model
meta_rmse = mean_squared_error(y_valid_meta, meta_preds, squared=False)
print(f"Stacked Ensemble Model - Validation RMSE: {meta_rmse:.5f}")


test = pd.read_parquet("/kaggle/input/msdb-2024/final_test.parquet")

test_data = test.drop(columns=["site_id","counter_id", "counter_technical_id", "counter_installation_date", "coordinates"])

# Ensure 'date' columns are in datetime format
test_data["date"] = pd.to_datetime(test_data["date"])
external_data["date"] = pd.to_datetime(external_data["date"])

# Select relevant columns from the weather dataset
# Example: Keep 'date', 't' (temperature), 'u' (humidity), 'rr1' (rainfall in 1 hour)
weather_data = external_data[["date", "t", "u", "rr1"]]  # Adjust based on relevant columns

# Merge datasets on the 'date' column
merged_data = pd.merge(test_data, weather_data, on="date", how="left")
merged_data["t"].fillna(merged_data["t"].mean(), inplace=True)  # Fill missing temperature with the mean
merged_data["u"].fillna(merged_data["u"].median(), inplace=True)  # Fill missing humidity with the median
merged_data["rr1"].fillna(0, inplace=True)

test_data = merged_data

X_test = test_data
X_test_preprocessed = preprocess_datetime(test_data)

# Generate predictions from LightGBM
lgb_preds_test = final_lgb_model.predict(X_test_preprocessed)

# Prepare CatBoost Pool for prediction
cat_features_indices = [X_test_preprocessed.columns.get_loc(col) for col in categorical_columns]
test_pool = Pool(data=X_test_preprocessed, cat_features=cat_features_indices)

# Generate predictions from CatBoost
cat_preds_test = final_cat_model.predict(test_pool)

# Generate predictions from XGBoost
xgb_preds_test = final_xg_model.predict(X_test)

# Generate predictions from LightGBM
lgb_preds_test = final_lgb_model.predict(X_test_preprocessed)

# Ensure predictions from all models align
assert len(cat_preds_test) == len(xgb_preds_test) == len(lgb_preds_test), "Mismatch in prediction sizes!"

# Combine predictions as features for stacking
stacked_features_test = np.vstack((cat_preds_test, xgb_preds_test, lgb_preds_test)).T

# Generate final predictions using the already trained Ridge meta-model
final_predictions = best_ridge_model.predict(stacked_features_test)

# Create the submission DataFrame
results = pd.DataFrame(
    dict(
        Id=np.arange(final_predictions.shape[0]),  # Create an Id column from 0 to len(final_predictions) - 1
        log_bike_count=final_predictions,  # Assign predictions to the log_bike_count column
    )
)

# Save the submission file
results.to_csv("submission.csv", index=False)