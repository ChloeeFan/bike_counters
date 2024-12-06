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


### XGBoosting

#### Parameter Tuning

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the date encoding function
def encode_dates(X):
    X = X.copy()
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["hour"] = X["date"].dt.hour
    X["weekday"] = X["date"].dt.weekday
    X["hour_weekday"] = X["hour"] + X["weekday"] * 24  # Interaction term
    return X.drop(columns=["date"])

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

# Define the parameter grid for tuning
param_distributions = {
    "regressor__n_estimators": [100, 200, 300],
    "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "regressor__max_depth": [3, 5, 7, 10],
    "regressor__min_child_weight": [1, 3, 5],
    "regressor__subsample": [0.6, 0.8, 1.0],
    "regressor__colsample_bytree": [0.6, 0.8, 1.0],
    "regressor__gamma": [0, 0.1, 0.2, 0.3]
}

# Use RandomizedSearchCV for tuning
random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=50,  # Number of random samples to try
    cv=3,  # 3-fold cross-validation
    scoring="neg_root_mean_squared_error",  # Metric for evaluation
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit the model with randomized search
random_search.fit(X_train, y_train)

# Best parameters and model
print("Best parameters:", random_search.best_params_)
best_model = random_search.best_estimator_

# Evaluate RMSE for Train and Validation sets
print(
    f"Train set, RMSE={mean_squared_error(y_train, best_model.predict(X_train), squared=False):.5f}"
)
print(
    f"Valid set, RMSE={mean_squared_error(y_valid, best_model.predict(X_valid), squared=False):.5f}"
)


# Extract the best parameters
best_params = random_search.best_params_

# Train the model with the best parameters
final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        subsample=best_params['regressor__subsample'],
        n_estimators=best_params['regressor__n_estimators'],
        min_child_weight=best_params['regressor__min_child_weight'],
        max_depth=best_params['regressor__max_depth'],
        learning_rate=best_params['regressor__learning_rate'],
        gamma=best_params['regressor__gamma'],
        colsample_bytree=best_params['regressor__colsample_bytree'],
        random_state=42,
        n_jobs=-1
    ))
])

# Fit the model on the full training set
final_model.fit(X_train, y_train)

# Evaluate RMSE for Train and Validation sets
train_rmse = mean_squared_error(y_train, final_model.predict(X_train), squared=False)
valid_rmse = mean_squared_error(y_valid, final_model.predict(X_valid), squared=False)

print(f"Final Model - Train set RMSE: {train_rmse:.5f}")
print(f"Final Model - Valid set RMSE: {valid_rmse:.5f}")


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

# Train CatBoostRegressor
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=5,
    random_seed=42,
    cat_features=cat_features_indices,
    verbose=100
)
cat_model.fit(
    X_train_preprocessed, y_train,
    eval_set=(X_valid_preprocessed, y_valid),
    early_stopping_rounds=50
)

# Evaluate Train and Validation RMSE
train_rmse = mean_squared_error(y_train, cat_model.predict(X_train_preprocessed), squared=False)
valid_rmse = mean_squared_error(y_valid, cat_model.predict(X_valid_preprocessed), squared=False)

print(f"CatBoost - Train set, RMSE={train_rmse:.5f}")
print(f"CatBoost - Valid set, RMSE={valid_rmse:.5f}")


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


### Stacking(Catboost and Xgboost)

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


final_cat_model.fit(X_train_preprocessed, y_train, eval_set=(X_valid_preprocessed, y_valid), early_stopping_rounds=50)
final_cat_model.predict(X_valid_preprocessed)

final_xg_model.fit(X_train, y_train)
final_xg_model.predict(X_valid)

cat_preds = final_cat_model.predict(X_valid_preprocessed)
xgb_preds = final_xg_model.predict(X_valid)

# Combine predictions as features for stacking
stacked_features_full = np.vstack((cat_preds, xgb_preds)).T

# Split stacked features for meta-model training and validation
X_train_meta, X_valid_meta, y_train_meta, y_valid_meta = train_test_split(
    stacked_features_full, y_valid, test_size=0.2, random_state=42
)

# Train the Ridge regression model as the meta-model
meta_model = Ridge(alpha=1.0)
meta_model.fit(X_train_meta, y_train_meta)

# Predict on the validation meta-set
meta_preds = meta_model.predict(X_valid_meta)

# Calculate and print the RMSE for the stacked model
meta_rmse = mean_squared_error(y_valid_meta, meta_preds, squared=False)
print(f"Stacked Ensemble Model - Validation RMSE: {meta_rmse:.5f}")

test = pd.read_parquet("/kaggle/input/msdb-2024/final_test.parquet")

test.head()

test_data = test.drop(columns=["counter_id", "counter_technical_id", "counter_installation_date", "coordinates"])

test_data

X_test = test_data
X_test_preprocessed = preprocess_datetime(test_data)

# Prepare CatBoost Pool for prediction
cat_features_indices = [X_test_preprocessed.columns.get_loc(col) for col in categorical_columns]  # Indices of categorical columns
test_pool = Pool(data=X_test_preprocessed, cat_features=cat_features_indices)

date_columns = ["date"]
categorical_columns = ["counter_name", "site_name"]
numeric_columns = ["latitude", "longitude"]

categorical_columns

from catboost import Pool

# Prepare CatBoost Pool for prediction
cat_features_indices = [X_test_preprocessed.columns.get_loc(col) for col in categorical_columns]  # Indices of categorical columns
test_pool = Pool(data=X_test_preprocessed, cat_features=cat_features_indices)


# Predict using CatBoost
cat_preds_test = final_cat_model.predict(test_pool)
xgb_preds = final_xg_model.predict(test_data)

# Ensure the test data preprocessing aligns for both models
assert len(X_test_preprocessed) == len(X_test), "Mismatch in preprocessed test data sizes!"

# Generate predictions from CatBoost and XGBoost
cat_preds_test = final_cat_model.predict(Pool(data=X_test_preprocessed, cat_features=cat_features_indices))
xgb_preds_test = final_xg_model.predict(X_test)

# Ensure predictions have the same length
assert len(cat_preds_test) == len(xgb_preds_test), "Mismatch in predictions sizes!"

# Combine predictions as features for the stacked model
stacked_features_test = np.vstack((cat_preds_test, xgb_preds_test)).T

# Generate final predictions using the already trained meta-model
final_predictions = meta_model.predict(stacked_features_test)



import pandas as pd


results = pd.DataFrame(
    dict(
        Id=np.arange(final_predictions.shape[0]),  # Create an Id column from 0 to len(final_predictions) - 1
        log_bike_count=final_predictions,  # Assign predictions to the log_bike_count column
    )
)

# Save the submission file
results.to_csv("submission.csv", index=False)