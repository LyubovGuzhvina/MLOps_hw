from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from typing import Any, Dict, Literal

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle
import json
import logging
import io

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
import os

os.getenv("MLFLOW_TRACKING_URI", "No env")
mlflow.get_registry_uri() 

exp_name = "Lyubov_Guzhvina"
my_experiment = mlflow.create_experiment(name = exp_name, artifact_location = 's3://lubguzh/mlflow')

my_experiment

mlflow.search_experiments(
    filter_string="name = 'Lyubov_Guzhvina'"
)

mlflow.set_experiment(experiment_name = exp_name)

mlflow.search_experiments()

model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))

housing = fetch_california_housing(as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(housing['data'], housing['target'])
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

# scaler = StandardScaler()
# X_train_fitted = pd.DataFrame(scaler.fit_transform(X_train))
# X_test_fitted = pd.DataFrame(scaler.transform(X_test))
# X_val_fitted = pd.DataFrame(scaler.transform(X_val))

with mlflow.start_run(run_name="@nokeryy", experiment_id = my_experiment, description = "parent") as parent_run:
    for model_name in models:
        with mlflow.start_run(run_name=model_name, experiment_id=my_experiment, nested=True) as child_run:
            model = models[model_name]

            model.fit(pd.DataFrame(X_train), y_train)
            prediction = model.predict(X_val)

            eval_df = X_val.copy()
            eval_df["target"] = y_val
        
            signature = infer_signature(X_test, prediction)
            model_info = mlflow.sklearn.log_model(model, "linreg", signature=signature, 
                                                  registered_model_name=f"sk-learn-{model_name}-reg-model")
            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="target",
                model_type="regressor",
                evaluators=["default"],
            )