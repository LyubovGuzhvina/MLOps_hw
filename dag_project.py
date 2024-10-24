from airflow.models import DAG, Variable
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from typing import Any, Dict, Literal
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import os

from datetime import datetime, timedelta
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
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

BUCKET = Variable.get("S3_BUCKET")

DEFAULT_ARGS = {
    "owner" : "Lyubov Guzhvina",
    "retry" : 3,
    "retry_delay" : timedelta(minutes = 1)
}

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)
    os.getenv("MLFLOW_TRACKING_URI", "No env")

exp_name = "guzhvina_lub"

mlflow.search_experiments(
    filter_string="name = 'guzhvina_lub'"
)
mlflow.set_experiment(experiment_name = exp_name)

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"

model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

dag = DAG(dag_id = "guzhvina_lubov",
schedule_interval="0 1 * * *",
start_date=days_ago(2),
catchup=False,
tags=["mlops"],
default_args=DEFAULT_ARGS)

    ####### DAG STEPS #######
        
def init() -> Dict[str, Any]:
    metrics = {}
    configure_mlflow()
    with mlflow.start_run() as run:
        metrics['experiment_id'] = str(run.info.experiment_id)
        metrics['run_id'] = str(run.info.run_id)
    metrics["start_time"] = str(datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
    return metrics

    _LOG.info("Train pipeline started.")
    
def get_data(**kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids = "init")

    metrics["dataset_fecthing_start_time"] = str(datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
    
    housing = fetch_california_housing(as_frame=True)
    data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)

    s3_hook = S3Hook("s3_connection")
    filebuffer = io.BytesIO()
    data.to_pickle(filebuffer)
    filebuffer.seek(0)

    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key="LyubovGuzhvina_project/datasets/california_housing.pkl",
        bucket_name=BUCKET,
        replace=True,
    )

    metrics["dataset_fecthing_end_time"] = str(datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
    metrics["dataset_size"] = len(data)

    return metrics
    
    _LOG.info("Data downloaded.")

def prepare_data(**kwargs) -> Dict[str, Any]:

    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids = "get_data")

    metrics["data_preparing_start_time"] = str(datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
    
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(key=f'LyubovGuzhvina_project/datasets/california_housing.pkl', bucket_name=BUCKET)
    data = pd.read_pickle(file)

    X, y = data[FEATURES], data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    for name, data in zip(
        ["X_train", "X_test", "y_train", "y_test"],
        [X_train_fitted, X_test_fitted, y_train, y_test],
    ):
        filebuffer = io.BytesIO()
        pickle.dump(data, filebuffer)
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f'LyubovGuzhvina_project/datasets/{name}.pkl',
            bucket_name=BUCKET,
            replace=True,
        )

    metrics["data_preparing_end_time"] = str(datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
    metrics["feature_names"] = list(X.columns)

    return metrics
    
    _LOG.info("Data prepared.")

def train_model(**kwargs) -> Dict[str, Any]:
    
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids = "prepare_data")
    configure_mlflow()
    m_name = kwargs["model_name"]
    s3_hook = S3Hook("s3_connection")
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(
            key=f'LyubovGuzhvina_project/datasets/{name}.pkl',
            bucket_name=BUCKET,
        )
        data[name] = pd.read_pickle(file)
    
    model = models[m_name]
    metrics[f"{m_name}_training_start_time"] = str(datetime.now().strftime("%d.%m.%Y %H:%M:%S"))

    with mlflow.start_run(run_id=metrics['run_id']):
        model.fit(data["X_train"], data["y_train"])
        predictions = model.predict(data["X_test"])

        signature = infer_signature(X_test, prediction)
        model_info = mlflow.sklearn.log_model(model, "linreg", signature=signature, 
                                              registered_model_name=f"sk-learn-{model_name}-reg-model")
        
        metrics[f"{m_name}_training_end_time"] = str(datetime.now().strftime("%d.%m.%Y %H:%M:%S"))

        eval_df = X_test.copy()
        eval_df["target"] = y_test
        
        mlflow.evaluate(
            model=model_info.model_uri,
            data=eval_df,
            targets="target",
            model_type="regressor",
            evaluators=["default"],
        )
        
        mlflow.sklearn.log_model(model, artifact_path=model_name)
    
    return metrics
    
    _LOG.info("Model trained.")

def save_results(**kwargs) -> None:   
    ti = kwargs["ti"]
    # metrics = ti.xcom_pull(task_ids = ["train_random_forest", "train_linear_regression", "train_desicion_tree"])

    metrics_rf = ti.xcom_pull(task_ids="train_random_forest")
    metrics_lr = ti.xcom_pull(task_ids="train_linear_regression")
    metrics_dt = ti.xcom_pull(task_ids="train_decision_tree")
    
    
    metrics = {
        "random_forest": metrics_rf,
        "linear_regression": metrics_lr,
        "decision_tree": metrics_dt
    }
    

    s3_hook = S3Hook("s3_connection")

    filebuffer = io.BytesIO()
    filebuffer.write(json.dumps(metrics).encode())
    filebuffer.seek(0)
    
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key="LyubovGuzhvina_project/results/metrics.json",
        bucket_name=BUCKET,
        replace=True,
    )
    _LOG.info("Success!")


task_init = PythonOperator(task_id="init", python_callable=init, dag=dag, provide_context = True)

task_get_data = PythonOperator(task_id="get_data", python_callable=get_data, dag=dag, provide_context = True)

task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=dag, provide_context = True)

training_model_tasks = [PythonOperator(task_id=f"train_{model_name}", 
                                    python_callable=train_model, 
                                    dag=dag, provide_context = True,
                                    op_kwargs = {"model_name" : model_name})
                                    for model_name in models.keys()]

task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=dag, provide_context = True)


task_init >> task_get_data >> task_prepare_data >> training_model_tasks >> task_save_results