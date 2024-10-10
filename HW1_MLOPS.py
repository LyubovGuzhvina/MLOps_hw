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

from datetime import datetime, timedelta
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
import numpy as np
import pickle
import json
import logging
import io

BUCKET = Variable.get("S3_BUCKET")

DEFAULT_ARGS = {
    "owner" : "Lyubov Guzhvina",
    "retry" : 3,
    "retry_delay" : timedelta(minutes = 1)
}

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

def create_dag(dag_id: str, m_name: Literal["random_forest", "linear_regression", "desicion_tree"]):

    ####### DAG STEPS #######

    def init(m_name: Literal["random_forest", "linear_regression", "desicion_tree"]) -> Dict[str, Any]:
        metrics = {}
        metrics["start_time"] = str(datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
        metrics["model_name"] = m_name

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
            key=f'LyubovGuzhvina/{m_name}/datasets/california_housing.pkl',
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
        file = s3_hook.download_file(key=f'LyubovGuzhvina/{m_name}/datasets/california_housing.pkl', bucket_name=BUCKET)
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
                key=f'LyubovGuzhvina/{m_name}/datasets/{name}.pkl',
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

        metrics["model_training_start_time"] = str(datetime.now().strftime("%d.%m.%Y %H:%M:%S"))

        s3_hook = S3Hook("s3_connection")
        data = {}
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            file = s3_hook.download_file(
                key=f'LyubovGuzhvina/{m_name}/datasets/{name}.pkl',
                bucket_name=BUCKET,
            )
            data[name] = pd.read_pickle(file)
        
        model = models[m_name]
        model.fit(data["X_train"], data["y_train"])
        prediction = model.predict(data["X_test"])

        metrics["model_training_end_time"] = str(datetime.now().strftime("%d.%m.%Y %H:%M:%S"))

        result = {}
        result["r2_score"] = r2_score(data["y_test"], prediction)
        result["rmse"] = mean_squared_error(data["y_test"], prediction) ** 0.5
        result["mae"] = median_absolute_error(data["y_test"], prediction)    

        metrics["model_metrics"] = result

        return metrics
        
        _LOG.info("Model trained.")

    def save_results(**kwargs) -> None:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids = "train_model")

        s3_hook = S3Hook("s3_connection")

        filebuffer = io.BytesIO()
        filebuffer.write(json.dumps(metrics).encode())
        filebuffer.seek(0)
        
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f'LyubovGuzhvina/{m_name}/results/metrics.json',
            bucket_name=BUCKET,
            replace=True,
        )
        _LOG.info("Success!")

    return init, get_data, prepare_data, train_model, save_results

    ####### INIT DAG #######

for model_name in models.keys():
    dag = DAG(dag_id = f"Lyubov_Guzhvina_{model_name}",
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS)

 
    with dag:

        init, get_data, prepare_data, train_model, save_results = create_dag(f"Lyubov_Guzhvina_{model_name}", model_name)
        
        task_init = PythonOperator(task_id="init", python_callable=init, dag=dag, op_kwargs = {"m_name" : f"{model_name}"})

        task_get_data = PythonOperator(task_id="get_data", python_callable=get_data, dag=dag)

        task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=dag)

        task_train_model = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)

        task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=dag)

        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results


for model_name in models.keys():
    create_dag(f"Lyubov_Guzhvina_{model_name}", model_name)