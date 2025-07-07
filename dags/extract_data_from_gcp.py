from airflow import DAG
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime
from datetime import timedelta
import pandas as pd
import sqlalchemy

def load_to_sql(file_path):
    conn = BaseHook.get_connection('postgres_default')  
    engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{conn.login}:{conn.password}@user-survival-prediction_28f81a-postgres-1:{conn.port}/{conn.schema}")
    df = pd.read_csv(file_path)
    df.to_sql(name="titanic", con=engine, if_exists="replace", index=False)

with DAG(
    dag_id="extract_user_survival_data", 
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    list_files = GCSListObjectsOperator(
        task_id="list_files",
        bucket="user-survival-prediction", 
    )

    download_file = GCSToLocalFilesystemOperator(
        task_id="download_file",
        bucket="user-survival-prediction", 
        object_name="Titanic-Dataset.csv", 
        filename="/tmp/Titanic-Dataset.csv", 
    )
    
    load_data = PythonOperator(
        task_id="load_to_sql",
        python_callable=load_to_sql,
        op_kwargs={"file_path": "/tmp/Titanic-Dataset.csv"}
    )

    list_files >> download_file >> load_data
