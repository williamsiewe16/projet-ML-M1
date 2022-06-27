# Python standard modules
from datetime import datetime, timedelta, date
import os
import requests

# Airflow modules
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator


def gen_file():
    name = date.today().strftime('%m-%d-%Y,%H:%M:%S')
    req = requests.get("http://www.ip-api.com/json").text
    os.system(f"echo {req} >> /home/siewe/airflow/dags/{name}.txt")


def train_model():
    import mlflow
    import os

    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.run(
        #".",
        "/home/siewe/Documents/France/Efrei/Cours/M1/Semestre 8/UE - Big Data Fundamentals II - 6/Machine Learning II/Machine Learning Project/", 
        parameters={"embedding_dim": 16, "epochs": 5, "maxlen": 150, "tracking_uri": tracking_uri}
    )


# passer la version actuellement en staging en Archived
def current_staging_to_archived():
    import mlflow

    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    name = "SentimentAnalyzerModel"

    print("Passing current Staging Model to Archived...")
    current_staging_version = [int(run.version) for run in client.get_latest_versions(name) if run.current_stage=='Staging'][0]

    client.transition_model_version_stage(
        name = name,
        version=current_staging_version,
        stage="archived"
    )

# passer la version actuellement en production en staging
def current_prod_to_staging():
    import mlflow

    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    name = "SentimentAnalyzerModel"

    print("Passing current Production Model to Staging...")
    current_prod_version = [int(run.version) for run in client.get_latest_versions(name) if run.current_stage=='Production'][0]

    client.transition_model_version_stage(
        name = name,
        version=current_prod_version,
        stage="staging"
    )
    
    
# recuperer le dernier run et le passer en production
def new_model_to_prod():
    import mlflow
    from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    name = "SentimentAnalyzerModel"

    run_id = client.list_run_infos('0')[0].run_id
    desc = "A new version of the model automatically created"
    runs_uri = "runs:/{}/sentiment-analyzer".format(run_id)
    model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)
    client.create_model_version(name, model_src, run_id, description=desc)

    new_version = [int(run.version) for run in client.get_latest_versions(name) if run.current_stage=='None'][0]

    print("Passing new Model to Production...")
    client.transition_model_version_stage(
        name = name,
        version=new_version,
        stage="production"
    )



default_args = {
    'owner': 'siewe',
    'depends_on_past': False,
    # Start on 27th of June, 2020s
    'start_date': datetime(2022, 6, 22),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    # In case of         errors, do one retry
    'retries': 1,
    # Do the retry with 30 seconds delay after the error
    'retry_delay': timedelta(seconds=30),
    # Run once every 15 minutes
    'schedule_interval': '*/30 * * * *'
}


with DAG(
    dag_id="retrain_dag",
    default_args=default_args,
    schedule_interval='*/15 * * * *',
    tags=["my_dags"],
    catchup=False
) as dag:  
    
    #Here we define a task
    t1 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        dag=dag
    )
    
    t2 = PythonOperator(
        task_id="current_staging_to_archived",
        python_callable=current_staging_to_archived,
        dag=dag
    )

    t3 = PythonOperator(
        task_id="current_prod_to_staging",
        python_callable=current_prod_to_staging,
        dag=dag
    )
    
    t4 = PythonOperator(
        task_id="new_model_to_prod",
        python_callable=new_model_to_prod,
        dag=dag
    )
    
    t5 = BashOperator(
        task_id='deploy_to_sagemaker_endpoint',
        bash_command="./deploy.sh",
        dag=dag
    )
    
    # Configure T2 to be dependent on T1’s execution
    t1 >> t2 >> t3 >> t4 >> t5