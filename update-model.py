import mlflow
import os

tracking_uri = "http://localhost:5000"
mlflow.set_tracking_uri(tracking_uri)
mlflow.run(
    ".",
    #"/home/siewe/Documents/France/Efrei/Cours/M1/Semestre 8/UE - Big Data Fundamentals II - 6/Machine Learning II/Machine Learning Project/", 
    parameters={"embedding_dim": 16, "epochs": 5, "maxlen": 150, "tracking_uri": tracking_uri}
    )

#dir="/home/siewe/Documents/France/Efrei/Cours/M1/'Semestre 8'/'UE - Big Data Fundamentals II - 6'/'Machine Learning II'/'Machine Learning Project'/"
#os.system(f"python {dir}/sentiment.py 16 5 150 {tracking_uri}")


from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

client = mlflow.tracking.MlflowClient()

name = "SentimentAnalyzerModel"

# passer la version actuellement en staging en Archived
print("Passing current Staging Model to Archived...")
current_staging_version = [int(run.version) for run in client.get_latest_versions(name) if run.current_stage=='Staging'][0]

client.transition_model_version_stage(
    name = name,
    version=current_staging_version,
    stage="archived"
)

# passer la version actuellement en production en staging
print("Passing current Production Model to Staging...")
current_prod_version = [int(run.version) for run in client.get_latest_versions(name) if run.current_stage=='Production'][0]

client.transition_model_version_stage(
    name = name,
    version=current_prod_version,
    stage="staging"
)



# recuperer le dernier run et le passer en production
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


print('done')