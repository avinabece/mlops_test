from typing import NamedTuple
from kfp.v2 import dsl
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics,
                        component,
                        OutputPath,
                        InputPath)
from kfp.v2 import compiler
from google.cloud import bigquery
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from google_cloud_pipeline_components import aiplatform as gcc_aip
import logging
import google.cloud.logging
from google.logging.type import log_severity_pb2 as severity
client = google.cloud.logging.Client()
logger = client.logger("DELOITTE_CUSTOM_BUILD_LOGGING_2022_11_09")


PROJECT_ID='dca-sandbox-project-4' # Change to your projecr
# Set bucket name
BUCKET_NAME="gs://"+PROJECT_ID+"-mlops_poc_health_2022"

# Create bucket
PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_artifacts/"

REGION="us-central1"

@component(
    packages_to_install=["pandas", "pyarrow", "sklearn", "fsspec", "gcsfs"],
    base_image="python:3.9",
    output_component_file="tabular_template.yaml"
)
def train_test_split(
        url: str,
        dataset_train: Output[Dataset],
        dataset_test: Output[Dataset]
):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(url, delimiter=",")
    df['target'] = df.attrition
    df = df.drop(['attrition'], axis=1)
    le = LabelEncoder()
    encoded_labels = le.fit_transform(df.target)
    df['target'] = encoded_labels

    for col,data_type in df.dtypes.iteritems():
        if data_type == 'object':
            le = LabelEncoder()
            encoded_values = le.fit_transform(df[col])
            df[col] = encoded_values

    train, test = train_test_split(df, test_size=0.3, random_state=42)
    train.to_csv(dataset_train.path + ".csv", index=False, encoding='utf-8-sig')
    test.to_csv(dataset_test.path + ".csv", index=False, encoding='utf-8-sig')

    return


@component(
    packages_to_install=[
        "pandas",
        "sklearn",
        "fsspec",
        "gcsfs"
    ], base_image="python:3.9",
)
def train(
        dataset: Input[Dataset],
        model: Output[Model],
):
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import pickle

    data = pd.read_csv(dataset.path + ".csv")
    model_rf = RandomForestClassifier(n_estimators=10)
    model_rf.fit(
        data.drop(columns=["target"]),
        data.target,
    )
    model.metadata["framework"] = "RF"
    file_name = model.path + f".pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(model_rf, file)

    return


@component(
    packages_to_install=[
        "pandas",
        "sklearn",
        "fsspec",
        "gcsfs"
    ], base_image="python:3.9",
)
def model_evaluation(
        test_set: Input[Dataset],
        rf_model: Input[Model],
        thresholds_dict_str: str,
        metrics: Output[ClassificationMetrics],
        kpi: Output[Metrics]
) -> NamedTuple("output", [("deploy", str)]):
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import logging
    import pickle
    from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score
    import json
    import typing
    from sklearn.preprocessing import LabelEncoder

    def threshold_check(val1, val2):
        cond = "false"
        if val1 >= val2:
            cond = "true"
        return cond

    data = pd.read_csv(test_set.path + ".csv")
    model = RandomForestClassifier()
    file_name = rf_model.path + ".pkl"
    with open(file_name, 'rb') as file:
        model = pickle.load(file)

    y_test = data.drop(columns=["target"])
    le = LabelEncoder()
    y_target = le.fit_transform(data.target)
    y_pred = model.predict(y_test)

    y_scores = model.predict_proba(data.drop(columns=["target"]))[:, 1]
    fpr, tpr, thresholds = roc_curve(
        y_true=data.target.to_numpy(), y_score=y_scores, pos_label=True
    )
    metrics.log_roc_curve(fpr.tolist(), tpr.tolist(), thresholds.tolist())

    metrics.log_confusion_matrix(
        ["False", "True"],
        confusion_matrix(
            data.target, y_pred
        ).tolist(),
    )

    accuracy = accuracy_score(data.target, y_pred.round())
    thresholds_dict = json.loads(thresholds_dict_str)
    rf_model.metadata["accuracy"] = float(accuracy)
    kpi.log_metric("accuracy", float(accuracy))
    deploy = threshold_check(float(accuracy), int(thresholds_dict['roc']))
    return (deploy,)


@component(
    packages_to_install=["google-cloud-aiplatform", "scikit-learn==1.0.0", "kfp"],
    base_image="python:3.9",
    output_component_file="tabular_template_coponent.yml"
)
def deploy_model(
        model: Input[Model],
        project: str,
        region: str,
        serving_container_image_uri: str,
        vertex_endpoint: Output[Artifact],
        vertex_model: Output[Model]
):
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)

    DISPLAY_NAME = "tabular"
    MODEL_NAME = "tabular-rf"
    ENDPOINT_NAME = "tabular_endpoint"

    def create_endpoint():
        endpoints = aiplatform.Endpoint.list(
            filter='display_name="{}"'.format(ENDPOINT_NAME),
            order_by='create_time desc',
            project=project,
            location=region,
        )
        if len(endpoints) > 0:
            endpoint = endpoints[0]  # most recently created
        else:
            endpoint = aiplatform.Endpoint.create(
                display_name=ENDPOINT_NAME, project=project, location=region
            )
        return


    endpoint = create_endpoint()

    # Import a model programmatically
    model_upload = aiplatform.Model.upload(
        display_name=DISPLAY_NAME,
        artifact_uri=model.uri.replace("model", ""),
        serving_container_image_uri=serving_container_image_uri,
        serving_container_health_route=f"/v1/models/{MODEL_NAME}",
        serving_container_predict_route=f"/v1/models/{MODEL_NAME}:predict",
        serving_container_environment_variables={
            "MODEL_NAME": MODEL_NAME,
        },
    )
    model_deploy = model_upload.deploy(
        machine_type="n1-standard-4",
        endpoint=endpoint,
        traffic_split={"0": 100},
        deployed_model_display_name=DISPLAY_NAME,
    )

    # Save data to the output params
    vertex_model.uri = model_deploy.resource_name
    return


from datetime import datetime
TIMESTAMP =datetime.now().strftime("%Y%m%d%H%M%S")
DISPLAY_NAME = 'pipeline-tabular-job{}'.format(TIMESTAMP)


@dsl.pipeline(
    # Default pipeline root. You can override it when submitting the pipeline.
    pipeline_root=PIPELINE_ROOT,
    # A name for the pipeline. Use to determine the pipeline Context.
    name="pipeline-tabular",

)
def pipeline(
        url: str = 'gs://mlops_poc_health_2022/Tabular/HR_Employee_Attrition.csv',
        project: str = PROJECT_ID,
        region: str = REGION,
        display_name: str = DISPLAY_NAME,
        api_endpoint: str = REGION + "-aiplatform.googleapis.com",
        thresholds_dict_str: str = '{"roc":0.8}',
        serving_container_image_uri: str = "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"
):
    data_op = train_test_split(url)
    train_model_op = train(data_op.outputs["dataset_train"])
    model_evaluation_op = model_evaluation(
        test_set=data_op.outputs["dataset_test"],
        rf_model=train_model_op.outputs["model"],
        thresholds_dict_str=thresholds_dict_str,
        # I deploy the model anly if the model performance is above the threshold
    )

    with dsl.Condition(
            model_evaluation_op.outputs["deploy"] == "true",
            name="deploy-tabular",
    ):
        deploy_model_op = deploy_model(
            model=train_model_op.outputs['model'],
            project=project,
            region=region,
            serving_container_image_uri=serving_container_image_uri,
        )

def compile_pipeline():
    logger.log_text("COMILATION STARTED", severity=severity.INFO)
    compiled=compiler.Compiler().compile(pipeline_func=pipeline,package_path='tabular_template.json')
    logger.log_text("COMPILATION DONE", severity=severity.INFO)

    return compiled

def main():
    print("i am here now")
    result = compile_pipeline()

if __name__ == "__main__":
    main()

