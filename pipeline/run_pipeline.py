"""
Azure ML pipeline orchestration for the DKV Mobility interview case.

This script runs locally and submits the actual ML workload to Azure.

Pipeline:
    preprocess -> train -> evaluate

After successful completion, the fitted sklearn pipeline is registered
as a versioned Azure ML model asset.
"""

from __future__ import annotations

from pathlib import Path

from azure.ai.ml import (
    MLClient,
    Input,
    Output,
    command,
    dsl,
)
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment, Model
from azure.identity import AzureCliCredential


# ---------------------------------------------------------------------
# Azure configuration
# ---------------------------------------------------------------------

SUBSCRIPTION_ID = "d950fab4-9c1d-4990-87d2-11fb95ce511d"
RESOURCE_GROUP = "rg-dkv-interview"
WORKSPACE_NAME = "mlw-dkv-credit-risk"
COMPUTE_NAME = "cpu-cluster"

ENVIRONMENT_NAME = "dkv-credit-risk-env"
REGISTERED_MODEL_NAME = "dkv-credit-default-model"

EXPERIMENT_NAME = "dkv-credit-risk"
PIPELINE_NAME = "dkv-credit-default-pipeline"


# ---------------------------------------------------------------------
# Local project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
CONDA_FILE = PROJECT_ROOT / "environments" / "conda.yaml"


# ---------------------------------------------------------------------
# Azure connection
# ---------------------------------------------------------------------

credential = AzureCliCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)


# ---------------------------------------------------------------------
# Azure ML environment
# ---------------------------------------------------------------------

pipeline_environment = Environment(
    name=ENVIRONMENT_NAME,
    description=(
        "Environment for the DKV credit default Azure ML pipeline."
    ),
    image=(
        "mcr.microsoft.com/azureml/"
        "openmpi4.1.0-ubuntu22.04:latest"
    ),
    conda_file=str(CONDA_FILE),
)

registered_environment = ml_client.environments.create_or_update(
    pipeline_environment
)

environment_reference = (
    f"azureml:{registered_environment.name}:"
    f"{registered_environment.version}"
)

print(
    "Using Azure ML environment: "
    f"{environment_reference}"
)


# ---------------------------------------------------------------------
# Component 1: deterministic data preparation
# ---------------------------------------------------------------------

preprocess_component = command(
    name="dkv_credit_preprocess",
    display_name="Prepare credit default data",
    description=(
        "Load UCI data, validate schema, apply deterministic "
        "preprocessing, and create stratified train/test split."
    ),
    code=str(SRC_DIR),
    command=(
        "python preprocess.py "
        "--train-output ${{outputs.train_data}} "
        "--test-output ${{outputs.test_data}} "
        "--test-size 0.20 "
        "--random-state 42"
    ),
    environment=environment_reference,
    outputs={
        "train_data": Output(
            type="uri_folder",
            mode="rw_mount",
        ),
        "test_data": Output(
            type="uri_folder",
            mode="rw_mount",
        ),
    },
)


# ---------------------------------------------------------------------
# Component 2: model training and cross-validation
# ---------------------------------------------------------------------

train_component = command(
    name="dkv_credit_train",
    display_name="Train credit default model",
    description=(
        "Run leakage-safe cross-validation, compare candidate models, "
        "and fit the selected sklearn pipeline."
    ),
    code=str(SRC_DIR),
    command=(
        "python train.py "
        "--train-data ${{inputs.train_data}}/train.csv "
        "--model-output ${{outputs.model_output}} "
        "--random-state 42 "
        "--cv-folds 5"
    ),
    environment=environment_reference,
    inputs={
        "train_data": Input(
            type="uri_folder",
        ),
    },
    outputs={
        "model_output": Output(
            type="uri_folder",
            mode="rw_mount",
        ),
    },
)


# ---------------------------------------------------------------------
# Component 3: untouched holdout evaluation
# ---------------------------------------------------------------------

evaluate_component = command(
    name="dkv_credit_evaluate",
    display_name="Evaluate credit default model",
    description=(
        "Evaluate the fitted pipeline on the untouched holdout set "
        "and log discrimination, classification, and calibration metrics."
    ),
    code=str(SRC_DIR),
    command=(
        "python evaluate.py "
        "--test-data ${{inputs.test_data}}/test.csv "
        "--model-path ${{inputs.model_input}}/model.joblib "
        "--evaluation-output ${{outputs.evaluation_output}} "
        "--threshold 0.50"
    ),
    environment=environment_reference,
    inputs={
        "test_data": Input(
            type="uri_folder",
        ),
        "model_input": Input(
            type="uri_folder",
        ),
    },
    outputs={
        "evaluation_output": Output(
            type="uri_folder",
            mode="rw_mount",
        ),
    },
)


# ---------------------------------------------------------------------
# Pipeline DAG
# ---------------------------------------------------------------------

@dsl.pipeline(
    name=PIPELINE_NAME,
    description=(
        "End-to-end credit default training and evaluation pipeline."
    ),
    compute=COMPUTE_NAME,
)
def credit_default_pipeline():

    preprocess_job = preprocess_component()

    train_job = train_component(
        train_data=preprocess_job.outputs.train_data,
    )

    evaluate_job = evaluate_component(
        test_data=preprocess_job.outputs.test_data,
        model_input=train_job.outputs.model_output,
    )

    return {
        "model_output": train_job.outputs.model_output,
        "evaluation_output": evaluate_job.outputs.evaluation_output,
    }


# ---------------------------------------------------------------------
# Submit pipeline
# ---------------------------------------------------------------------

pipeline_job = credit_default_pipeline()

submitted_job = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name=EXPERIMENT_NAME,
)

print(
    "\nPipeline submitted successfully."
)
print(
    f"Azure ML job name: {submitted_job.name}"
)
print(
    "The workload is now running remotely on Azure."
)

# Stream Azure logs back into the LOCAL terminal.
ml_client.jobs.stream(
    submitted_job.name
)


# ---------------------------------------------------------------------
# Register model after successful pipeline completion
# ---------------------------------------------------------------------

completed_job = ml_client.jobs.get(
    submitted_job.name
)

if completed_job.status != "Completed":
    raise RuntimeError(
        "Pipeline did not complete successfully. "
        f"Final status: {completed_job.status}"
    )


model_path = (
    f"azureml://jobs/{submitted_job.name}/"
    "outputs/model_output/paths/model.joblib"
)

model = Model(
    name=REGISTERED_MODEL_NAME,
    description=(
        "Fitted sklearn feature-transformation and "
        "credit-default classification pipeline."
    ),
    path=model_path,
    type=AssetTypes.CUSTOM_MODEL,
)

registered_model = ml_client.models.create_or_update(
    model
)

print(
    "\nModel registered successfully."
)
print(
    f"Model: {registered_model.name}"
)
print(
    f"Version: {registered_model.version}"
)