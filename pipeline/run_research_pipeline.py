"""
Submit the compact-model research workflow to Azure ML.

Pipeline:
    budget_feature_selection
              |
              v
    train_evaluate_compact_10
              |
              +--> compact_model output
              +--> compact_evaluation output

The first job logs cross-validated experiment comparisons to MLflow.
The second job logs the frozen compact model's final holdout metrics.

After successful completion, the compact model artifact is registered
separately from the existing 23-feature benchmark model.
"""

from __future__ import annotations

from pathlib import Path

from azure.ai.ml import (
    Input,
    MLClient,
    Output,
    command,
    dsl,
)
from azure.ai.ml.constants import (
    AssetTypes,
)
from azure.ai.ml.entities import (
    Environment,
    Model,
)
from azure.identity import (
    AzureCliCredential,
)


SUBSCRIPTION_ID = (
    "d950fab4-9c1d-4990-87d2-11fb95ce511d"
)
RESOURCE_GROUP = (
    "rg-dkv-interview"
)
WORKSPACE_NAME = (
    "mlw-dkv-credit-risk"
)
COMPUTE_NAME = (
    "cpu-cluster"
)
EXPERIMENT_NAME = (
    "dkv-credit-risk-research"
)

REGISTERED_MODEL_NAME = (
    "dkv-credit-default-compact-model"
)

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

CONDA_FILE = (
    PROJECT_ROOT
    / "environments"
    / "conda.yaml"
)

TRAIN_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train"
    / "train.csv"
)

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test"
    / "test.csv"
)


def validate_local_inputs() -> None:
    required = [
        CONDA_FILE,
        TRAIN_CSV,
        TEST_CSV,
        (
            PROJECT_ROOT
            / "analysis"
            / "budget_feature_selection_experiment.py"
        ),
        (
            PROJECT_ROOT
            / "src"
            / "train_compact_azure.py"
        ),
        (
            PROJECT_ROOT
            / "src"
            / "features.py"
        ),
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Required project files are missing:\n"
            + formatted
        )


def build_environment() -> Environment:
    return Environment(
        name=(
            "dkv-credit-risk-research-env"
        ),
        description=(
            "Environment for DKV credit-risk "
            "research and compact-model experiments."
        ),
        image=(
            "mcr.microsoft.com/azureml/"
            "openmpi4.1.0-ubuntu22.04:latest"
        ),
        conda_file=str(
            CONDA_FILE
        ),
    )


def build_components(
    environment: Environment,
):
    budget_selection = command(
        name=(
            "budget_feature_selection"
        ),
        display_name=(
            "Budget Feature Selection"
        ),
        description=(
            "Nested-CV comparison of raw vs engineered "
            "feature spaces at budgets 6 and 10."
        ),
        code=str(
            PROJECT_ROOT
        ),
        command=(
            "python analysis/"
            "budget_feature_selection_experiment.py "
            "--train-data ${{inputs.train_data}} "
            "--output-dir ${{outputs.analysis_output}} "
            "--outer-folds 5 "
            "--inner-folds 3 "
            "--random-state 42"
        ),
        inputs={
            "train_data": Input(
                type=AssetTypes.URI_FILE,
            ),
        },
        outputs={
            "analysis_output": Output(
                type=AssetTypes.URI_FOLDER,
                mode="rw_mount",
            ),
        },
        environment=environment,
        compute=COMPUTE_NAME,
    )

    compact_training = command(
        name=(
            "train_evaluate_compact_10"
        ),
        display_name=(
            "Train & Evaluate Compact-10 HGB"
        ),
        description=(
            "Fit the frozen 10-feature compact HGB "
            "on all training data and evaluate once "
            "on the untouched holdout."
        ),
        code=str(
            PROJECT_ROOT
        ),
        command=(
            "python src/train_compact_azure.py "
            "--train-data ${{inputs.train_data}} "
            "--test-data ${{inputs.test_data}} "
            "--feature-selection-output "
            "${{inputs.feature_selection_output}} "
            "--model-output ${{outputs.model_output}} "
            "--evaluation-output "
            "${{outputs.evaluation_output}} "
            "--random-state 42 "
            "--threshold 0.5"
        ),
        inputs={
            "train_data": Input(
                type=AssetTypes.URI_FILE,
            ),
            "test_data": Input(
                type=AssetTypes.URI_FILE,
            ),
            "feature_selection_output": Input(
                type=AssetTypes.URI_FOLDER,
            ),
        },
        outputs={
            "model_output": Output(
                type=AssetTypes.URI_FOLDER,
                mode="rw_mount",
            ),
            "evaluation_output": Output(
                type=AssetTypes.URI_FOLDER,
                mode="rw_mount",
            ),
        },
        environment=environment,
        compute=COMPUTE_NAME,
    )

    return (
        budget_selection,
        compact_training,
    )


def build_pipeline(
    budget_selection,
    compact_training,
):
    @dsl.pipeline(
        name=(
            "dkv-credit-risk-research-pipeline"
        ),
        description=(
            "Tracked Azure ML research pipeline for "
            "feature-budget selection and final "
            "compact HGB evaluation."
        ),
    )
    def research_pipeline(
        train_data,
        test_data,
    ):
        selection_job = (
            budget_selection(
                train_data=train_data,
            )
        )

        compact_job = (
            compact_training(
                train_data=train_data,
                test_data=test_data,
                feature_selection_output=(
                    selection_job.outputs[
                        "analysis_output"
                    ]
                ),
            )
        )

        return {
            "feature_selection_analysis": (
                selection_job.outputs[
                    "analysis_output"
                ]
            ),
            "compact_model": (
                compact_job.outputs[
                    "model_output"
                ]
            ),
            "compact_evaluation": (
                compact_job.outputs[
                    "evaluation_output"
                ]
            ),
        }

    return research_pipeline


def main() -> None:
    validate_local_inputs()

    credential = AzureCliCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=(
            SUBSCRIPTION_ID
        ),
        resource_group_name=(
            RESOURCE_GROUP
        ),
        workspace_name=(
            WORKSPACE_NAME
        ),
    )

    environment = (
        build_environment()
    )

    budget_selection, (
        compact_training
    ) = build_components(
        environment
    )

    research_pipeline = (
        build_pipeline(
            budget_selection,
            compact_training,
        )
    )

    pipeline_job = research_pipeline(
        train_data=Input(
            type=AssetTypes.URI_FILE,
            path=str(
                TRAIN_CSV
            ),
        ),
        test_data=Input(
            type=AssetTypes.URI_FILE,
            path=str(
                TEST_CSV
            ),
        ),
    )

    pipeline_job.settings.default_compute = (
        COMPUTE_NAME
    )

    created_job = (
        ml_client.jobs.create_or_update(
            pipeline_job,
            experiment_name=(
                EXPERIMENT_NAME
            ),
        )
    )

    print(
        "\nSubmitted Azure ML research pipeline"
    )
    print(
        "===================================="
    )
    print(
        f"Run name: "
        f"{created_job.name}"
    )
    print(
        f"Experiment: "
        f"{EXPERIMENT_NAME}"
    )

    ml_client.jobs.stream(
        created_job.name
    )

    completed_job = (
        ml_client.jobs.get(
            created_job.name
        )
    )

    status = (
        completed_job.status
        or ""
    ).lower()

    if status != "completed":
        raise RuntimeError(
            "Pipeline did not complete successfully. "
            f"Final status: {completed_job.status}"
        )

    model_uri = (
        "azureml://jobs/"
        f"{created_job.name}/"
        "outputs/compact_model"
    )

    registered_model = (
        ml_client.models.create_or_update(
            Model(
                name=(
                    REGISTERED_MODEL_NAME
                ),
                type=(
                    AssetTypes.CUSTOM_MODEL
                ),
                path=model_uri,
                description=(
                    "Compact 10-feature HGB challenger. "
                    "Feature set selected by nested-CV "
                    "Elastic-Net ranking from the "
                    "36-feature raw+engineered candidate space."
                ),
                tags={
                    "model_role": (
                        "interpretable_challenger"
                    ),
                    "feature_count": "10",
                    "algorithm": (
                        "HistGradientBoostingClassifier"
                    ),
                    "selection_method": (
                        "nested_cv_elastic_net_budget_10"
                    ),
                    "benchmark_model": (
                        "23_feature_raw_hgb"
                    ),
                    "pipeline_run": (
                        created_job.name
                    ),
                },
            )
        )
    )

    print(
        "\nRegistered compact model"
    )
    print(
        "========================"
    )
    print(
        f"Name: "
        f"{registered_model.name}"
    )
    print(
        f"Version: "
        f"{registered_model.version}"
    )
    print(
        f"Source run: "
        f"{created_job.name}"
    )


if __name__ == "__main__":
    main()
