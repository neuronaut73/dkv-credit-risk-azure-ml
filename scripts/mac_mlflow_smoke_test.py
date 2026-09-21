from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential
import mlflow


SUBSCRIPTION_ID = "d950fab4-9c1d-4990-87d2-11fb95ce511d"
RESOURCE_GROUP = "rg-dkv-interview"
WORKSPACE_NAME = "mlw-dkv-credit-risk"


def main():
    print("Connecting to Azure ML workspace...")

    credential = AzureCliCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    workspace = ml_client.workspaces.get(WORKSPACE_NAME)

    print(f"Workspace: {workspace.name}")
    print(f"Location:  {workspace.location}")

    mlflow.set_tracking_uri(workspace.mlflow_tracking_uri)
    mlflow.set_experiment("mac-smoke-test")

    with mlflow.start_run(run_name="macbook-connection-test"):
        mlflow.log_metric("mac_smoke_test", 1.0)
        mlflow.log_param("client", "macbook")

        print("Mac -> Azure ML -> MLflow OK")


if __name__ == "__main__":
    main()