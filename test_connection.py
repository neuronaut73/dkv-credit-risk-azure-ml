from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential

SUBSCRIPTION_ID = "d950fab4-9c1d-4990-87d2-11fb95ce511d"
RESOURCE_GROUP = "rg-dkv-interview"
WORKSPACE_NAME = "mlw-dkv-credit-risk"

credential = AzureCliCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

workspace = ml_client.workspaces.get(WORKSPACE_NAME)
compute = ml_client.compute.get("cpu-cluster")

print("Azure ML connection successful!")
print(f"Workspace: {workspace.name}")
print(f"Location:  {workspace.location}")
print(f"Compute:   {compute.name}")
print(f"VM size:   {compute.size}")
print(f"State:     {compute.provisioning_state}")