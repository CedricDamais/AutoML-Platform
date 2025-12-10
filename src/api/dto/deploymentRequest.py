from pydantic import BaseModel

class DeploymentRequest(BaseModel):
    experiment_id: str