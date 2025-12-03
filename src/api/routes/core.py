from fastapi import APIRouter, HTTPException
from api.dto.datasetRequest import DatasetRequest
from utils.logging import logger

router = APIRouter()

@router.post("/d_dataset")
async def send_dataset(req : DatasetRequest):
    """
    The user of the app sends the dataset through this endpoint.
    We should keep the dataset size under 10MB to not have any timeout issues
    """
    if not req.dataset_csv:
        raise HTTPException(status_code=400, detail="Bad Request, You did not give a dataset to train on !")

    logger.debug("Received Data set from user")

    logger.debug("Job Manager, Successfully received dataset.")
    


