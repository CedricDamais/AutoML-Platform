from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "API up and Running"}

@router.get("/ready")
async def  readyness_check():
    """
    Check if the app is ready to be ran,
    Its not done yet
    """
    # TODO
    return {
        "status":"ready",
        "service":"Auto ML"
    }