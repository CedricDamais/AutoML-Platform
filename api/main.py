import fastapi
from fastapi import FastAPI, APIRouter
from api.routes import health, core

app = FastAPI(
    title="AutoML API",
    description="AutoML is a ML training platform in which we train different models in parallel and then deplot the best model",
    version="1.0"
)

router = APIRouter()

app.include_router(router=health.router, tags=["health"])
app.include_router(router=core.router, prefix="/api/v1",tags=["core"])

@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message" : "Welcome to the Auto ML API",
        "docs": "/docs",
        "version": "v1.0"
    }