import fastapi
from fastapi import FastAPI, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .routes import health, core

from ..exceptions.internalServerError import InternalServerError
from ..exceptions.notFoundError import DataSetNotFoundError

app = FastAPI(
    title="AutoML API",
    description="AutoML is a ML training platform in which we train different models in parallel and then deploy the best model",
    version="1.0"
)

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501", "http://127.0.0.1:3000", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.exception_handler(DataSetNotFoundError)
async def data_set_not_found_exception_handler(request: Request, exc: DataSetNotFoundError):
    return JSONResponse(
        status_code=exc.code,
        content={"message": exc.message},
    )

@app.exception_handler(InternalServerError)
async def internal_server_error_exception_handler(request: Request, exc: InternalServerError):
    return JSONResponse(
        status_code=exc.code,
        content={"message": exc.message},
    )
