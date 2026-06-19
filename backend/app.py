"""FastAPI app factory — creates and configures the FastAPI application and registers routers."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import ALLOWED_ORIGINS
from .routes import health, analyze, download 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ALLOWED_ORIGINS,
    allow_methods = ["*"],
    allow_headers = ["*"],
    allow_credentials = True
)


app.include_router(analyze.router, prefix="/api", tags=["analyze"]) 
app.include_router(download.router, prefix="/api", tags=["download"])   
app.include_router(health.router, prefix="/api", tags=["health"])   

