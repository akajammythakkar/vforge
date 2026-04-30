from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from database import init_db
from routers import (
    chat,
    datasets,
    training,
    benchmarks,
    models as models_router,
    settings as settings_router,
    projects,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Open-source LLM fine-tuning benchmarking — TPU vs GPU.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(benchmarks.router, prefix="/api/benchmarks", tags=["benchmarks"])
app.include_router(models_router.router, prefix="/api/models", tags=["models"])
app.include_router(settings_router.router, prefix="/api/settings", tags=["settings"])
