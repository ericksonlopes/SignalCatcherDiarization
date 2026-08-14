import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware

from src.core.logger.logger import logger
from src.core.database.connector import Base, engine
from src.modules.diarization.presentation.schedules.scheduler_manager import start_scheduler
# Ensure models are loaded for create_all
import src.core.database

# Intercept default python logging to our custom logger
logging.basicConfig(handlers=[logger.get_intercept_handler()], level=logging.INFO, force=True)
for _log in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
    _logger = logging.getLogger(_log)
    _logger.handlers = [logger.get_intercept_handler()]

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting SignalCatcher API...")
    
    scheduler = start_scheduler()
    app.state.scheduler = scheduler
    
    yield
    
    logger.debug("Shutting down SignalCatcher scheduler...")
    scheduler.shutdown()

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = FastAPI(
    title="SignalCatcher Diarization API",
    description="API to manage content capture and monitoring (YouTube/RSS)",
    version="1.0.0",
    middleware=middleware,
    lifespan=lifespan,
)

# Register routes
from src.modules.diarization.presentation.router import router as diarization_router

app.include_router(diarization_router, prefix="/api/diarization", tags=["Diarization"])


@app.get("/health", tags=["Health"])
def get_health():
    """Returns the API health (useful for Docker healthchecks)."""
    return {"status": "online", "message": "SignalCatcher is running"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
