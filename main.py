from dotenv import load_dotenv
from fastapi import FastAPI, Request
import time

load_dotenv()

from app.routes import router
from app.logger import logger

# Load environment variables

# Create FastAPI app
app = FastAPI(title="Whisper Diarization API")

# Register routes
app.include_router(router)


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("[yellow]🚀 FastAPI server is running![/yellow]")


# Middleware for logging each request
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # ms

    logger.info(
        f"[cyan]{request.method}[/cyan] {request.url.path} "
        f"[green]{response.status_code}[/green] "
        f"({process_time:.2f} ms)"
    )

    return response
