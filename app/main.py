import asyncio
import os
import shutil
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import OperationalError

from app.config import APP_CONFIG, LOCAL_ONLY_MODE, PATHS, initialize_settings_file, load_app_config, logger
from app.database import Base, engine
from app.routers import router
from app.tasks import _whisper_cache_cleanup_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")

    create_attempts = 3
    for attempt in range(1, create_attempts + 1):
        try:
            with engine.begin() as conn:
                Base.metadata.create_all(bind=conn)
            logger.info("Database tables ensured.")
            break
        except OperationalError as oe:
            msg = str(oe).lower()
            if "already exists" in msg or "database is locked" in msg:
                logger.warning(f"DB race/lock detected (attempt {attempt}/{create_attempts}): {oe}. Retrying...")
                import time
                time.sleep(0.5 * attempt)
                continue
            else:
                logger.exception("Database initialization failed.")
                raise
        except Exception:
            logger.exception("Unexpected error during DB initialization.")
            raise

    initialize_settings_file()
    load_app_config()

    # Start cache cleanup thread
    cache_thread = threading.Thread(target=_whisper_cache_cleanup_worker, daemon=True)
    cache_thread.start()
    logger.info("Whisper model cache cleanup thread started.")

    # Kokoro download on startup
    DOWNLOAD_KOKORO = os.environ.get("DOWNLOAD_KOKORO_ON_STARTUP", "false").lower() == "true"
    if shutil.which("kokoro-tts") and DOWNLOAD_KOKORO:
        logger.info("Checking for Kokoro TTS models...")
        from app.tasks import download_kokoro_models_if_missing
        app.state.download_kokoro_task = asyncio.create_task(download_kokoro_models_if_missing())

    # Piper check
    try:
        from piper import PiperVoice
        logger.info("Piper TTS is available.")
    except ImportError:
        logger.warning("Piper TTS not installed.")

    ENV = os.environ.get("ENV", "dev").lower()
    ALLOW_LOCAL_ONLY = os.environ.get("ALLOW_LOCAL_ONLY", "false").lower() == "true"
    if LOCAL_ONLY_MODE and ENV != "dev" and not ALLOW_LOCAL_ONLY:
        raise RuntimeError("LOCAL_ONLY_MODE may only be enabled in dev or when ALLOW_LOCAL_ONLY=true.")

    yield

    # Cleanup
    if hasattr(app.state, "download_kokoro_task"):
        app.state.download_kokoro_task.cancel()
        try:
            await app.state.download_kokoro_task
        except asyncio.CancelledError:
            pass

    logger.info("Application shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Session middleware (only if auth is enabled or for UI sessions)
    from starlette.middleware.sessions import SessionMiddleware

    secret = os.environ.get("SECRET_KEY", "change-me-in-production")
    app.add_middleware(SessionMiddleware, secret_key=secret)

    # Static files
    app.mount("/static", StaticFiles(directory=str(PATHS.BASE_DIR / "static")), name="static")

    # Include all routes
    app.include_router(router)

    return app


app = create_app()

# Make huey accessible at module level for compatibility
from app.tasks import huey
