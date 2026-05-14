import html
import logging
import os
import resource
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel

# --- Environment Mode ---
LOCAL_ONLY_MODE = os.getenv("LOCAL_ONLY", "True").lower() in ("true", "1", "t")

# --- Path Safety ---
_BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_BASE = Path(os.environ.get("UPLOADS_DIR", str(_BASE_DIR / "uploads"))).resolve()
PROCESSED_BASE = Path(os.environ.get("PROCESSED_DIR", str(_BASE_DIR / "processed"))).resolve()
CHUNK_TMP_BASE = Path(
    os.environ.get("CHUNK_TMP_DIR", str(UPLOADS_BASE / "tmp"))
).resolve()


class AppPaths(BaseModel):
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOADS_DIR: Path = UPLOADS_BASE
    PROCESSED_DIR: Path = PROCESSED_BASE
    CHUNK_TMP_DIR: Path = CHUNK_TMP_BASE
    TTS_MODELS_DIR: Path = BASE_DIR / "models" / "tts"
    KOKORO_TTS_MODELS_DIR: Path = BASE_DIR / "models" / "tts" / "kokoro"
    KOKORO_MODEL_FILE: Path = KOKORO_TTS_MODELS_DIR / "kokoro-v1.0.onnx"
    KOKORO_VOICES_FILE: Path = KOKORO_TTS_MODELS_DIR / "voices-v1.0.bin"
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'jobs.db'}"
    HUEY_DB_PATH: str = str(BASE_DIR / "huey.db")
    CONFIG_DIR: Path = BASE_DIR / "config"
    SETTINGS_FILE: Path = CONFIG_DIR / "settings.yml"
    DEFAULT_SETTINGS_FILE: Path = BASE_DIR / "settings.default.yml"


PATHS = AppPaths()
APP_CONFIG: Dict[str, Any] = {}

# Ensure directories exist
PATHS.UPLOADS_DIR.mkdir(exist_ok=True, parents=True)
PATHS.PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
PATHS.CHUNK_TMP_DIR.mkdir(exist_ok=True, parents=True)
PATHS.CONFIG_DIR.mkdir(exist_ok=True, parents=True)
PATHS.TTS_MODELS_DIR.mkdir(exist_ok=True, parents=True)
PATHS.KOKORO_TTS_MODELS_DIR.mkdir(exist_ok=True, parents=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if LOCAL_ONLY_MODE:
    logger.warning("Authentication is DISABLED. Running in LOCAL_ONLY mode.")


# --- Resource Limiting ---
def _limit_resources_preexec():
    """Set resource limits for child processes to prevent DoS attacks."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (6000, 6000))
        resource.setrlimit(
            resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
        )
    except Exception as e:
        logger.warning(f"Could not set resource limits: {e}")


# --- Config Helpers ---
def deep_merge(source: dict, dest: dict) -> dict:
    """Recursively merges source dict into dest dict. Modifies dest in place."""
    for key, value in source.items():
        if isinstance(value, dict) and key in dest and isinstance(dest[key], dict):
            deep_merge(value, dest[key])
        else:
            dest[key] = value
    return dest


def initialize_settings_file():
    """Ensures that config/settings.yml exists. Copies from default if not."""
    if not PATHS.SETTINGS_FILE.exists():
        logger.info(
            f"'{PATHS.SETTINGS_FILE}' not found. Copying from '{PATHS.DEFAULT_SETTINGS_FILE}'."
        )
        try:
            import shutil

            shutil.copy(PATHS.DEFAULT_SETTINGS_FILE, PATHS.SETTINGS_FILE)
        except FileNotFoundError:
            logger.error(
                f"CRITICAL: Default settings file '{PATHS.DEFAULT_SETTINGS_FILE}' not found."
            )
            PATHS.SETTINGS_FILE.touch()
        except Exception as e:
            logger.error(f"CRITICAL: Failed to copy default settings file: {e}")
            PATHS.SETTINGS_FILE.touch()


def load_app_config():
    """Loads configuration by deeply merging defaults, default.yml, settings.yml, and env vars."""
    global APP_CONFIG

    hardcoded_defaults = {
        "app_settings": {
            "max_file_size_mb": 100,
            "allowed_all_extensions": [],
            "app_public_url": "",
            "model_concurrency": 1,
            "model_inactivity_timeout": 1800,
            "cache_check_interval": 300,
        },
        "transcription_settings": {
            "whisper": {
                "allowed_models": ["tiny", "base", "small"],
                "compute_type": "int8",
                "device": "cpu",
            }
        },
        "tts_settings": {
            "piper": {
                "model_dir": str(PATHS.TTS_MODELS_DIR),
                "use_cuda": False,
                "synthesis_config": {
                    "length_scale": 1.0,
                    "noise_scale": 0.667,
                    "noise_w": 0.8,
                },
            },
            "kokoro": {
                "model_dir": str(PATHS.KOKORO_TTS_MODELS_DIR),
                "command_template": "kokoro-tts {input} {output} --model {model_path} --voices {voices_path} --lang {lang} --voice {model_name}",
            },
        },
        "conversion_tools": {},
        "ocr_settings": {"ocrmypdf": {}},
        "auth_settings": {
            "oidc_client_id": "",
            "oidc_client_secret": "",
            "oidc_server_metadata_url": "",
            "admin_users": [],
        },
        "webhook_settings": {
            "enabled": False,
            "allow_chunked_api_uploads": False,
            "allowed_callback_urls": [],
            "callback_bearer_token": "",
        },
    }

    config = hardcoded_defaults.copy()

    # Merge settings.default.yml
    try:
        with open(PATHS.DEFAULT_SETTINGS_FILE, "r", encoding="utf8") as f:
            default_cfg = yaml.safe_load(f) or {}
        config = deep_merge(default_cfg, config)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.warning(f"Could not load settings.default.yml: {e}")

    # Merge settings.yml
    try:
        with open(PATHS.SETTINGS_FILE, "r", encoding="utf8") as f:
            user_cfg = yaml.safe_load(f) or {}
        config = deep_merge(user_cfg, config)
    except (FileNotFoundError, yaml.YAMLError):
        pass

    # Environment overrides for transcription
    trans_settings = config.get("transcription_settings", {}).get("whisper", {})
    transcription_device = os.environ.get(
        "TRANSCRIPTION_DEVICE", trans_settings.get("device", "cpu")
    )
    default_compute_type = "float16" if transcription_device == "cuda" else "int8"
    transcription_compute_type = os.environ.get(
        "TRANSCRIPTION_COMPUTE_TYPE",
        trans_settings.get("compute_type", default_compute_type),
    )
    transcription_device_index_str = os.environ.get("TRANSCRIPTION_DEVICE_INDEX", "0")

    try:
        if "," in transcription_device_index_str:
            transcription_device_index = [
                int(i.strip()) for i in transcription_device_index_str.split(",")
            ]
        else:
            transcription_device_index = int(transcription_device_index_str)
    except ValueError:
        logger.warning(
            f"Invalid TRANSCRIPTION_DEVICE_INDEX: '{transcription_device_index_str}'. Defaulting to 0."
        )
        transcription_device_index = 0

    config.setdefault("transcription_settings", {}).setdefault("whisper", {})
    config["transcription_settings"]["whisper"]["device"] = transcription_device
    config["transcription_settings"]["whisper"]["compute_type"] = transcription_compute_type
    config["transcription_settings"]["whisper"]["device_index"] = transcription_device_index

    # Final processing
    app_settings = config.get("app_settings", {})
    max_mb = app_settings.get("max_file_size_mb", 100)
    app_settings["max_file_size_bytes"] = int(max_mb) * 1024 * 1024
    allowed = app_settings.get("allowed_all_extensions", [])
    if not isinstance(allowed, (list, set)):
        allowed = []
    app_settings["allowed_all_extensions"] = set(allowed)
    config["app_settings"] = app_settings

    APP_CONFIG.clear()
    APP_CONFIG.update(config)
    logger.info("Application configuration loaded.")
