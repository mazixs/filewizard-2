# File Wizard — Agent Guide

> This file is intended for AI coding agents. It describes the project architecture, conventions, and critical implementation details that are not obvious from reading `README.md` alone.

---

## Project Overview

File Wizard is a self-hosted, browser-based utility for file conversion, OCR, and audio transcription. It wraps common CLI converters (FFmpeg, LibreOffice, Pandoc, ImageMagick, etc.) and integrates local ML models (`faster-whisper`, `piper-tts`, `kokoro-tts`).

**Repository:** `LoredCast/filewizard`  
**License:** See `LICENSE`  
**Language:** English (all comments, docs, and UI text)

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Web Framework | FastAPI (Python 3.12) |
| Frontend | Vanilla HTML/JS/CSS, Jinja2 templates |
| Database | SQLite (via SQLAlchemy ORM) |
| Task Queue | Huey (`SqliteHuey`) — runs in the same process family |
| Process Runner | Gunicorn + Uvicorn worker + Supervisor |
| Auth (optional) | OIDC/OAuth2 via `authlib` |
| ML Inference | `faster-whisper` (transcription), `piper-tts`, `kokoro-tts` |
| OCR | `ocrmypdf`, `pytesseract`, `pypdf` |
| Configuration | YAML (`settings.default.yml` → `config/settings.yml`) |

---

## Project Structure

```
.
├── main.py                  # Entire backend (~3750 lines). Contains:
│                            #   - FastAPI app, routes, auth
│                            #   - SQLAlchemy models & CRUD
│                            #   - Huey background tasks
│                            #   - CLI wrappers & conversion logic
│                            #   - TTS / OCR / transcription runners
├── settings.default.yml     # Default tool configurations, formats, auth templates
├── config/                  # Runtime config directory (created on start)
│   └── settings.yml         # User overrides (auto-copied from default on first run)
├── templates/
│   ├── index.html           # Main UI (drag-and-drop, job history, action dialogs)
│   └── settings.html        # Admin settings editor
├── static/
│   ├── css/style.css        # Main stylesheet
│   ├── css/settings.css     # Settings page stylesheet
│   ├── js/script.js         # Main frontend logic
│   └── js/settings.js       # Settings page logic
├── requirements.txt         # Full Python dependencies
├── requirements_small.txt   # Smaller dependency set (no TeX, reduced markitdown)
├── requirements_cuda.txt    # CUDA-enabled dependencies (+ctranslate2[cuda], marker-pdf)
├── Dockerfile               # Multi-stage build: cuda-builder, full-builder, small-builder, final stages
├── docker-compose.yml       # Pre-configured compose (port 6969:8000)
├── supervisor.conf          # Runs gunicorn + huey_consumer inside container
├── run.sh                   # Local startup script (gunicorn + huey_consumer)
├── .env.example             # Example environment variables
├── jobs.db                  # SQLite database (created at runtime)
├── huey.db                  # Huey task queue backing store
├── uploads/                 # Uploaded files (env: `UPLOADS_DIR`)
└── processed/               # Output files (env: `PROCESSED_DIR`)
```

**There are no separate Python packages or modules.** The entire application lives in a single `main.py` file. Any change to routing, models, tasks, or business logic is made there.

---

## Build & Run Commands

### Local (no Docker)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Copy and edit environment variables
cp .env.example .env
# Start
chmod +x run.sh
./run.sh
```
`run.sh` starts:
1. **Gunicorn** — `gunicorn -w 4 --threads 2 -k uvicorn.workers.UvicornWorker main:app -b 0.0.0.0:8000`
2. **Huey consumer** — `huey_consumer.py main.huey -w 4`

### Docker (recommended)
```bash
# Full image (default target: full-final)
docker compose up -d

# Explicit build types
docker build --build-arg BUILD_TYPE=full  -t filewizard:full .
docker build --build-arg BUILD_TYPE=small -t filewizard:small .
docker build --build-arg BUILD_TYPE=cuda  -t filewizard:cuda .
```

Inside the container, `supervisord` manages both gunicorn and the huey consumer.

---

## Configuration System

Configuration is loaded at startup via `load_app_config()` in `main.py` using a **deep-merge** cascade:

1. **Hardcoded defaults** (in `main.py`)
2. **`settings.default.yml`** (shipped with the repo)
3. **`config/settings.yml`** (user overrides, created from default if missing)
4. **Environment variable overrides** (e.g., `TRANSCRIPTION_DEVICE`, `TRANSCRIPTION_COMPUTE_TYPE`)

**Critical:** `settings.default.yml` must never be edited at runtime. All user changes go through the `/settings` page, which writes to `config/settings.yml`.

Key environment variables:
- `LOCAL_ONLY` — `True` disables auth (default). `False` enables OIDC.
- `SECRET_KEY` — Required for session middleware when auth is enabled.
- `UPLOADS_DIR`, `PROCESSED_DIR`, `CHUNK_TMP_DIR` — File storage paths.
- `TRANSCRIPTION_DEVICE` — `cpu` or `cuda`.
- `TRANSCRIPTION_COMPUTE_TYPE` — `int8`, `float16`, etc.
- `MODEL_CONCURRENCY` — Max parallel AI model loads (default 1).
- `DOWNLOAD_KOKORO_ON_STARTUP` — Auto-download Kokoro TTS models.

---

## Code Organization (within `main.py`)

`main.py` is organized into numbered sections. Maintain this ordering when adding code:

1. **Configuration & Security Helpers** — Path safety, sanitization, resource limits, model semaphore.
2. **Database & Schemas** — SQLAlchemy `Base`, `Job`, `Notification`, Pydantic schemas (`JobSchema`, `JobCreate`, etc.).
3. **CRUD Operations & Webhooks** — `get_job`, `create_job`, `update_job_status`, `send_webhook_notification`.
4. **Background Task Setup** — Huey instance, Whisper model cache, Piper voice cache, cache eviction thread.
5. **FastAPI Application** — Lifespan, middleware (CORS, sessions), static files, templates.
6. **Auth & User Helpers** — `get_current_user`, `require_user`, `require_admin`, OIDC registration.
7. **File Saving & Upload Utilities** — `save_upload_file`, chunked upload helpers.
8. **Task Runners (Huey tasks)** — `run_transcription_task`, `run_tts_task`, `run_pdf_ocr_task`, `run_image_ocr_task`, `run_conversion_task`, `run_academic_pandoc_task`, `unzip_and_dispatch_task`.
9. **API Routes** — Legacy direct-upload routes, API v1 webhook routes, job management, downloads.
10. **Page Routes** — `/`, `/settings`, `/login`, `/logout`, `/auth`.

---

## Database Schema

Two tables, both SQLite:

### `jobs`
| Column | Type | Notes |
|--------|------|-------|
| `id` | String PK | Hex UUID |
| `user_id` | String, indexed | `sub` from OIDC or `local_user` |
| `parent_job_id` | String, nullable | For ZIP batch jobs |
| `task_type` | String, indexed | `transcription`, `conversion`, `ocr`, `ocr-image`, `tts`, `unzip`, `academic_pandoc` |
| `status` | String | `pending`, `processing`, `completed`, `failed`, `cancelled` |
| `progress` | Integer | 0–100 |
| `original_filename` | String | |
| `input_filepath` | String | |
| `input_filesize` | Integer | Bytes |
| `processed_filepath` | String, nullable | |
| `output_filesize` | Integer, nullable | |
| `result_preview` | Text, nullable | First ~1000 chars of output |
| `error_message` | Text, nullable | |
| `callback_url` | String, nullable | Webhook callback |
| `created_at`, `updated_at` | DateTime | UTC |

### `notifications`
Used for WebSocket notification queuing (currently `ENABLE_WEBSOCKETS = False` by default).

The engine uses `NullPool` and sets `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;` on connect.

---

## Background Tasks (Huey)

All heavy work runs in Huey tasks. The FastAPI routes only enqueue jobs.

Key tasks:
- `run_transcription_task()` — Loads Whisper model (cached), streams segments, updates DB progress, supports SRT timestamps.
- `run_tts_task()` — Piper (Python API) or Kokoro (CLI wrapper).
- `run_pdf_ocr_task()` — `ocrmypdf` with configurable deskew/clean/force_ocr.
- `run_image_ocr_task()` — `pytesseract.image_to_pdf_or_hocr`, multi-frame TIFF support, merges with `PdfMerger`.
- `run_conversion_task()` — Generic CLI wrapper with cancellable `Popen` polling.
- `run_academic_pandoc_task()` — Unzips archive, finds `.md`/`.bib`/`.csl`, runs Pandoc with `--citeproc`.
- `unzip_and_dispatch_task()` — Extracts ZIP, dispatches each file as a child job.
- `_update_parent_zip_job_progress()` — Periodic progress aggregator for batch jobs.

**Model caching:**
- Whisper models are cached in `WHISPER_MODELS_CACHE` and evicted after `model_inactivity_timeout` seconds (default 1800) by a background daemon thread.
- Piper voices are cached in `PIPER_VOICES_CACHE`.
- A global `threading.Semaphore` (`get_model_semaphore()`) limits concurrent model loads.

---

## API Endpoints Summary

### Upload & Processing
- `POST /upload/chunk` — Chunked upload (UI)
- `POST /upload/finalize` — Stitch chunks, create job, dispatch task
- `POST /transcribe-audio` — Legacy direct upload
- `POST /convert-file` — Legacy direct upload
- `POST /ocr-pdf`, `POST /ocr-image` — Legacy direct upload

### API v1 (Webhook / Programmatic)
- `POST /api/v1/process` — Single-request processing with `callback_url`
- `POST /api/v1/upload/chunk`, `POST /api/v1/upload/finalize` — Chunked API uploads
- Requires bearer token auth (unless `LOCAL_ONLY_MODE`)
- Webhooks must be enabled in settings and callback URL must be in `allowed_callback_urls`

### Job Management
- `GET /jobs` — List user's jobs
- `POST /api/v1/jobs/status` — Batch status query (used by frontend polling)
- `GET /job/{job_id}` — Single job status
- `POST /job/{job_id}/cancel` — Request cancellation
- `GET /download/{filename}` — Download processed file
- `POST /download/batch` — ZIP multiple completed jobs
- `GET /download/zip-batch/{job_id}` — Download all outputs from a ZIP batch

### Settings & Admin
- `GET /settings` — Settings page (admin only to save)
- `POST /settings/save` — Merge UI config into `config/settings.yml`
- `POST /settings/clear-history` — Delete user's job rows
- `POST /settings/delete-files` — Delete user's processed files

### Auth
- `GET /login`, `/auth`, `/logout` — OIDC flow (only when `LOCAL_ONLY=False`)
- `GET /api/authz/forward-auth` — For reverse-proxy forward-auth

---

## Code Style Guidelines

- **Single file:** Keep new logic in `main.py`. Resist splitting into modules unless the file exceeds ~5000 lines and the change is architectural.
- **Imports:** Group as (1) stdlib, (2) third-party, (3) local. Existing grouping is loose; follow the surrounding style.
- **Type hints:** Use Python 3.10+ syntax (`str | None`, `list[str]`). Pydantic models use `ConfigDict(from_attributes=True)`.
- **Logging:** Use the module-level `logger = logging.getLogger(__name__)`. Always include `job_id` in log messages for tasks.
- **Exceptions:** In Huey tasks, catch broad exceptions, log with `logger.exception(...)`, update job status to `failed`, and clean up temp files in `finally`.
- **Path safety:** Every file operation must use `ensure_path_is_safe(path, [allowed_base, ...])` before resolving.
- **Atomic writes:** Write to `.tmp-{uuid}` files, then `Path.replace()`.
- **Command templates:** Use `validate_and_build_command()` to prevent shell injection. Only allowed placeholders are whitelisted.

---

## Testing Instructions

**There is currently no test suite.** The project relies on:
1. Manual UI testing (drag-and-drop, polling, downloads)
2. Health check (`GET /health`)
3. Docker build verification

If you add tests, place them in a `tests/` directory and use `pytest`. The SQLite database can be swapped to an in-memory `:memory:` URL for unit tests by overriding `PATHS.DATABASE_URL`.

---

## Security Considerations

> **WARNING:** This application executes arbitrary CLI commands based on user-provided file types and configured tool templates. Exposing it publicly without authentication is dangerous.

### Defenses already in place
1. **Path traversal:** `ensure_path_is_safe()` resolves paths and checks `is_relative_to()` against allowed bases.
2. **Filename sanitization:** `werkzeug.utils.secure_filename()` + `html.escape()`.
3. **Command injection:** Templates are split with `shlex.split()` before `.format()` substitution. Only whitelisted placeholders are allowed.
4. **Resource limits:** Child processes get `RLIMIT_CPU=6000s` and `RLIMIT_AS=4GB` via `_limit_resources_preexec()`.
5. **File size limits:** `max_file_size_mb` in app settings (default 2000 MB).
6. **Extension allowlist:** `allowed_all_extensions` can restrict uploads.
7. **Auth:** `LOCAL_ONLY=True` (default) bypasses auth for local use. Set `LOCAL_ONLY=False` + OIDC for multi-user/production.

### What to watch when modifying
- Never pass unsanitized user input directly into `subprocess` calls.
- Never bypass `ensure_path_is_safe()` for download or delete routes.
- If adding new command template placeholders, update `ALLOWED_VARS` in `validate_and_build_command()`.
- If adding new admin endpoints, decorate with `Depends(require_admin)`.

---

## Deployment Notes

### Docker Multi-Stage Targets
- `full-final` — All tools including TeX/LaTeX (~1.5 GB+).
- `small-final` — Omits TeX and large ML extras.
- `cuda-final` — CUDA 12.1 runtime for GPU transcription.

### Runtime Layout (container)
```
/app
├── main.py
├── config/settings.yml
├── uploads/
├── processed/
├── models/tts/          # Piper voices
├── models/tts/kokoro/   # Kokoro ONNX model
├── jobs.db
└── huey.db
```

### Supervisor Processes
- **gunicorn:** 4 workers, 2 threads, Uvicorn worker class, port 8000.
- **huey_consumer:** 4 workers consuming from `main.huey`.

Both log to stdout/stderr (captured by Docker).

---

## Common Pitfalls for Agents

1. **Do not edit `settings.default.yml` at runtime.** The app copies it once; subsequent changes belong in `config/settings.yml` via the `/settings/save` endpoint or direct file edit.
2. **The frontend polls; it does NOT use WebSockets by default.** `ENABLE_WEBSOCKETS = False` in `main.py`. Polling is done via `POST /api/v1/jobs/status`.
3. **ZIP uploads create parent/child jobs.** A ZIP upload creates one `unzip` parent job and many child jobs. The parent completes only when all children finish.
4. **Whisper models are cached and shared across workers.** The cache eviction thread runs in the main process; model objects live in memory.
5. **Huey uses SQLite.** Do not run the consumer and the web app from different filesystem views (e.g., different containers without shared volume) or tasks will not be picked up.
6. **Kokoro TTS models auto-download on startup** if `DOWNLOAD_KOKORO_ON_STARTUP=true` and `kokoro-tts` CLI is available. The download is ~350 MB.
7. **Adding a new conversion tool** only requires editing `settings.default.yml` (or `settings.yml`) with `command_template`, `supported_input`, and `formats`. No code changes needed unless the tool requires special argument parsing (like `ghostscript_pdf` or `sox`).

---

## Useful Files for Quick Reference

| File | Purpose |
|------|---------|
| `main.py` | All backend logic |
| `settings.default.yml` | Default tool configs and formats |
| `templates/index.html` | Main UI |
| `static/js/script.js` | Frontend polling, drag-and-drop, job table |
| `Dockerfile` | Build stages and system dependencies |
| `supervisor.conf` | Container process orchestration |
