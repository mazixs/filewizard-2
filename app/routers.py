import asyncio
import json
import logging
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import httpx
import yaml
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.templating import Jinja2Templates
from sqlalchemy import text

from app.config import APP_CONFIG, LOCAL_ONLY_MODE, PATHS, initialize_settings_file, load_app_config, logger
from app.database import Job, Session, SessionLocal, engine, get_db
from app.database import FinalizeUploadPayload, JobSchema, JobSelection, JobStatusRequest
from app.security import (
    ensure_path_is_safe,
    get_current_user,
    is_admin,
    require_admin,
    require_user,
    sanitize_filename,
    validate_file_type,
)
from app.services import (
    create_job,
    get_job,
    get_jobs,
    run_command,
    save_upload_file,
    send_webhook_notification,
    update_job_status,
)
from app.services import (
    list_kokoro_languages_cli,
    list_kokoro_voices_cli,
    safe_get_voices,
)
from app.tasks import (
    dispatch_single_file_job,
    huey,
    run_conversion_task,
    run_image_ocr_task,
    run_pdf_ocr_task,
    run_transcription_task,
    run_tts_task,
    _whisper_cache_cleanup_worker,
)

# --- Templates ---
templates = Jinja2Templates(directory=str(PATHS.BASE_DIR / "templates"))

# --- Router ---
router = APIRouter()

# --- Auth ---
http_bearer = HTTPBearer()


def require_api_user(request: Request, creds: HTTPAuthorizationCredentials = Depends(http_bearer)):
    if LOCAL_ONLY_MODE:
        return {"sub": "local_api_user", "email": "local@api.user.com", "name": "Local API User"}
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    # OIDC token validation would go here
    raise HTTPException(status_code=501, detail="OIDC API auth not implemented in this simplified version")


def check_oidc_availability():
    return LOCAL_ONLY_MODE


# --- File helpers ---

def is_allowed_file(filename: str, allowed_extensions: set) -> bool:
    if not allowed_extensions:
        return True
    return Path(filename).suffix.lower() in allowed_extensions


def is_allowed_callback_url(url: str, allowed: List[str]) -> bool:
    if not allowed:
        return False
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        for a in allowed:
            ap = urlparse(a)
            if ap.scheme and ap.netloc:
                if parsed.scheme == ap.scheme and parsed.netloc == ap.netloc:
                    return True
            else:
                if url.startswith(a):
                    return True
        return False
    except Exception:
        return False


def _parse_tool_and_task_key(output_format: str, all_tool_keys: list) -> tuple[str, str]:
    for tool_key in sorted(all_tool_keys, key=len, reverse=True):
        if output_format.startswith(tool_key + "_"):
            task_key = output_format[len(tool_key) + 1 :]
            return tool_key, task_key
    raise ValueError(f"Could not determine tool from output_format: {output_format}")


# --- Chunked Uploads ---

async def _stitch_chunks(temp_dir: Path, final_path: Path, total_chunks: int):
    ensure_path_is_safe(temp_dir, [PATHS.CHUNK_TMP_DIR])
    ensure_path_is_safe(final_path, [PATHS.UPLOADS_DIR])

    def do_stitch():
        with open(final_path, "wb") as final_file:
            for i in range(total_chunks):
                chunk_path = temp_dir / f"{i}.chunk"
                if not chunk_path.exists():
                    raise FileNotFoundError(f"Upload failed: missing chunk {i}")
                with open(chunk_path, "rb") as chunk_file:
                    shutil.copyfileobj(chunk_file, final_file)

    from fastapi.concurrency import run_in_threadpool

    try:
        await run_in_threadpool(do_stitch)
    except FileNotFoundError as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/upload/chunk")
async def upload_chunk(
    chunk: UploadFile = File(...),
    upload_id: str = Form(...),
    chunk_number: int = Form(...),
    user: dict = Depends(require_user),
):
    safe_upload_id = sanitize_filename(upload_id)
    temp_dir = ensure_path_is_safe(PATHS.CHUNK_TMP_DIR / safe_upload_id, [PATHS.CHUNK_TMP_DIR])
    temp_dir.mkdir(exist_ok=True)
    chunk_path = temp_dir / f"{chunk_number}.chunk"

    def save_chunk_sync():
        try:
            with open(chunk_path, "wb") as buffer:
                shutil.copyfileobj(chunk.file, buffer)
        finally:
            chunk.file.close()

    from fastapi.concurrency import run_in_threadpool

    await run_in_threadpool(save_chunk_sync)
    return JSONResponse({"message": f"Chunk {chunk_number} for {safe_upload_id} uploaded."})


@router.post("/upload/finalize", response_model=JobSchema, status_code=202)
async def finalize_upload(
    request: Request,
    payload: FinalizeUploadPayload,
    user: dict = Depends(require_user),
    db: Session = Depends(get_db),
):
    safe_upload_id = sanitize_filename(payload.upload_id)
    temp_dir = ensure_path_is_safe(PATHS.CHUNK_TMP_DIR / safe_upload_id, [PATHS.CHUNK_TMP_DIR])
    if not temp_dir.is_dir():
        raise HTTPException(status_code=404, detail="Upload session not found or already finalized.")

    webhook_config = APP_CONFIG.get("webhook_settings", {})
    if payload.callback_url and not is_allowed_callback_url(
        payload.callback_url, webhook_config.get("allowed_callback_urls", [])
    ):
        raise HTTPException(status_code=400, detail="Provided callback_url is not allowed.")

    allowed_extensions = APP_CONFIG.get("app_settings", {}).get("allowed_all_extensions", set())
    if not validate_file_type(payload.original_filename, allowed_extensions):
        raise HTTPException(
            status_code=400, detail=f"File type '{Path(payload.original_filename).suffix}' not allowed."
        )

    job_id = uuid.uuid4().hex
    safe_filename = sanitize_filename(payload.original_filename)
    final_path = PATHS.UPLOADS_DIR / f"{Path(safe_filename).stem}_{job_id}{Path(safe_filename).suffix}"
    await _stitch_chunks(temp_dir, final_path, payload.total_chunks)

    base_url = str(request.base_url)
    tool, task_key = None, None
    if payload.task_type == "conversion":
        try:
            all_tools = APP_CONFIG.get("conversion_tools", {}).keys()
            tool, task_key = _parse_tool_and_task_key(payload.output_format, list(all_tools))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid or missing output_format for conversion.")

    if tool == "pandoc_academic":
        dispatch_single_file_job(
            payload.original_filename,
            str(final_path),
            "conversion",
            user,
            db,
            APP_CONFIG,
            base_url,
            job_id=job_id,
            options={"output_format": payload.output_format},
        )
    elif Path(safe_filename).suffix.lower() == ".zip":
        job_data = dict(
            id=job_id,
            user_id=user["sub"],
            task_type="unzip",
            original_filename=payload.original_filename,
            input_filepath=str(final_path),
            input_filesize=final_path.stat().st_size,
        )
        from app.database import JobCreate
        create_job(db=db, job=JobCreate(**job_data))
        sub_task_options = {
            "model_size": payload.model_size,
            "model_name": payload.model_name,
            "output_format": payload.output_format,
        }
        from app.tasks import unzip_and_dispatch_task
        unzip_and_dispatch_task(
            job_id, str(final_path), payload.task_type, sub_task_options, user, APP_CONFIG, base_url
        )
    else:
        options = {
            "model_size": payload.model_size,
            "model_name": payload.model_name,
            "output_format": payload.output_format,
            "generate_timestamps": payload.generate_timestamps,
        }
        dispatch_single_file_job(
            payload.original_filename, str(final_path), payload.task_type, user, db, APP_CONFIG, base_url, job_id=job_id, options=options
        )

    db.flush()
    db_job = get_job(db, job_id)
    if not db_job:
        raise HTTPException(status_code=500, detail="Job was created but could not be retrieved.")
    return db_job


# --- Legacy Routes ---

@router.post("/transcribe-audio", status_code=202)
async def submit_audio_transcription(
    request: Request,
    file: UploadFile = File(...),
    model_size: str = Form("base"),
    generate_timestamps: bool = Form(False),
    db: Session = Depends(get_db),
    user: dict = Depends(require_user),
):
    allowed = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}
    if not is_allowed_file(file.filename, allowed):
        raise HTTPException(status_code=400, detail="Invalid audio file type.")
    whisper_config = APP_CONFIG.get("transcription_settings", {}).get("whisper", {})
    if model_size not in whisper_config.get("allowed_models", []):
        raise HTTPException(status_code=400, detail=f"Invalid model size: {model_size}.")

    job_id = uuid.uuid4().hex
    safe_basename = sanitize_filename(file.filename)
    stem, suffix = Path(safe_basename).stem, Path(safe_basename).suffix
    upload_path = PATHS.UPLOADS_DIR / f"{stem}_{job_id}{suffix}"
    output_suffix = ".srt" if generate_timestamps else ".txt"
    processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}{output_suffix}"
    input_size = await save_upload_file(file, upload_path)
    base_url = str(request.base_url)

    from app.database import JobCreate
    job_data = JobCreate(
        id=job_id,
        user_id=user["sub"],
        task_type="transcription",
        original_filename=file.filename,
        input_filepath=str(upload_path),
        input_filesize=input_size,
        processed_filepath=str(processed_path),
    )
    new_job = create_job(db=db, job=job_data)
    run_transcription_task(
        new_job.id,
        str(upload_path),
        str(processed_path),
        model_size,
        whisper_settings=whisper_config,
        app_config=APP_CONFIG,
        base_url=base_url,
        generate_timestamps=generate_timestamps,
    )
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}


@router.post("/convert-file", status_code=202)
async def submit_file_conversion(
    request: Request,
    file: UploadFile = File(...),
    output_format: str = Form(...),
    db: Session = Depends(get_db),
    user: dict = Depends(require_user),
):
    allowed_exts = APP_CONFIG.get("app_settings", {}).get("allowed_all_extensions", set())
    if not is_allowed_file(file.filename, allowed_exts):
        raise HTTPException(status_code=400, detail=f"File type '{Path(file.filename).suffix}' not allowed.")
    conversion_tools = APP_CONFIG.get("conversion_tools", {})
    try:
        tool, task_key = output_format.split("_", 1)
        if tool not in conversion_tools:
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid output format selected.")

    job_id = uuid.uuid4().hex
    safe_basename = sanitize_filename(file.filename)
    original_stem = Path(safe_basename).stem
    target_ext = task_key.split("_")[0]
    if tool == "ghostscript_pdf":
        target_ext = "pdf"
    upload_path = PATHS.UPLOADS_DIR / f"{original_stem}_{job_id}{Path(safe_basename).suffix}"
    processed_path = PATHS.PROCESSED_DIR / f"{original_stem}_{job_id}.{target_ext}"
    input_size = await save_upload_file(file, upload_path)
    base_url = str(request.base_url)

    from app.database import JobCreate
    job_data = JobCreate(
        id=job_id,
        user_id=user["sub"],
        task_type="conversion",
        original_filename=file.filename,
        input_filepath=str(upload_path),
        input_filesize=input_size,
        processed_filepath=str(processed_path),
    )
    new_job = create_job(db=db, job=job_data)
    run_conversion_task(new_job.id, str(upload_path), str(processed_path), tool, task_key, conversion_tools, APP_CONFIG, base_url)
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}


@router.post("/ocr-pdf", status_code=202)
async def submit_pdf_ocr(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: dict = Depends(require_user),
):
    if not is_allowed_file(file.filename, {".pdf"}):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    job_id = uuid.uuid4().hex
    safe_basename = sanitize_filename(file.filename)
    unique = f"{Path(safe_basename).stem}_{job_id}{Path(safe_basename).suffix}"
    upload_path = PATHS.UPLOADS_DIR / unique
    processed_path = PATHS.PROCESSED_DIR / unique
    input_size = await save_upload_file(file, upload_path)
    base_url = str(request.base_url)

    from app.database import JobCreate
    job_data = JobCreate(
        id=job_id,
        user_id=user["sub"],
        task_type="ocr",
        original_filename=file.filename,
        input_filepath=str(upload_path),
        input_filesize=input_size,
        processed_filepath=str(processed_path),
    )
    new_job = create_job(db=db, job=job_data)
    ocr_settings = APP_CONFIG.get("ocr_settings", {}).get("ocrmypdf", {})
    run_pdf_ocr_task(new_job.id, str(upload_path), str(processed_path), ocr_settings, APP_CONFIG, base_url)
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}


@router.post("/ocr-image", status_code=202)
async def submit_image_ocr(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: dict = Depends(require_user),
):
    allowed_exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    if not is_allowed_file(file.filename, allowed_exts):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload PNG, JPG, or TIFF.")
    job_id = uuid.uuid4().hex
    safe_basename = sanitize_filename(file.filename)
    file_ext = Path(safe_basename).suffix
    unique = f"{Path(safe_basename).stem}_{job_id}{file_ext}"
    upload_path = PATHS.UPLOADS_DIR / unique
    processed_path = PATHS.PROCESSED_DIR / f"{Path(safe_basename).stem}_{job_id}.pdf"
    input_size = await save_upload_file(file, upload_path)
    base_url = str(request.base_url)

    from app.database import JobCreate
    job_data = JobCreate(
        id=job_id,
        user_id=user["sub"],
        task_type="ocr-image",
        original_filename=file.filename,
        input_filepath=str(upload_path),
        input_filesize=input_size,
        processed_filepath=str(processed_path),
    )
    new_job = create_job(db=db, job=job_data)
    run_image_ocr_task(new_job.id, str(upload_path), str(processed_path), APP_CONFIG, base_url)
    return {"job_id": new_job.id, "status": new_job.status, "status_url": f"/job/{new_job.id}"}


# --- API V1 ---

@router.get("/api/v1/tts-voices")
async def get_tts_voices_list(user: dict = Depends(require_user)):
    kokoro_available = shutil.which("kokoro-tts") is not None
    piper_available = False
    try:
        import piper
        piper_available = True
    except ImportError:
        pass

    if not piper_available and not kokoro_available:
        return JSONResponse(content={"error": "TTS feature not configured."}, status_code=501)

    all_voices = []
    try:
        if piper_available:
            piper_voices = safe_get_voices(PATHS.TTS_MODELS_DIR)
            for voice in piper_voices:
                voice["id"] = f"piper/{voice.get('id')}"
                voice["name"] = f"Piper: {voice.get('name', voice.get('id'))}"
            all_voices.extend(piper_voices)
        if kokoro_available:
            kokoro_voices = list_kokoro_voices_cli()
            kokoro_langs = list_kokoro_languages_cli()
            for lang in kokoro_langs:
                for voice in kokoro_voices:
                    all_voices.append({"id": f"kokoro/{lang}/{voice}", "name": f"Kokoro ({lang}): {voice}", "local": False})
        return sorted(all_voices, key=lambda x: x["name"])
    except Exception as e:
        logger.exception("Could not fetch TTS voices.")
        raise HTTPException(status_code=500, detail=f"Could not retrieve voices: {e}")


@router.post("/api/v1/process", status_code=202, tags=["Webhook API"])
async def api_process_file(
    request: Request,
    file: UploadFile = File(...),
    task_type: str = Form(...),
    callback_url: str = Form(...),
    model_size: str = Form("base"),
    model_name: str = Form(None),
    output_format: str = Form(None),
    generate_timestamps: bool = Form(False),
    db: Session = Depends(get_db),
    user: dict = Depends(require_api_user),
):
    webhook_config = APP_CONFIG.get("webhook_settings", {})
    if not webhook_config.get("enabled", False):
        raise HTTPException(status_code=403, detail="Webhook processing is disabled.")
    if not is_allowed_callback_url(callback_url, webhook_config.get("allowed_callback_urls", [])):
        raise HTTPException(status_code=400, detail="Provided callback_url is not allowed.")

    job_id = uuid.uuid4().hex
    safe_basename = sanitize_filename(file.filename)
    stem, suffix = Path(safe_basename).stem, Path(safe_basename).suffix
    upload_path = PATHS.UPLOADS_DIR / f"{stem}_{job_id}{suffix}"
    input_size = await save_upload_file(file, upload_path)
    base_url = str(request.base_url)

    from app.database import JobCreate
    job_data_args = {
        "id": job_id,
        "user_id": user["sub"],
        "original_filename": file.filename,
        "input_filepath": str(upload_path),
        "input_filesize": input_size,
        "callback_url": callback_url,
        "task_type": task_type,
    }

    if task_type == "transcription":
        whisper_config = APP_CONFIG.get("transcription_settings", {}).get("whisper", {})
        if model_size not in whisper_config.get("allowed_models", []):
            raise HTTPException(status_code=400, detail=f"Invalid model_size '{model_size}'")
        out_suffix = ".srt" if generate_timestamps else ".txt"
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}{out_suffix}"
        job_data_args["processed_filepath"] = str(processed_path)
        create_job(db=db, job=JobCreate(**job_data_args))
        run_transcription_task(
            job_id, str(upload_path), str(processed_path), model_size, whisper_config, APP_CONFIG, base_url, generate_timestamps=generate_timestamps
        )
    elif task_type == "tts":
        if not is_allowed_file(file.filename, {".txt"}):
            raise HTTPException(status_code=400, detail="TTS requires .txt file.")
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name is required for TTS.")
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.wav"
        job_data_args["processed_filepath"] = str(processed_path)
        create_job(db=db, job=JobCreate(**job_data_args))
        run_tts_task(job_id, str(upload_path), str(processed_path), model_name, APP_CONFIG.get("tts_settings", {}), APP_CONFIG, base_url)
    elif task_type == "conversion":
        if not output_format:
            raise HTTPException(status_code=400, detail="output_format is required for conversion.")
        conversion_tools = APP_CONFIG.get("conversion_tools", {})
        try:
            tool, task_key = output_format.split("_", 1)
            if tool not in conversion_tools:
                raise ValueError()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid output_format.")
        target_ext = task_key.split("_")[0]
        if tool == "ghostscript_pdf":
            target_ext = "pdf"
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.{target_ext}"
        job_data_args["processed_filepath"] = str(processed_path)
        create_job(db=db, job=JobCreate(**job_data_args))
        run_conversion_task(job_id, str(upload_path), str(processed_path), tool, task_key, conversion_tools, APP_CONFIG, base_url)
    elif task_type == "ocr":
        if not is_allowed_file(file.filename, {".pdf"}):
            raise HTTPException(status_code=400, detail="OCR requires .pdf file.")
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}{suffix}"
        job_data_args["processed_filepath"] = str(processed_path)
        create_job(db=db, job=JobCreate(**job_data_args))
        run_pdf_ocr_task(job_id, str(upload_path), str(processed_path), APP_CONFIG.get("ocr_settings", {}).get("ocrmypdf", {}), APP_CONFIG, base_url)
    elif task_type == "ocr-image":
        if not is_allowed_file(file.filename, {".png", ".jpg", ".jpeg", ".tiff", ".tif"}):
            raise HTTPException(status_code=400, detail="Invalid file type for ocr-image.")
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.txt"
        job_data_args["processed_filepath"] = str(processed_path)
        create_job(db=db, job=JobCreate(**job_data_args))
        run_image_ocr_task(job_id, str(upload_path), str(processed_path), APP_CONFIG, base_url)
    else:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid task_type: '{task_type}'")

    return {"job_id": job_id, "status": "pending"}


@router.post("/api/v1/upload/chunk", tags=["Webhook API"])
async def api_upload_chunk(
    chunk: UploadFile = File(...),
    upload_id: str = Form(...),
    chunk_number: int = Form(...),
    user: dict = Depends(require_api_user),
):
    webhook_config = APP_CONFIG.get("webhook_settings", {})
    if not webhook_config.get("enabled", False) or not webhook_config.get("allow_chunked_api_uploads", False):
        raise HTTPException(status_code=403, detail="Chunked API uploads are disabled.")
    return await upload_chunk(chunk, upload_id, chunk_number, user)


@router.post("/api/v1/upload/finalize", status_code=202, tags=["Webhook API"])
async def api_finalize_upload(
    request: Request,
    payload: FinalizeUploadPayload,
    user: dict = Depends(require_api_user),
    db: Session = Depends(get_db),
):
    webhook_config = APP_CONFIG.get("webhook_settings", {})
    if not webhook_config.get("enabled", False) or not webhook_config.get("allow_chunked_api_uploads", False):
        raise HTTPException(status_code=403, detail="Chunked API uploads are disabled.")
    if payload.callback_url and not is_allowed_callback_url(
        payload.callback_url, webhook_config.get("allowed_callback_urls", [])
    ):
        raise HTTPException(status_code=400, detail="Provided callback_url is not allowed.")
    return await finalize_upload(request, payload, user, db)


# --- Pages ---

@router.get("/")
async def get_index(request: Request):
    user = get_current_user(request)
    admin_status = is_admin(request)
    whisper_models = APP_CONFIG.get("transcription_settings", {}).get("whisper", {}).get("allowed_models", [])
    conversion_tools = APP_CONFIG.get("conversion_tools", {})
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "user": user,
            "is_admin": admin_status,
            "whisper_models": sorted(list(whisper_models)),
            "conversion_tools": conversion_tools,
            "local_only_mode": LOCAL_ONLY_MODE,
        },
    )


@router.get("/settings")
async def get_settings_page(request: Request):
    user = get_current_user(request)
    admin_status = is_admin(request)
    current_config = APP_CONFIG.copy()

    for key in ["app_settings", "transcription_settings", "tts_settings", "conversion_tools", "ocr_settings", "auth_settings", "webhook_settings"]:
        if key not in current_config:
            if key == "transcription_settings":
                current_config[key] = {"whisper": {}}
            else:
                current_config[key] = {}

    for tool_id, tool_config in current_config.get("conversion_tools", {}).items():
        if "formats" in tool_config:
            fd = tool_config["formats"]
            if isinstance(fd, list):
                d = {}
                for item in fd:
                    if isinstance(item, str) and ":" in item:
                        parts = item.split(":", 1)
                        d[parts[0].strip()] = parts[1].strip() if len(parts) > 1 else ""
                current_config["conversion_tools"][tool_id]["formats"] = d
            elif not isinstance(fd, dict):
                current_config["conversion_tools"][tool_id]["formats"] = {}

    config_source = "none"
    if PATHS.SETTINGS_FILE.exists():
        config_source = PATHS.SETTINGS_FILE.name
    elif PATHS.DEFAULT_SETTINGS_FILE.exists():
        config_source = PATHS.DEFAULT_SETTINGS_FILE.name

    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "request": request,
            "config": current_config,
            "config_source": config_source,
            "user": user,
            "is_admin": admin_status,
            "local_only_mode": LOCAL_ONLY_MODE,
        },
    )


# --- Settings Save ---

def deep_merge(source: dict, destination: dict) -> dict:
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination


def _preprocess_settings_for_saving(config: dict) -> dict:
    if "conversion_tools" in config and isinstance(config["conversion_tools"], dict):
        for tool_name, tool_config in config["conversion_tools"].items():
            if isinstance(tool_config, dict):
                if "command_template" in tool_config:
                    ct = tool_config["command_template"]
                    if isinstance(ct, list) and len(ct) == 1 and isinstance(ct[0], str):
                        tool_config["command_template"] = ct[0]
                if "formats" in tool_config:
                    fd = tool_config["formats"]
                    if isinstance(fd, list):
                        nd = {}
                        for item in fd:
                            if isinstance(item, str):
                                for line in item.split("\n"):
                                    line = line.strip()
                                    if ":" in line:
                                        k, v = line.split(":", 1)
                                        nd[k.strip()] = v.strip()
                        tool_config["formats"] = nd
                    elif isinstance(fd, str):
                        nd = {}
                        for line in fd.split("\n"):
                            line = line.strip()
                            if ":" in line:
                                k, v = line.split(":", 1)
                                nd[k.strip()] = v.strip()
                        tool_config["formats"] = nd
    return config


@router.post("/settings/save")
async def save_settings(
    request: Request,
    new_config_from_ui: dict = Body(...),
    admin: bool = Depends(require_admin),
):
    tmp_path = PATHS.SETTINGS_FILE.with_suffix(".tmp")
    user = get_current_user(request)
    try:
        if not new_config_from_ui:
            if PATHS.SETTINGS_FILE.exists():
                PATHS.SETTINGS_FILE.unlink()
                logger.info(f"Admin '{user.get('email')}' reverted to default settings.")
            load_app_config()
            return JSONResponse({"message": "Settings reverted to default."})

        processed = _preprocess_settings_for_saving(new_config_from_ui)
        try:
            with PATHS.SETTINGS_FILE.open("r", encoding="utf8") as f:
                current = yaml.safe_load(f) or {}
        except FileNotFoundError:
            current = {}

        merged = deep_merge(processed, current)
        with tmp_path.open("w", encoding="utf8") as f:
            yaml.safe_dump(merged, f, default_flow_style=False, sort_keys=False, width=float("inf"))
        tmp_path.replace(PATHS.SETTINGS_FILE)
        logger.info(f"Admin '{user.get('email')}' updated settings.yml.")
        load_app_config()
        return JSONResponse({"message": "Settings saved successfully."})
    except Exception as e:
        logger.exception(f"Failed to update settings for admin '{user.get('email')}'")
        if tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Could not save settings: {e}")


# --- Job Management ---

@router.post("/settings/clear-history")
async def clear_job_history(db: Session = Depends(get_db), user: dict = Depends(require_user)):
    try:
        num_deleted = db.query(Job).filter(Job.user_id == user["sub"]).delete()
        db.commit()
        logger.info(f"Cleared {num_deleted} jobs for user {user['sub']}.")
        return {"deleted_count": num_deleted}
    except Exception:
        db.rollback()
        logger.exception("Failed to clear job history")
        raise HTTPException(status_code=500, detail="Database error while clearing history.")


@router.post("/settings/delete-files")
async def delete_processed_files(db: Session = Depends(get_db), user: dict = Depends(require_user)):
    deleted_count, errors = 0, []
    for job in get_jobs(db, user_id=user["sub"]):
        if job.processed_filepath:
            try:
                p = ensure_path_is_safe(Path(job.processed_filepath), [PATHS.PROCESSED_DIR])
                if p.is_file():
                    p.unlink()
                    deleted_count += 1
            except Exception:
                errors.append(Path(job.processed_filepath).name)
                logger.exception(f"Could not delete file {Path(job.processed_filepath).name}")
    if errors:
        raise HTTPException(status_code=500, detail=f"Could not delete some files: {', '.join(errors)}")
    logger.info(f"Deleted {deleted_count} files for user {user['sub']}.")
    return {"deleted_count": deleted_count}


@router.post("/job/{job_id}/cancel", status_code=202)
async def cancel_job(job_id: str, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    job = get_job(db, job_id)
    if not job or job.user_id != user["sub"]:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status in ["pending", "processing"]:
        update_job_status(db, job_id, status="cancelled")
        return {"message": "Job cancellation requested."}
    raise HTTPException(status_code=400, detail=f"Job is already in a final state ({job.status}).")


@router.get("/jobs", response_model=List[JobSchema])
async def get_all_jobs(db: Session = Depends(get_db), user: dict = Depends(require_user)):
    return get_jobs(db, user_id=user["sub"])


@router.get("/job/{job_id}", response_model=JobSchema)
async def get_job_status(job_id: str, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    job = get_job(db, job_id)
    if not job or job.user_id != user["sub"]:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@router.post("/api/v1/jobs/status", response_model=List[JobSchema])
async def get_jobs_status(payload: JobStatusRequest, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    if not payload.job_ids:
        return []
    return db.query(Job).filter(Job.id.in_(payload.job_ids), Job.user_id == user["sub"]).all()


@router.get("/download/{filename}")
async def download_file(filename: str, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    file_path = ensure_path_is_safe(PATHS.PROCESSED_DIR / filename, [PATHS.PROCESSED_DIR])
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    job = db.query(Job).filter(Job.processed_filepath == str(file_path), Job.user_id == user.get("sub")).first()
    if not job:
        raise HTTPException(status_code=403, detail="You do not have permission to download this file.")
    download_filename = Path(job.original_filename).stem + Path(job.processed_filepath).suffix
    return FileResponse(path=file_path, filename=download_filename, media_type="application/octet-stream")


@router.post("/download/batch", response_class=StreamingResponse)
async def download_batch(payload: JobSelection, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    if not payload.job_ids:
        raise HTTPException(status_code=400, detail="No job IDs provided.")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for job_id in payload.job_ids:
            job = get_job(db, job_id)
            if job and job.user_id == user["sub"] and job.status == "completed" and job.processed_filepath:
                file_path = ensure_path_is_safe(Path(job.processed_filepath), [PATHS.PROCESSED_DIR])
                if file_path.exists():
                    download_filename = f"{Path(job.original_filename).stem}_{job_id}{file_path.suffix}"
                    zip_file.write(file_path, arcname=download_filename)
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f'attachment; filename="file-wizard-batch-{uuid.uuid4().hex[:8]}.zip"'},
    )


@router.get("/download/zip-batch/{job_id}", response_class=StreamingResponse)
async def download_zip_batch(job_id: str, db: Session = Depends(get_db), user: dict = Depends(require_user)):
    parent_job = get_job(db, job_id)
    if not parent_job or parent_job.user_id != user["sub"]:
        raise HTTPException(status_code=404, detail="Parent job not found.")
    if parent_job.task_type != "unzip":
        raise HTTPException(status_code=400, detail="This job is not a batch upload.")

    child_jobs = db.query(Job).filter(Job.parent_job_id == job_id, Job.status == "completed").all()
    if not child_jobs:
        raise HTTPException(status_code=404, detail="No completed sub-jobs found.")

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        files_added = 0
        for job in child_jobs:
            if job.processed_filepath:
                file_path = ensure_path_is_safe(Path(job.processed_filepath), [PATHS.PROCESSED_DIR])
                if file_path.exists():
                    zip_file.write(file_path, arcname=f"{Path(job.original_filename).stem}{file_path.suffix}")
                    files_added += 1
    if files_added == 0:
        raise HTTPException(status_code=404, detail="No processed files found.")
    zip_buffer.seek(0)
    batch_filename = f"{Path(parent_job.original_filename).stem}_processed.zip"
    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f'attachment; filename="{batch_filename}"'},
    )


@router.get("/api/v1/supported-formats/{file_extension}")
async def get_supported_formats_for_file_type(file_extension: str, user: dict = Depends(require_user)):
    if not file_extension.startswith("."):
        file_extension = "." + file_extension
    file_extension = file_extension.lower()
    conversion_tools = APP_CONFIG.get("conversion_tools", {})
    supported_formats = []
    for tool_name, tool_config in conversion_tools.items():
        supported_inputs = [ext.lower() for ext in tool_config.get("supported_input", [])]
        if file_extension in supported_inputs:
            for format_key, format_label in tool_config.get("formats", {}).items():
                supported_formats.append({
                    "value": f"{tool_name}_{format_key}",
                    "label": f"{tool_config['name']} - {format_label}",
                    "tool": tool_name,
                    "format": format_key,
                })
    return {"formats": supported_formats}


@router.get("/api/formats/count")
async def get_formats_count():
    try:
        with open(PATHS.DEFAULT_SETTINGS_FILE, "r") as f:
            settings = yaml.safe_load(f)
        input_formats = set()
        output_formats = set()
        for tool, config in settings.get("conversion_tools", {}).items():
            if "supported_input" in config:
                for fmt in config["supported_input"]:
                    input_formats.add(fmt)
            if "formats" in config:
                for fmt in config["formats"]:
                    output_formats.add(fmt)
        return {"input_formats_count": len(input_formats), "output_formats_count": len(output_formats)}
    except Exception as e:
        logger.error(f"Error counting formats: {e}")
        raise HTTPException(status_code=500, detail="Error counting formats")


@router.get("/health")
async def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        logger.exception("Health check failed")
        return JSONResponse({"ok": False}, status_code=500)
    return {"ok": True}


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(str(PATHS.BASE_DIR / "static" / "favicon.png"))
