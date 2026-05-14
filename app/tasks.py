import gc
import io
import os
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.config import APP_CONFIG, PATHS, logger
from app.database import SessionLocal
from app.security import ensure_path_is_safe, sanitize_filename
from app.services import (
    get_job,
    mark_job_as_completed,
    run_command,
    send_webhook_notification,
    update_job_status,
    validate_and_build_command,
    _update_parent_zip_job_progress,
)

# --- Huey Instance ---
from huey import SqliteHuey

huey = SqliteHuey(filename=PATHS.HUEY_DB_PATH)


async def download_kokoro_models_if_missing():
    """Checks for Kokoro TTS model files and downloads them if missing."""
    files_to_download = {
        "model": {
            "path": PATHS.KOKORO_MODEL_FILE,
            "url": "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx",
            "size": 325532387,
        },
        "voices": {
            "path": PATHS.KOKORO_VOICES_FILE,
            "url": "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin",
            "size": 26124436,
        },
    }
    import httpx

    async with httpx.AsyncClient() as client:
        for name, details in files_to_download.items():
            path, url, expected_size = details["path"], details["url"], details["size"]
            if path.exists() and path.stat().st_size == expected_size:
                logger.info(f"Found valid Kokoro TTS {name} file at {path}.")
                continue
            logger.info(f"Downloading Kokoro TTS {name} from {url}...")
            for attempt in range(3):
                try:
                    with path.open("wb") as f:
                        async with client.stream("GET", url, follow_redirects=True, timeout=300) as response:
                            response.raise_for_status()
                            total = 0
                            async for chunk in response.aiter_bytes():
                                f.write(chunk)
                                total += len(chunk)
                    if total == expected_size:
                        logger.info(f"Downloaded Kokoro TTS {name} to {path}.")
                        break
                    else:
                        logger.warning(f"Incomplete download. Expected {expected_size}, got {total}.")
                except Exception as e:
                    logger.error(f"Failed to download Kokoro TTS {name} (attempt {attempt + 1}): {e}")
                    if path.exists():
                        path.unlink(missing_ok=True)
                    await asyncio.sleep(5)
            else:
                logger.critical(f"Failed to download Kokoro TTS {name} after 3 attempts.")

# --- Model Caches ---
WHISPER_MODELS_CACHE: Dict[str, Any] = {}
PIPER_VOICES_CACHE: Dict[str, Any] = {}
WHISPER_MODELS_LAST_USED: Dict[str, float] = {}

_cache_lock = threading.Lock()
_model_locks: Dict[str, threading.Lock] = {}
_global_lock = threading.Lock()
_model_semaphore: Optional[threading.Semaphore] = None


def get_model_semaphore() -> threading.Semaphore:
    global _model_semaphore
    if _model_semaphore is None:
        concurrency = APP_CONFIG.get("app_settings", {}).get(
            "model_concurrency", int(os.environ.get("MODEL_CONCURRENCY", "1"))
        )
        _model_semaphore = threading.Semaphore(concurrency)
        logger.info(f"Model concurrency semaphore initialized with limit: {concurrency}")
    return _model_semaphore


def _get_or_create_model_lock(model_size: str) -> threading.Lock:
    if model_size in _model_locks:
        return _model_locks[model_size]
    with _global_lock:
        return _model_locks.setdefault(model_size, threading.Lock())


def _whisper_cache_cleanup_worker():
    """Periodically unloads inactive Whisper models."""
    while True:
        app_settings = APP_CONFIG.get("app_settings", {})
        check_interval = app_settings.get("cache_check_interval", 300)
        inactivity_timeout = app_settings.get("model_inactivity_timeout", 1800)
        time.sleep(check_interval)

        with _cache_lock:
            expired = [
                model_size
                for model_size, last_used in WHISPER_MODELS_LAST_USED.items()
                if time.time() - last_used > inactivity_timeout
            ]
            for model_size in expired:
                model_lock = _get_or_create_model_lock(model_size)
                with model_lock:
                    if model_size in WHISPER_MODELS_CACHE:
                        logger.info(f"Unloading inactive Whisper model: {model_size}")
                        WHISPER_MODELS_CACHE.pop(model_size, None)
                        WHISPER_MODELS_LAST_USED.pop(model_size, None)
        gc.collect()


def get_whisper_model(model_size: str, whisper_settings: dict) -> Any:
    with _cache_lock:
        if model_size in WHISPER_MODELS_CACHE:
            WHISPER_MODELS_LAST_USED[model_size] = time.time()
            return WHISPER_MODELS_CACHE[model_size]

    model_lock = _get_or_create_model_lock(model_size)
    with model_lock:
        with _cache_lock:
            if model_size in WHISPER_MODELS_CACHE:
                WHISPER_MODELS_LAST_USED[model_size] = time.time()
                return WHISPER_MODELS_CACHE[model_size]

        logger.info(f"Loading Whisper model '{model_size}'...")
        try:
            from faster_whisper import WhisperModel

            device = whisper_settings.get("device", "cpu")
            compute_type = whisper_settings.get("compute_type", "int8")
            device_index = whisper_settings.get("device_index", 0)
            model = WhisperModel(
                model_size,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                cpu_threads=max(1, os.cpu_count() // 2),
                num_workers=1,
            )
            with _cache_lock:
                WHISPER_MODELS_CACHE[model_size] = model
                WHISPER_MODELS_LAST_USED[model_size] = time.time()
            logger.info(f"Model '{model_size}' loaded (device={device}, compute={compute_type})")
            return model
        except Exception as e:
            logger.error(f"Model '{model_size}' failed to load: {e}", exc_info=True)
            raise RuntimeError(f"Whisper model initialization failed: {e}") from e


# --- Piper TTS Helpers ---

def _find_model_files(model_name: str, model_dir: Path):
    from app.security import sanitize_filename

    safe_name = sanitize_filename(model_name)
    candidates = [model_dir / safe_name, model_dir / f"{safe_name}.onnx"]
    for base in candidates:
        onnx = base.with_suffix(".onnx")
        config_json = base.with_suffix(".onnx.json")
        if not config_json.exists():
            config_json = Path(str(base) + ".onnx.json")
        if onnx.exists() and config_json.exists():
            return onnx, config_json
    return None, None


def get_piper_voice(model_name: str, tts_settings: dict | None):
    try:
        from piper import PiperVoice
        from piper.synthesis import SynthesisConfig
        from piper.download import get_voices, ensure_voice_exists, find_voice
    except ImportError:
        raise RuntimeError("piper-tts is not installed.")

    model_dir = Path(tts_settings.get("model_dir", PATHS.TTS_MODELS_DIR))
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_name in PIPER_VOICES_CACHE:
        return PIPER_VOICES_CACHE[model_name]

    with get_model_semaphore():
        if model_name in PIPER_VOICES_CACHE:
            return PIPER_VOICES_CACHE[model_name]

        onnx_path, config_path = _find_model_files(model_name, model_dir)
        if not (onnx_path and config_path) and ensure_voice_exists and find_voice:
            try:
                voices_info = get_voices(str(model_dir), update_voices=True)
            except TypeError:
                voices_info = get_voices(str(model_dir))
            ensure_voice_exists(model_name, [model_dir], model_dir, voices_info)
            onnx_path, config_path = find_voice(model_name, [model_dir])

        if not (onnx_path and config_path):
            onnx_path, config_path = _find_model_files(model_name, model_dir)

        if not (onnx_path and config_path):
            raise RuntimeError(f"Piper voice files for '{model_name}' are missing.")

        use_cuda = bool(tts_settings.get("use_cuda", False))
        voice = PiperVoice.load(str(onnx_path), config_path=str(config_path), use_cuda=use_cuda)
        PIPER_VOICES_CACHE[model_name] = voice
        return voice


# --- SRT Formatter ---

class SrtFormatter:
    def format_segment(self, segment) -> str:
        def _ts(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        start = _ts(segment.start)
        end = _ts(segment.end)
        return f"{start} --> {end}\n{segment.text.strip()}\n\n"


# =============================================================================
# DISPATCH HELPER
# =============================================================================

def dispatch_single_file_job(
    original_filename: str,
    input_filepath: str,
    task_type: str,
    user: dict,
    db: Session,
    app_config: Dict,
    base_url: str,
    job_id: str | None = None,
    options: Dict | None = None,
    parent_job_id: str | None = None,
):
    from app.database import JobCreate
    from app.services import create_job

    options = options or {}
    if not job_id:
        job_id = uuid.uuid4().hex
    safe_basename = sanitize_filename(original_filename)
    stem = Path(safe_basename).stem
    suffix = Path(safe_basename).suffix

    job_data = JobCreate(
        id=job_id,
        user_id=user["sub"],
        parent_job_id=parent_job_id,
        task_type=task_type,
        original_filename=original_filename,
        input_filepath=input_filepath,
    )

    if task_type == "transcription":
        model_size = options.get("model_size", "base")
        generate_timestamps = options.get("generate_timestamps", False)
        out_suffix = ".srt" if generate_timestamps else ".txt"
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}{out_suffix}"
        job_data.processed_filepath = str(processed_path)
        create_job(db=db, job=job_data)
        whisper_config = app_config.get("transcription_settings", {}).get("whisper", {})
        run_transcription_task(
            job_id,
            input_filepath,
            str(processed_path),
            model_size,
            whisper_config,
            app_config,
            base_url,
            generate_timestamps=generate_timestamps,
        )

    elif task_type == "tts":
        model_name = options.get("model_name", "")
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.wav"
        job_data.processed_filepath = str(processed_path)
        create_job(db=db, job=job_data)
        tts_config = app_config.get("tts_settings", {})
        run_tts_task(job_id, input_filepath, str(processed_path), model_name, tts_config, app_config, base_url)

    elif task_type == "conversion":
        output_format = options.get("output_format", "")
        conversion_tools = app_config.get("conversion_tools", {})
        tool, task_key = __import__("app.services").services._parse_tool_and_task_key(
            output_format, list(conversion_tools.keys())
        )
        target_ext = task_key.split("_")[0]
        if tool == "ghostscript_pdf":
            target_ext = "pdf"
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.{target_ext}"
        job_data.processed_filepath = str(processed_path)
        create_job(db=db, job=job_data)
        run_conversion_task(
            job_id, input_filepath, str(processed_path), tool, task_key, conversion_tools, app_config, base_url
        )

    elif task_type == "ocr":
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}{suffix}"
        job_data.processed_filepath = str(processed_path)
        create_job(db=db, job=job_data)
        ocr_settings = app_config.get("ocr_settings", {}).get("ocrmypdf", {})
        run_pdf_ocr_task(job_id, input_filepath, str(processed_path), ocr_settings, app_config, base_url)

    elif task_type == "ocr-image":
        processed_path = PATHS.PROCESSED_DIR / f"{stem}_{job_id}.pdf"
        job_data.processed_filepath = str(processed_path)
        create_job(db=db, job=job_data)
        run_image_ocr_task(job_id, input_filepath, str(processed_path), app_config, base_url)


# =============================================================================
# HUEY TASKS
# =============================================================================


@huey.task()
def run_transcription_task(
    job_id: str,
    input_path_str: str,
    output_path_str: str,
    model_size: str,
    whisper_settings: dict,
    app_config: dict,
    base_url: str,
    generate_timestamps: bool = False,
):
    db = SessionLocal()
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    try:
        job = get_job(db, job_id)
        if not job or job.status == "cancelled":
            return
        update_job_status(db, job_id, "processing", progress=0)

        model = get_whisper_model(model_size, whisper_settings)
        segments_generator, info = model.transcribe(str(input_path), beam_size=5)

        last_update = time.time()
        tmp_output = output_path.with_name(f"{output_path.stem}.tmp-{uuid.uuid4().hex}{output_path.suffix}")
        preview_segments = []
        PREVIEW_MAX = 1000
        current_preview_len = 0

        with tmp_output.open("w", encoding="utf-8") as f:
            if generate_timestamps:
                formatter = SrtFormatter()
                for segment in segments_generator:
                    f.write(formatter.format_segment(segment))
                    txt = segment.text.strip()
                    if current_preview_len < PREVIEW_MAX:
                        preview_segments.append(txt)
                        current_preview_len += len(txt)
                    if time.time() - last_update > 1:
                        last_update = time.time()
                        if info.duration > 0:
                            progress = int((segment.end / info.duration) * 100)
                            update_job_status(db, job_id, "processing", progress=progress)
            else:
                for segment in segments_generator:
                    txt = segment.text.strip()
                    f.write(txt + "\n")
                    if current_preview_len < PREVIEW_MAX:
                        preview_segments.append(txt)
                        current_preview_len += len(txt)
                    if time.time() - last_update > 1:
                        last_update = time.time()
                        if info.duration > 0:
                            progress = int((segment.end / info.duration) * 100)
                            update_job_status(db, job_id, "processing", progress=progress)

        tmp_output.replace(output_path)
        preview = " ".join(preview_segments)[:PREVIEW_MAX]
        mark_job_as_completed(db, job_id, output_filepath_str=output_path_str, preview=preview)
        logger.info(f"Transcription job {job_id} completed.")
    except Exception as e:
        logger.exception(f"Transcription job {job_id} failed: {e}")
        update_job_status(db, job_id, "failed", error=str(e))
    finally:
        if "tmp_output" in locals() and tmp_output.exists():
            tmp_output.unlink(missing_ok=True)
        try:
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR, PATHS.CHUNK_TMP_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        db.close()
        _update_parent_if_needed(job_id)
        send_webhook_notification(job_id, app_config, base_url)


@huey.task()
def run_tts_task(
    job_id: str,
    input_path_str: str,
    output_path_str: str,
    model_name: str,
    tts_settings: dict,
    app_config: dict,
    base_url: str,
):
    db = SessionLocal()
    input_path = Path(input_path_str)
    try:
        job = get_job(db, job_id)
        if not job or job.status == "cancelled":
            return
        update_job_status(db, job_id, "processing")

        engine, actual_model = "piper", model_name
        if "/" in model_name:
            engine, actual_model = model_name.split("/", 1)

        out_path = Path(output_path_str)
        tmp_out = out_path.with_name(f"{out_path.stem}.tmp-{uuid.uuid4().hex}{out_path.suffix}")

        if engine == "piper":
            voice = get_piper_voice(actual_model, tts_settings.get("piper"))
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()
            import wave

            with wave.open(str(tmp_out), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(voice.config.sample_rate)
                voice.synthesize_wav(text, wav_file)

        elif engine == "kokoro":
            kokoro_settings = tts_settings.get("kokoro", {})
            template = kokoro_settings.get("command_template")
            if not template:
                raise ValueError("Kokoro TTS command_template is not defined.")
            try:
                lang, voice_name = actual_model.split("/", 1)
            except ValueError:
                raise ValueError(f"Invalid Kokoro model format. Expected 'lang/voice', got '{actual_model}'.")
            mapping = {
                "input": str(input_path),
                "output": str(tmp_out),
                "model_path": str(PATHS.KOKORO_MODEL_FILE),
                "voices_path": str(PATHS.KOKORO_VOICES_FILE),
                "lang": lang,
                "model_name": voice_name,
            }
            cmd = validate_and_build_command(template, mapping)
            rc, stdout, stderr = run_command(cmd, timeout=600)
            if rc != 0:
                raise RuntimeError(f"Kokoro TTS failed: {stderr}")
        else:
            raise ValueError(f"Unknown TTS engine: {engine}")

        tmp_out.replace(out_path)
        mark_job_as_completed(db, job_id, output_filepath_str=output_path_str)
        logger.info(f"TTS job {job_id} completed.")
    except Exception as e:
        logger.exception(f"TTS job {job_id} failed: {e}")
        update_job_status(db, job_id, "failed", error=str(e))
    finally:
        if "tmp_out" in locals() and tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        try:
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        db.close()
        _update_parent_if_needed(job_id)
        send_webhook_notification(job_id, app_config, base_url)


@huey.task()
def run_pdf_ocr_task(
    job_id: str, input_path_str: str, output_path_str: str, ocr_settings: dict, app_config: dict, base_url: str
):
    db = SessionLocal()
    input_path = Path(input_path_str)
    try:
        job = get_job(db, job_id)
        if not job or job.status == "cancelled":
            return
        update_job_status(db, job_id, "processing")
        logger.info(f"Starting PDF OCR for job {job_id}")

        # Try ocrmypdf first
        try:
            import ocrmypdf
            ocrmypdf.ocr(
                str(input_path),
                str(output_path_str),
                deskew=ocr_settings.get("deskew", True),
                force_ocr=ocr_settings.get("force_ocr", True),
                clean=ocr_settings.get("clean", True),
                optimize=ocr_settings.get("optimize", 1),
                progress_bar=False,
            )
        except Exception:
            logger.warning("ocrmypdf failed, falling back to pytesseract + pymupdf")
            import pytesseract
            from PIL import Image
            import fitz

            doc = fitz.open(str(input_path))
            out_doc = fitz.open()
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, extension="pdf")
                out_doc.insert_pdf(fitz.open(stream=pdf_bytes, filetype="pdf"))
            out_doc.save(str(output_path_str))
            out_doc.close()
            doc.close()

        # Extract preview text
        try:
            import fitz

            doc = fitz.open(str(output_path_str))
            preview = "\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception:
            preview = ""

        mark_job_as_completed(db, job_id, output_filepath_str=output_path_str, preview=preview)
        logger.info(f"PDF OCR job {job_id} completed.")
    except Exception as e:
        logger.exception(f"PDF OCR job {job_id} failed: {e}")
        update_job_status(db, job_id, "failed", error=f"PDF OCR failed: {e}")
    finally:
        try:
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        db.close()
        _update_parent_if_needed(job_id)
        send_webhook_notification(job_id, app_config, base_url)


@huey.task()
def run_image_ocr_task(job_id: str, input_path_str: str, output_path_str: str, app_config: dict, base_url: str):
    db = SessionLocal()
    input_path = Path(input_path_str)
    out_path = Path(output_path_str)

    try:
        job = get_job(db, job_id)
        if not job or job.status == "cancelled":
            return
        update_job_status(db, job_id, "processing", progress=10)
        logger.info(f"Starting Image OCR for job {job_id}")

        try:
            pil_img = Image.open(str(input_path))
        except UnidentifiedImageError as e:
            raise RuntimeError(f"Cannot identify input image: {e}")

        frames = []
        try:
            n_frames = getattr(pil_img, "n_frames", 1)
            for i in range(n_frames):
                pil_img.seek(i)
                frames.append(pil_img.convert("RGB").copy())
        except Exception:
            frames = [pil_img.convert("RGB")]

        import pytesseract

        pdf_bytes_list = []
        text_parts = []
        for idx, frame in enumerate(frames):
            try:
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(frame, extension="pdf")
            except Exception as e:
                raise RuntimeError(f"Tesseract failed on frame {idx}: {e}")
            pdf_bytes_list.append(pdf_bytes)
            try:
                text_parts.append(pytesseract.image_to_string(frame))
            except Exception:
                text_parts.append("")
            prog = 30 + int((idx + 1) / max(1, len(frames)) * 50)
            update_job_status(db, job_id, "processing", progress=min(prog, 80))

        # Merge PDFs with pymupdf
        if len(pdf_bytes_list) == 1:
            final_pdf_bytes = pdf_bytes_list[0]
        else:
            try:
                import fitz

                out_doc = fitz.open()
                for b in pdf_bytes_list:
                    out_doc.insert_pdf(fitz.open(stream=b, filetype="pdf"))
                buf = io.BytesIO()
                out_doc.save(buf)
                out_doc.close()
                final_pdf_bytes = buf.getvalue()
            except Exception:
                logger.warning("PDF merge failed, using first frame only")
                final_pdf_bytes = pdf_bytes_list[0]

        tmp_out = out_path.with_name(f"{out_path.stem}.tmp-{uuid.uuid4().hex}{out_path.suffix or '.pdf'}")
        with tmp_out.open("wb") as f:
            f.write(final_pdf_bytes)
        tmp_out.replace(out_path)

        preview = "\n\n".join(text_parts).strip()[:1000]
        mark_job_as_completed(db, job_id, output_filepath_str=str(out_path), preview=preview)
        logger.info(f"Image OCR job {job_id} completed.")
    except Exception as e:
        logger.exception(f"Image OCR job {job_id} failed: {e}")
        update_job_status(db, job_id, "failed", error=f"Image OCR failed: {e}")
    finally:
        if "tmp_out" in locals() and tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        try:
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        db.close()
        _update_parent_if_needed(job_id)
        send_webhook_notification(job_id, app_config, base_url)


@huey.task()
def run_conversion_task(
    job_id: str,
    input_path_str: str,
    output_path_str: str,
    tool: str,
    task_key: str,
    conversion_tools_config: dict,
    app_config: dict,
    base_url: str,
):
    db = SessionLocal()
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)
    temp_input_file: Optional[Path] = None
    temp_output_file: Optional[Path] = None

    POLL_INTERVAL = 1.0
    STDERR_LIMIT = 4000

    def _parse_task_key(tool_name: str, tk: str, tool_cfg: dict, mapping: dict):
        try:
            if tool_name.startswith("ghostscript"):
                parts = tk.split("_", 1)
                mapping.update({"device": parts[0], "dpi": parts[1] if len(parts) > 1 else "", "preset": parts[1] if len(parts) > 1 else ""})
            elif tool_name == "pngquant":
                parts = tk.split("_", 1)
                qk = parts[1] if len(parts) > 1 else "mq"
                qmap = {"hq": "80-95", "mq": "65-80", "fast": "65-80"}
                smap = {"hq": "1", "mq": "3", "fast": "11"}
                mapping.update({"quality": qmap.get(qk, "65-80"), "speed": smap.get(qk, "3")})
            elif tool_name == "sox":
                parts = tk.split("_")
                rate_token = parts[-2] if len(parts) >= 3 else (parts[-1] if len(parts) == 2 else "")
                depth_token = parts[-1] if len(parts) >= 3 else ("" if len(parts) == 2 else "")
                rate_val = rate_token.replace("k", "000") if rate_token else ""
                depth_val = "-b" + depth_token.replace("b", "") if "b" in depth_token else depth_token
                mapping.update({"samplerate": rate_val, "bitdepth": depth_val})
            elif tool_name == "mozjpeg":
                parts = tk.split("_", 1)
                quality = parts[1].replace("q", "") if len(parts) > 1 else ""
                mapping.update({"quality": quality})
            elif tool_name == "libreoffice":
                target_ext = mapping["output_ext"]
                mapping["filter"] = tool_cfg.get("filters", {}).get(target_ext, target_ext)
            elif tool_name == "pandoc":
                target_ext = mapping["output_ext"]
                mapping["output_ext"] = tool_cfg.get("filters", {}).get(target_ext, target_ext)
        except Exception:
            logger.exception("Failed to parse task_key for tool %s", tool_name)

    def _run_cancellable_command(command: List[str], timeout: int):
        from app.config import _limit_resources_preexec

        if not command:
            raise Exception("Empty command")
        binary = shutil.which(command[0])
        if binary is None:
            raise Exception(f"Required tool '{command[0]}' is not installed or not in PATH. "
                            f"Please install it on your system (Docker images include all tools).")

        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=_limit_resources_preexec,
        )
        start = time.monotonic()
        try:
            while True:
                ret = proc.poll()
                job_check = get_job(db, job_id)
                if job_check is None:
                    proc.kill()
                    raise Exception("Job disappeared during conversion")
                if job_check.status == "cancelled":
                    proc.kill()
                    raise Exception("Conversion cancelled")
                if ret is not None:
                    out, err = proc.communicate(timeout=2)
                    if ret != 0:
                        raise Exception(f"Conversion failed (rc={ret}): {(err or '')[:STDERR_LIMIT]}")
                    return subprocess.CompletedProcess(args=command, returncode=ret, stdout=out, stderr=err)
                elapsed = time.monotonic() - start
                if timeout and elapsed > timeout:
                    proc.kill()
                    raise Exception("Conversion command timed out")
                time.sleep(POLL_INTERVAL)
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass

    try:
        job = get_job(db, job_id)
        if not job or job.status == "cancelled":
            return
        update_job_status(db, job_id, "processing", progress=25)
        logger.info(f"Starting conversion job {job_id} using {tool}/{task_key}")

        tool_config = conversion_tools_config.get(tool)
        if not tool_config:
            raise ValueError(f"Unknown conversion tool: {tool}")

        current_input = input_path

        if tool == "mozjpeg":
            temp_input_file = input_path.with_suffix(".temp.ppm")
            vips_bin = shutil.which("vips") or "vips"
            rc, _, err = run_command([vips_bin, "copy", str(input_path), str(temp_input_file)], timeout=30)
            if rc != 0:
                raise Exception(f"MozJPEG pre-conversion failed: {err}")
            current_input = temp_input_file

        update_job_status(db, job_id, "processing", progress=50)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_output_file = output_path.with_name(f"{output_path.stem}.tmp-{uuid.uuid4().hex}{output_path.suffix}")

        mapping = {
            "input": str(current_input),
            "output": str(temp_output_file),
            "output_dir": str(current_input.parent),
            "output_ext": output_path.suffix.lstrip("."),
        }
        _parse_task_key(tool, task_key, tool_config, mapping)

        template = tool_config.get("command_template")
        if not template:
            raise ValueError(f"Tool '{tool}' missing command_template")
        command = validate_and_build_command(template, mapping)
        command = [str(x) for x in command]

        logger.info("Executing: %s", " ".join(shlex.quote(c) for c in command))
        timeout_val = int(tool_config.get("timeout", 300))
        _run_cancellable_command(command, timeout=timeout_val)

        # Tools like LibreOffice write to output_dir using the input filename,
        # so we need to locate and move the result to our temp_output_file path.
        if tool == "libreoffice":
            expected_lo_output = current_input.with_suffix(output_path.suffix)
            if expected_lo_output.exists():
                shutil.move(str(expected_lo_output), str(temp_output_file))

        # Wait for output file
        for _ in range(60):
            if temp_output_file.exists() and temp_output_file.stat().st_size > 0:
                break
            time.sleep(0.5)
            job_check = get_job(db, job_id)
            if job_check is None or job_check.status == "cancelled":
                raise Exception("Job disappeared or was cancelled")

        if not temp_output_file.exists() or temp_output_file.stat().st_size == 0:
            raise Exception("Conversion produced empty or missing output file")

        temp_output_file.replace(output_path)
        mark_job_as_completed(db, job_id, output_filepath_str=str(output_path), preview="Successfully converted file.")
        logger.info(f"Conversion job {job_id} completed.")
    except Exception as e:
        logger.exception(f"Conversion job {job_id} failed: {e}")
        update_job_status(db, job_id, "failed", error=f"Conversion failed: {e}")
    finally:
        for p in (input_path, temp_input_file, temp_output_file):
            if p:
                try:
                    ensure_path_is_safe(p, [PATHS.UPLOADS_DIR, PATHS.CHUNK_TMP_DIR, PATHS.PROCESSED_DIR])
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        db.close()
        _update_parent_if_needed(job_id)
        send_webhook_notification(job_id, app_config, base_url)
        gc.collect()


@huey.task()
def run_academic_pandoc_task(
    job_id: str, input_path_str: str, output_path_str: str, task_key: str, app_config: dict, base_url: str
):
    db = SessionLocal()
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)
    tmp_extract_dir = None

    try:
        job = get_job(db, job_id)
        if not job or job.status == "cancelled":
            return
        update_job_status(db, job_id, "processing", progress=10)
        logger.info(f"Starting academic Pandoc job {job_id}")

        import zipfile

        tmp_extract_dir = input_path.parent / f"academic_{job_id}"
        tmp_extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(tmp_extract_dir)

        md_files = list(tmp_extract_dir.rglob("*.md"))
        tex_files = list(tmp_extract_dir.rglob("*.tex"))
        bib_files = list(tmp_extract_dir.rglob("*.bib"))
        csl_files = list(tmp_extract_dir.rglob("*.csl"))

        main_doc = None
        if md_files:
            main_doc = md_files[0]
        elif tex_files:
            main_doc = tex_files[0]

        if not main_doc:
            raise ValueError("No .md or .tex main document found in archive.")

        style = task_key.split("_")[-1] if "_" in task_key else "apa"
        csl_url = app_config.get("academic_settings", {}).get("pandoc", {}).get("csl_files", {}).get(style)
        if csl_url and csl_url.startswith("http"):
            import httpx

            csl_path = tmp_extract_dir / f"{style}.csl"
            with httpx.Client() as client:
                r = client.get(csl_url)
                r.raise_for_status()
                csl_path.write_text(r.text)
            csl_files = [csl_path]

        csl_file = str(csl_files[0]) if csl_files else None
        bib_file = str(bib_files[0]) if bib_files else None

        cmd = [
            "pandoc",
            str(main_doc),
            "-o",
            str(output_path),
            "--pdf-engine=xelatex",
        ]
        if bib_file:
            cmd.extend(["--bibliography", bib_file, "--citeproc"])
        if csl_file:
            cmd.extend(["--csl", csl_file])

        rc, _, err = run_command(cmd, timeout=300)
        if rc != 0:
            raise Exception(f"Pandoc failed: {err}")

        mark_job_as_completed(db, job_id, output_filepath_str=str(output_path))
        logger.info(f"Academic Pandoc job {job_id} completed.")
    except Exception as e:
        logger.exception(f"Academic Pandoc job {job_id} failed: {e}")
        update_job_status(db, job_id, "failed", error=f"Academic Pandoc failed: {e}")
    finally:
        if tmp_extract_dir and tmp_extract_dir.exists():
            import shutil

            shutil.rmtree(tmp_extract_dir, ignore_errors=True)
        try:
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        db.close()
        _update_parent_if_needed(job_id)
        send_webhook_notification(job_id, app_config, base_url)


@huey.task()
def unzip_and_dispatch_task(
    job_id: str,
    input_path_str: str,
    sub_task_type: str,
    sub_task_options: dict,
    user: dict,
    app_config: dict,
    base_url: str,
):
    db = SessionLocal()
    input_path = Path(input_path_str)
    extract_dir = None

    try:
        job = get_job(db, job_id)
        if not job or job.status == "cancelled":
            return
        update_job_status(db, job_id, "processing", progress=5)
        logger.info(f"Starting ZIP batch job {job_id}")

        import zipfile

        extract_dir = input_path.parent / f"batch_{job_id}"
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(extract_dir)

        files = [f for f in extract_dir.rglob("*") if f.is_file()]
        if not files:
            raise ValueError("ZIP archive is empty.")

        for idx, file_path in enumerate(files):
            child_id = uuid.uuid4().hex
            dispatch_single_file_job(
                original_filename=file_path.name,
                input_filepath=str(file_path),
                task_type=sub_task_type,
                user=user,
                db=db,
                app_config=app_config,
                base_url=base_url,
                job_id=child_id,
                options=sub_task_options,
                parent_job_id=job_id,
            )
            progress = 10 + int((idx + 1) / len(files) * 80)
            update_job_status(db, job_id, "processing", progress=progress)

        update_job_status(db, job_id, "processing", progress=95)
        logger.info(f"ZIP batch job {job_id} dispatched {len(files)} sub-jobs.")
    except Exception as e:
        logger.exception(f"ZIP batch job {job_id} failed: {e}")
        update_job_status(db, job_id, "failed", error=f"ZIP batch failed: {e}")
    finally:
        if extract_dir and extract_dir.exists():
            import shutil

            shutil.rmtree(extract_dir, ignore_errors=True)
        try:
            ensure_path_is_safe(input_path, [PATHS.UPLOADS_DIR])
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        db.close()
        _update_parent_if_needed(job_id)
        send_webhook_notification(job_id, app_config, base_url)


# --- Helper ---

def _update_parent_if_needed(job_id: str):
    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if job and job.parent_job_id:
            _update_parent_zip_job_progress(job.parent_job_id)
    finally:
        db.close()
