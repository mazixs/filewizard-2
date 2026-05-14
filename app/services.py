import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import APP_CONFIG, PATHS, _limit_resources_preexec, logger
from app.database import Job, JobCreate, JobSchema, SessionLocal
from app.security import ensure_path_is_safe, sanitize_filename, validate_file_type

# --- CRUD Operations ---

def get_job(db: Session, job_id: str) -> Job | None:
    return db.query(Job).filter(Job.id == job_id).first()


def get_jobs(db: Session, user_id: str | None = None, skip: int = 0, limit: int = 100):
    query = db.query(Job)
    if user_id:
        query = query.filter(Job.user_id == user_id)
    return query.order_by(Job.created_at.desc()).offset(skip).limit(limit).all()


def create_job(db: Session, job: JobCreate) -> Job:
    db_job = Job(**job.model_dump())
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


def update_job_status(
    db: Session, job_id: str, status: str, progress: int | None = None, error: str | None = None
):
    db_job = get_job(db, job_id)
    if db_job:
        db_job.status = status
        if progress is not None:
            db_job.progress = progress
        if error:
            db_job.error_message = error
        db.commit()
    return db_job


def mark_job_as_completed(
    db: Session,
    job_id: str,
    output_filepath_str: str | None = None,
    preview: str | None = None,
):
    db_job = get_job(db, job_id)
    if db_job and db_job.status != "cancelled":
        if preview:
            db_job.result_preview = preview.strip()[:2000]
        if output_filepath_str:
            try:
                output_path = Path(output_filepath_str)
                if output_path.exists():
                    db_job.output_filesize = output_path.stat().st_size
            except Exception:
                logger.exception(f"Could not stat output file {output_filepath_str} for job {job_id}")
        update_job_status(db, job_id, "completed", progress=100)
    return db_job


# --- Webhooks ---

def send_webhook_notification(job_id: str, app_config: Dict[str, Any], base_url: str):
    webhook_config = app_config.get("webhook_settings", {})
    if not webhook_config.get("enabled", False):
        return

    db = SessionLocal()
    try:
        job = get_job(db, job_id)
        if not job or not job.callback_url:
            return

        download_url = None
        if job.status == "completed" and job.processed_filepath:
            filename = Path(job.processed_filepath).name
            public_url = app_config.get("app_settings", {}).get("app_public_url", base_url)
            if not public_url:
                download_url = f"/download/{filename}"
            else:
                download_url = urljoin(public_url, f"/download/{filename}")

        payload = {
            "job_id": job.id,
            "status": job.status,
            "original_filename": job.original_filename,
            "download_url": download_url,
            "error_message": job.error_message,
            "created_at": job.created_at.isoformat() + "Z",
            "updated_at": job.updated_at.isoformat() + "Z",
        }

        headers = {"Content-Type": "application/json", "User-Agent": "FileWizard-Webhook/1.0"}
        token = webhook_config.get("callback_bearer_token")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            with httpx.Client() as client:
                response = client.post(job.callback_url, json=payload, headers=headers, timeout=15)
                response.raise_for_status()
            logger.info(f"Webhook sent for job {job_id} to {job.callback_url} ({response.status_code})")
        except httpx.RequestError as e:
            logger.error(f"Webhook request failed for job {job_id}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Webhook non-2xx for job {job_id}: {e.response.status_code}")
    except Exception as e:
        logger.exception(f"Unexpected error in webhook for job {job_id}: {e}")
    finally:
        db.close()


# --- File Operations ---

async def save_upload_file(upload_file: UploadFile, destination: Path) -> int:
    max_size = APP_CONFIG.get("app_settings", {}).get("max_file_size_bytes", 100 * 1024 * 1024)
    allowed_extensions = APP_CONFIG.get("app_settings", {}).get("allowed_all_extensions", set())
    if not validate_file_type(upload_file.filename, allowed_extensions):
        raise HTTPException(
            status_code=400, detail=f"File type '{Path(upload_file.filename).suffix}' not allowed."
        )

    tmp_path = destination.with_name(f"{destination.stem}.tmp-{uuid.uuid4().hex}{destination.suffix}")
    size = 0
    try:
        with tmp_path.open("wb") as buffer:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_size:
                    raise HTTPException(
                        status_code=413, detail=f"File exceeds {max_size / 1024 / 1024} MB limit"
                    )
                buffer.write(chunk)
        tmp_path.replace(destination)
        return size
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error saving upload file: {e}")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    finally:
        try:
            await upload_file.close()
        except Exception:
            pass


# --- Command Execution ---

def run_command(
    cmd: List[str],
    timeout: int = 300,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    preexec_fn=None,
) -> tuple[int, str, str]:
    """Run a subprocess command and return (returncode, stdout, stderr)."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=merged_env,
            preexec_fn=preexec_fn or _limit_resources_preexec,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout.decode("utf-8", errors="replace"), stderr.decode(
            "utf-8", errors="replace"
        )
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return -1, stdout.decode("utf-8", errors="replace"), stderr.decode(
            "utf-8", errors="replace"
        )
    except Exception as e:
        return -1, "", str(e)


# --- TTS Helpers ---

def _find_model_files(model_name: str, model_dir: Path):
    """Finds ONNX and JSON config files for a Piper voice."""
    safe_name = sanitize_filename(model_name)
    candidates = [
        model_dir / safe_name,
        model_dir / f"{safe_name}.onnx",
    ]
    for base in candidates:
        onnx = base.with_suffix(".onnx")
        config_json = base.with_suffix(".onnx.json")
        if not config_json.exists():
            config_json = Path(str(base) + ".onnx.json")
        if onnx.exists() and config_json.exists():
            return onnx, config_json
    return None, None


def list_voices_cli(timeout: int = 30) -> List[str]:
    """List available Piper voices via CLI if available."""
    python_exes = ["python", "python3", shutil.which("python") or "python"]
    for py in set(filter(None, python_exes)):
        try:
            rc, out, _ = run_command([py, "-m", "piper", "--list-voices"], timeout=timeout)
            if rc == 0:
                return [line.strip() for line in out.splitlines() if line.strip()]
        except Exception:
            continue
    return []


def safe_get_voices(model_dir: Path) -> List[Dict[str, Any]]:
    """Get available Piper voices from local model directory."""
    voices = []
    if not model_dir.exists():
        return voices
    for onnx_file in model_dir.glob("*.onnx"):
        voice_id = onnx_file.stem
        voices.append({"id": voice_id, "name": voice_id, "local": True})
    return voices


def list_kokoro_voices_cli(timeout: int = 60) -> List[str]:
    if not shutil.which("kokoro-tts"):
        return []
    rc, out, _ = run_command(["kokoro-tts", "--list-voices"], timeout=timeout)
    if rc != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip() and not line.startswith("-")]


def list_kokoro_languages_cli(timeout: int = 60) -> List[str]:
    if not shutil.which("kokoro-tts"):
        return []
    rc, out, _ = run_command(["kokoro-tts", "--list-langs"], timeout=timeout)
    if rc != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip() and not line.startswith("-")]


# --- Conversion Helpers ---

def validate_and_build_command(template_str: str, mapping: Dict[str, str]) -> List[str]:
    """Builds a command list from a template string, preventing shell injection."""
    import shlex
    from string import Formatter

    # Extract placeholders from template
    placeholders = {fname for _, fname, _, _ in Formatter().parse(template_str) if fname}
    allowed = {"input", "output", "output_dir", "output_ext", "filter", "device", "dpi", "preset", "quality", "speed", "samplerate", "bitdepth", "model_path", "voices_path", "lang", "model_name", "main_document", "bib_file", "csl_style"}
    for ph in placeholders:
        if ph not in allowed:
            raise ValueError(f"Disallowed placeholder in command template: {{{ph}}}")

    formatted = template_str.format(**mapping)
    return shlex.split(formatted)


def _parse_tool_and_task_key(output_format: str, all_tool_keys: list) -> tuple[str, str]:
    for tool_key in sorted(all_tool_keys, key=len, reverse=True):
        if output_format.startswith(tool_key + "_"):
            task_key = output_format[len(tool_key) + 1 :]
            return tool_key, task_key
    raise ValueError(f"Could not determine tool from output_format: {output_format}")


# --- Parent ZIP Progress ---

def _update_parent_zip_job_progress(parent_job_id: str):
    db = SessionLocal()
    try:
        parent = get_job(db, parent_job_id)
        if not parent or parent.status in ("completed", "failed", "cancelled"):
            return
        children = db.query(Job).filter(Job.parent_job_id == parent_job_id).all()
        if not children:
            return
        total = len(children)
        done = sum(1 for c in children if c.status in ("completed", "failed", "cancelled"))
        progress = int((done / total) * 100)
        update_job_status(db, parent_job_id, "processing", progress=progress)
        if done == total:
            failed = sum(1 for c in children if c.status == "failed")
            if failed == total:
                update_job_status(db, parent_job_id, "failed", error="All sub-jobs failed")
            else:
                mark_job_as_completed(db, parent_job_id)
    except Exception:
        logger.exception(f"Error updating parent ZIP job {parent_job_id}")
    finally:
        db.close()


