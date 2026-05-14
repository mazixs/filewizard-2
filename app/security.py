import html
import os
from pathlib import Path
from typing import List

from fastapi import HTTPException, Request, status
from werkzeug.utils import secure_filename

from app.config import APP_CONFIG, LOCAL_ONLY_MODE, PATHS, logger

# --- Path Safety ---
def ensure_path_is_safe(p: Path, allowed_bases: List[Path]):
    """Enhanced path safety check with traversal prevention"""
    try:
        resolved_p = p.resolve()
        if not any(resolved_p.is_relative_to(base) for base in allowed_bases):
            raise ValueError(f"Path {resolved_p} is outside of allowed directories.")
        return resolved_p
    except Exception as e:
        logger.error(f"Path safety check failed for {p}: {e}")
        raise ValueError("Invalid or unsafe path specified.")


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and XSS"""
    safe_name = secure_filename(filename or "")
    return html.escape(safe_name)


def sanitize_output(output: str) -> str:
    """Sanitize output to prevent XSS"""
    if not output:
        return ""
    return html.escape(output[:2000])


def validate_file_type(filename: str, allowed_extensions: set) -> bool:
    """Validate file type by extension"""
    if not allowed_extensions:
        return True
    return Path(filename).suffix.lower() in allowed_extensions


def get_file_mime_type(filename: str) -> str:
    import mimetypes

    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def get_supported_output_formats_for_file(filename: str, conversion_tools_config: dict) -> list:
    """Get all supported output formats for a given input file."""
    file_ext = get_file_extension(filename)
    supported_formats = []
    for tool_name, tool_config in conversion_tools_config.items():
        supported_inputs = [ext.lower() for ext in tool_config.get("supported_input", [])]
        if file_ext in supported_inputs:
            for format_key, format_label in tool_config.get("formats", {}).items():
                full_format_key = f"{tool_name}_{format_key}"
                supported_formats.append({
                    "value": full_format_key,
                    "label": f"{tool_config['name']} - {format_label}",
                    "tool": tool_name,
                    "format": format_key,
                })
    return supported_formats


# --- Auth Helpers ---
def get_current_user(request: Request):
    if LOCAL_ONLY_MODE:
        return {"sub": "local_user", "email": "local@user.com", "name": "Local User"}
    return request.session.get("user")


def is_admin(request: Request) -> bool:
    if LOCAL_ONLY_MODE:
        return True
    user = get_current_user(request)
    if not user:
        return False
    admin_users = APP_CONFIG.get("auth_settings", {}).get("admin_users", [])
    return user.get("email") in admin_users


def require_user(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


def require_admin(request: Request):
    if not is_admin(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Administrator privileges required."
        )
    return True
