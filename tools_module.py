"""
tools_module.py
---------------
Executes actions based on the structured intent JSON produced by the LLM.
All file operations are strictly sandboxed to the `output/` directory.

Supported intents:
    - create_file  → Creates a file (or folder) inside output/
    - write_code   → Writes generated code to a file inside output/
    - summarize    → Summarizes provided text (no file written)
    - chat         → Returns the LLM's conversational reply
"""

import logging
import os
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Security: all file I/O is restricted to this directory ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure output directory exists at import time
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Public API ───────────────────────────────────────────────────────────────

def execute_intent(intent_data: dict[str, Any]) -> dict[str, str]:
    """
    Dispatch and execute the appropriate tool based on the classified intent.

    Args:
        intent_data: Parsed intent dict with keys 'intent' and 'parameters'.

    Returns:
        A result dict with keys:
            - 'action'   : Human-readable description of what was done.
            - 'result'   : The output or confirmation message.
            - 'filepath' : Absolute path of any file created (empty string if none).

    Raises:
        ValueError: If the intent is unknown or parameters are missing.
    """
    intent = intent_data.get("intent", "chat")
    params = intent_data.get("parameters", {})

    dispatch = {
        "create_file": _handle_create_file,
        "write_code":  _handle_write_code,
        "summarize":   _handle_summarize,
        "chat":        _handle_chat,
    }

    handler = dispatch.get(intent)
    if handler is None:
        logger.warning(f"Unknown intent '{intent}', falling back to chat.")
        handler = _handle_chat

    return handler(params)


# ── Intent handlers ──────────────────────────────────────────────────────────

def _handle_create_file(params: dict[str, str]) -> dict[str, str]:
    """
    Create a new file (or empty folder) inside the output/ directory.

    Args:
        params: Must contain 'filename'. 'content' is optional.

    Returns:
        Result dict describing the created file.

    Raises:
        ValueError: If 'filename' is not provided.
    """
    filename = params.get("filename", "").strip()
    content = params.get("content", "")

    if not filename:
        raise ValueError("'filename' parameter is required for create_file intent.")

    safe_path = _safe_output_path(filename)

    # If the name has no extension, treat it as a directory request
    if not Path(filename).suffix:
        safe_path.mkdir(parents=True, exist_ok=True)
        action = f"Created folder: output/{filename}"
        result = f"Folder '{filename}' created successfully in /output."
        logger.info(action)
        return {"action": action, "result": result, "filepath": str(safe_path)}

    # Otherwise create the file
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    safe_path.write_text(content, encoding="utf-8")
    action = f"Created file: output/{filename}"
    result = f"File '{filename}' created successfully in /output."
    logger.info(action)
    return {"action": action, "result": result, "filepath": str(safe_path)}


def _handle_write_code(params: dict[str, str]) -> dict[str, str]:
    """
    Write generated code to a file inside the output/ directory.

    Args:
        params: Must contain 'filename' and 'content'.

    Returns:
        Result dict describing the saved code file.

    Raises:
        ValueError: If 'filename' or 'content' is missing.
    """
    filename = params.get("filename", "").strip()
    content = params.get("content", "").strip()

    if not filename:
        raise ValueError("'filename' parameter is required for write_code intent.")
    if not content:
        raise ValueError("'content' parameter (the code) is required for write_code intent.")

    safe_path = _safe_output_path(filename)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    safe_path.write_text(content, encoding="utf-8")

    line_count = len(content.splitlines())
    action = f"Saved script to: output/{filename}"
    result = (
        f"Code written to '{filename}' in /output.\n"
        f"Lines of code: {line_count}\n\n"
        f"```\n{content}\n```"
    )
    logger.info(action)
    return {"action": action, "result": result, "filepath": str(safe_path)}


def _handle_summarize(params: dict[str, str]) -> dict[str, str]:
    """
    Return a summary produced by the LLM.
    Optionally saves the summary to output/ if a filename is provided.

    Args:
        params: 'content' holds the summary text. 'filename' is optional.

    Returns:
        Result dict with the summary.
    """
    summary = params.get("content", "").strip()
    filename = params.get("filename", "").strip()
    filepath = ""

    if not summary:
        summary = "No summary content was returned by the LLM."

    if filename:
        safe_path = _safe_output_path(filename)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(summary, encoding="utf-8")
        filepath = str(safe_path)
        action = f"Summarized text and saved to: output/{filename}"
    else:
        action = "Summarized text (not saved to file)"

    logger.info(action)
    return {"action": action, "result": summary, "filepath": filepath}


def _handle_chat(params: dict[str, str]) -> dict[str, str]:
    """
    Return the LLM's conversational response without any file I/O.

    Args:
        params: 'content' holds the chat reply from the LLM.

    Returns:
        Result dict with the chat response.
    """
    reply = params.get("content", "").strip()
    if not reply:
        reply = "I understood your message, but I have no specific response."

    action = "Responded to general query (no file operation)"
    logger.info(action)
    return {"action": action, "result": reply, "filepath": ""}


# ── Security helper ──────────────────────────────────────────────────────────

def _safe_output_path(filename: str) -> Path:
    """
    Resolve a filename to a path strictly inside OUTPUT_DIR.

    Args:
        filename: Relative filename (may include subdirectories).

    Returns:
        Resolved absolute Path within OUTPUT_DIR.

    Raises:
        PermissionError: If the resolved path escapes OUTPUT_DIR (path traversal attempt).
    """
    # Strip any absolute path prefix from the filename for safety
    sanitized = Path(filename.lstrip("/\\"))
    resolved = (OUTPUT_DIR / sanitized).resolve()

    # Ensure the final path is still inside OUTPUT_DIR
    if not str(resolved).startswith(str(OUTPUT_DIR.resolve())):
        raise PermissionError(
            f"Security violation: '{filename}' resolves outside the output/ sandbox."
        )

    return resolved


def list_output_files() -> list[str]:
    """
    Return a sorted list of all files currently in the output/ directory.

    Returns:
        List of relative file paths (relative to output/).
    """
    files = []
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file():
            files.append(str(path.relative_to(OUTPUT_DIR)))
    return files
