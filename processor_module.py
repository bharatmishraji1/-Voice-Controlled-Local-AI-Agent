"""
processor_module.py
-------------------
Sends transcribed text to a local Ollama LLM instance for intent classification.
Forces the LLM to return a structured JSON object describing the user's intent
and associated parameters.
"""

import json
import logging
from typing import Any

import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"
DEFAULT_MODEL = "llama3"   # Swap to "mistral" if preferred
REQUEST_TIMEOUT = 60       # Seconds

# System prompt that enforces strict JSON output from the LLM
SYSTEM_PROMPT = """You are a precise intent-classification engine. Your ONLY job is to analyze the user's voice command and return a single, valid JSON object — nothing else.

STRICT RULES:
1. Return ONLY raw JSON. No markdown, no code blocks, no explanation, no extra text.
2. Always use exactly this schema:
{
  "intent": "<one of: create_file | write_code | summarize | chat>",
  "parameters": {
    "filename": "<target filename or empty string>",
    "content": "<content to write, code to generate, text to summarize, or chat reply>"
  }
}

INTENT DEFINITIONS:
- "create_file"  → User wants to create a new file or folder (e.g., "create a file called notes.txt").
- "write_code"   → User wants code written and saved (e.g., "write a Python hello world script").
- "summarize"    → User wants text summarized (e.g., "summarize this paragraph: ...").
- "chat"         → Any general question or conversation (e.g., "what is machine learning?").

PARAMETER RULES:
- "filename": Required for create_file and write_code. Use snake_case. Include extension. Empty string otherwise.
- "content": The file content, generated code, summary, or chat response as appropriate.

Examples:
User: "create a text file called shopping list"
Output: {"intent": "create_file", "parameters": {"filename": "shopping_list.txt", "content": ""}}

User: "write a python script that prints hello world and save it as hello.py"
Output: {"intent": "write_code", "parameters": {"filename": "hello.py", "content": "print('Hello, World!')"}}

User: "what is the capital of France?"
Output: {"intent": "chat", "parameters": {"filename": "", "content": "The capital of France is Paris."}}

Now classify the following user command and respond ONLY with valid JSON:"""


def check_ollama_connection() -> bool:
    """
    Verify that the Ollama server is reachable.

    Returns:
        True if the server responds, False otherwise.
    """
    try:
        response = requests.get(OLLAMA_BASE_URL, timeout=5)
        return response.status_code == 200
    except requests.ConnectionError:
        return False
    except Exception:
        return False


def classify_intent(
    transcription: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Send a transcription to Ollama and return the parsed intent JSON.

    Args:
        transcription: The user's voice command as plain text.
        model: The Ollama model name to use (e.g., 'llama3', 'mistral').

    Returns:
        A dictionary with keys 'intent' and 'parameters'.

    Raises:
        ConnectionError: If the Ollama server is not reachable.
        ValueError: If the LLM response cannot be parsed as valid JSON.
        RuntimeError: For any other unexpected API errors.
    """
    if not check_ollama_connection():
        raise ConnectionError(
            "Cannot connect to Ollama at localhost:11434. "
            "Ensure Ollama is running: `ollama serve`"
        )

    prompt = f"{SYSTEM_PROMPT}\n\nUser: {transcription}"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # Low temperature for deterministic structured output
            "top_p": 0.9,
        },
    }

    logger.info(f"Sending transcription to Ollama [{model}]: '{transcription}'")

    try:
        response = requests.post(
            OLLAMA_API_ENDPOINT,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise ConnectionError(f"Ollama connection error: {exc}") from exc
    except requests.Timeout:
        raise RuntimeError(f"Ollama request timed out after {REQUEST_TIMEOUT}s.")
    except requests.HTTPError as exc:
        raise RuntimeError(f"Ollama API HTTP error: {exc}") from exc

    raw_text = response.json().get("response", "").strip()
    logger.info(f"Raw LLM response: {raw_text}")

    return _parse_llm_response(raw_text)


def _parse_llm_response(raw_text: str) -> dict[str, Any]:
    """
    Parse and validate the LLM's raw text output as a structured JSON intent.

    Args:
        raw_text: The raw string returned by the LLM.

    Returns:
        Validated intent dictionary.

    Raises:
        ValueError: If the response is not valid JSON or missing required keys.
    """
    # Strip markdown code fences if present (defensive handling)
    clean = raw_text.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        clean = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    # Extract the first JSON object found in the response
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response: '{raw_text}'")

    json_str = clean[start:end]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON: {exc}\nRaw response: '{raw_text}'"
        ) from exc

    # Validate required schema keys
    if "intent" not in parsed:
        raise ValueError(f"'intent' key missing from LLM response: {parsed}")
    if "parameters" not in parsed:
        raise ValueError(f"'parameters' key missing from LLM response: {parsed}")

    valid_intents = {"create_file", "write_code", "summarize", "chat"}
    if parsed["intent"] not in valid_intents:
        logger.warning(
            f"Unknown intent '{parsed['intent']}', defaulting to 'chat'."
        )
        parsed["intent"] = "chat"

    # Ensure parameter keys exist with defaults
    parsed["parameters"].setdefault("filename", "")
    parsed["parameters"].setdefault("content", "")

    logger.info(f"Parsed intent: {parsed}")
    return parsed
