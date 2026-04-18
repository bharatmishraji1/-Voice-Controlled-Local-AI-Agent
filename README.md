# 🎙️ Voice-Controlled Local AI Agent

A fully local, privacy-first voice agent that understands spoken commands and executes them — no cloud required.

## Pipeline

```
Audio Input ➔ Whisper STT ➔ Ollama LLM (intent JSON) ➔ Tool Executor ➔ Streamlit UI
```

## Project Structure

```
voice_ai_agent/
├── app.py               # Streamlit UI — main entry point
├── stt_module.py        # Whisper speech-to-text (mic + file upload)
├── processor_module.py  # Ollama LLM intent classification
├── tools_module.py      # Tool executor (file ops sandboxed to output/)
├── requirements.txt     # Python dependencies
├── output/              # ALL generated files are saved here (auto-created)
└── README.md
```

## Prerequisites

### 1. Python 3.10+
### 2. Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (choose one)
ollama pull llama3
# or
ollama pull mistral

# Start the Ollama server (keep this running in a separate terminal)
ollama serve
```

### 3. Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** If you have a CUDA GPU, install the CUDA build of torch for faster Whisper inference:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

### 4. System Audio (for microphone recording)

- **Linux:** `sudo apt-get install portaudio19-dev`
- **macOS:** `brew install portaudio`
- **Windows:** No extra steps needed.

## Running the App

```bash
# From the project directory
streamlit run app.py
```

The UI will open at `http://localhost:8501`.

## Supported Voice Commands

| Example Command | Intent |
|---|---|
| "Create a file called meeting_notes.txt" | `create_file` |
| "Write a Python script that prints hello world and save it as hello.py" | `write_code` |
| "Summarize the following: ..." | `summarize` |
| "What is machine learning?" | `chat` |

## Security

All file operations are strictly sandboxed to the `output/` directory. Path traversal attempts (e.g., `../../etc/passwd`) are blocked by `tools_module._safe_output_path()`.

## Architecture

| Module | Responsibility |
|---|---|
| `stt_module.py` | Loads Whisper `base`, transcribes mic or WAV bytes |
| `processor_module.py` | POSTs to Ollama `/api/generate`, parses JSON intent |
| `tools_module.py` | Dispatches to `create_file`, `write_code`, `summarize`, or `chat` handlers |
| `app.py` | Streamlit UI, caches Whisper model, renders results |
