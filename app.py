"""
app.py
------
Streamlit UI for the Voice-Controlled Local AI Agent.

Pipeline:
    Audio Input (mic / WAV upload)
        ↓  stt_module  → Transcription
        ↓  processor_module  → Intent JSON (via Ollama)
        ↓  tools_module  → Tool Execution
        ↓  Streamlit  → Results displayed to user
"""

import time
import logging
from pathlib import Path

import streamlit as st

from stt_module import load_whisper_model, transcribe_bytes, record_from_microphone
from processor_module import classify_intent, check_ollama_connection, DEFAULT_MODEL
from tools_module import execute_intent, list_output_files, OUTPUT_DIR

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice AI Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Session state initialisation ─────────────────────────────────────────────
def _init_session_state() -> None:
    """Initialise all Streamlit session state variables."""
    defaults = {
        "whisper_model": None,
        "transcription": "",
        "intent_data": None,
        "tool_result": None,
        "error_message": "",
        "processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Cached resource: Whisper model ───────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Whisper model (base) — first run only…")
def get_whisper_model():
    """Load and cache the Whisper base model across Streamlit re-runs."""
    return load_whisper_model("base")


# ── Pipeline runner ───────────────────────────────────────────────────────────
def run_pipeline(audio_bytes: bytes | None = None, from_mic: bool = False, mic_duration: int = 10) -> None:
    """
    Execute the full STT → LLM → Tool pipeline and store results in session state.

    Args:
        audio_bytes: Raw WAV bytes from file uploader (None if using mic).
        from_mic: If True, record from microphone instead of using audio_bytes.
        mic_duration: Recording duration in seconds when using microphone.
    """
    st.session_state.error_message = ""
    st.session_state.transcription = ""
    st.session_state.intent_data = None
    st.session_state.tool_result = None

    model = get_whisper_model()

    # ── Step 1: Speech-to-Text ────────────────────────────────────────────────
    with st.spinner("🎙️ Transcribing audio…"):
        try:
            if from_mic:
                transcription = record_from_microphone(duration=mic_duration)
            else:
                transcription = transcribe_bytes(audio_bytes, model=model)
            st.session_state.transcription = transcription
        except Exception as exc:
            st.session_state.error_message = f"STT Error: {exc}"
            logger.error(st.session_state.error_message)
            return

    if not transcription.strip():
        st.session_state.error_message = "No speech detected in the audio. Please try again."
        return

    # ── Step 2: Intent Classification via Ollama ─────────────────────────────
    with st.spinner("Classifying intent with Ollama…"):
        try:
            intent_data = classify_intent(
                transcription,
                model=st.session_state.get("ollama_model", DEFAULT_MODEL),
            )
            st.session_state.intent_data = intent_data
        except ConnectionError as exc:
            st.session_state.error_message = str(exc)
            logger.error(st.session_state.error_message)
            return
        except ValueError as exc:
            st.session_state.error_message = f"LLM Parsing Error: {exc}"
            logger.error(st.session_state.error_message)
            return
        except Exception as exc:
            st.session_state.error_message = f"Unexpected LLM Error: {exc}"
            logger.error(st.session_state.error_message)
            return

    # ── Step 3: Tool Execution ────────────────────────────────────────────────
    with st.spinner(" Executing action…"):
        try:
            tool_result = execute_intent(intent_data)
            st.session_state.tool_result = tool_result
        except PermissionError as exc:
            st.session_state.error_message = f"Security Error: {exc}"
            logger.error(st.session_state.error_message)
        except ValueError as exc:
            st.session_state.error_message = f"Tool Error: {exc}"
            logger.error(st.session_state.error_message)
        except Exception as exc:
            st.session_state.error_message = f"Unexpected Tool Error: {exc}"
            logger.error(st.session_state.error_message)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    """Render the sidebar with configuration and status indicators."""
    with st.sidebar:
        st.title("⚙️ Configuration")

        # Ollama status
        st.subheader("Ollama Status")
        if check_ollama_connection():
            st.success(" Ollama is running", icon="✅")
        else:
            st.error(" Ollama not reachable on localhost:11434", icon="🚫")
            st.info("Start Ollama with: `ollama serve`")

        st.divider()

        # Model selection
        st.subheader("LLM Model")
        model_choice = st.selectbox(
            "Select Ollama model:",
            options=["llama3", "mistral", "llama2", "gemma"],
            index=0,
            help="Ensure the selected model is pulled: `ollama pull <model>`",
        )
        st.session_state["ollama_model"] = model_choice

        st.divider()

        # Output directory browser
        st.subheader("📁 Output Directory")
        output_files = list_output_files()
        if output_files:
            for f in output_files:
                file_path = OUTPUT_DIR / f
                with open(file_path, "rb") as fh:
                    st.download_button(
                        label=f"⬇ {f}",
                        data=fh,
                        file_name=f,
                        key=f"dl_{f}_{time.time()}",
                    )
        else:
            st.caption("No files generated yet.")

        st.divider()
        st.caption("Voice AI Agent v1.0 | Powered by Whisper + Ollama")


# ── Main UI ───────────────────────────────────────────────────────────────────
def render_main() -> None:
    """Render the main Streamlit interface."""
    st.title("🎙️ Voice-Controlled Local AI Agent")
    st.markdown(
        "Speak a command — the agent will transcribe it, classify the intent, "
        "and execute the appropriate action locally."
    )

    # ── Input section ─────────────────────────────────────────────────────────
    st.subheader("1. Provide Audio Input")
    tab_mic, tab_upload = st.tabs(["Record from Microphone", "Upload WAV File"])

    with tab_mic:
        st.info(
            "Recording uses your system microphone via `sounddevice`. "
            "Ensure microphone permissions are granted.",
            icon="ℹ️",
        )
        mic_duration = st.slider(
            "Recording duration (seconds):",
            min_value=3,
            max_value=30,
            value=10,
            step=1,
        )
        if st.button("Start Recording", type="primary", use_container_width=True):
            run_pipeline(from_mic=True, mic_duration=mic_duration)
            st.rerun()

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a WAV audio file:",
            type=["wav"],
            help="Only .wav files are supported.",
        )
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            if st.button(" Transcribe & Process", type="primary", use_container_width=True):
                run_pipeline(audio_bytes=uploaded_file.read())
                st.rerun()

    # ── Results section ───────────────────────────────────────────────────────
    if st.session_state.error_message:
        st.error(st.session_state.error_message, icon="🚨")

    if st.session_state.transcription or st.session_state.intent_data or st.session_state.tool_result:
        st.divider()
        st.subheader("2. Pipeline Results")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            # Transcription card
            with st.container(border=True):
                st.markdown("####  Transcribed Text")
                if st.session_state.transcription:
                    st.write(f'"{st.session_state.transcription}"')
                else:
                    st.caption("—")

            # Intent card
            with st.container(border=True):
                st.markdown("####  Detected Intent")
                if st.session_state.intent_data:
                    intent = st.session_state.intent_data.get("intent", "—")
                    params = st.session_state.intent_data.get("parameters", {})

                    intent_emoji = {
                        "create_file": "📄",
                        "write_code": "💻",
                        "summarize": "📋",
                        "chat": "💬",
                    }.get(intent, "❓")

                    st.metric(label="Intent", value=f"{intent_emoji} {intent}")

                    if params.get("filename"):
                        st.markdown(f"**Filename:** `{params['filename']}`")

                    with st.expander("Raw JSON response"):
                        st.json(st.session_state.intent_data)
                else:
                    st.caption("—")

        with col2:
            # Action taken card
            with st.container(border=True):
                st.markdown("#### ⚙️ Action Taken")
                if st.session_state.tool_result:
                    action = st.session_state.tool_result.get("action", "—")
                    filepath = st.session_state.tool_result.get("filepath", "")
                    st.success(action, icon="✅")
                    if filepath:
                        st.markdown(f"**Saved to:** `{filepath}`")
                else:
                    st.caption("—")

            # Final result card
            with st.container(border=True):
                st.markdown("#### Final Response / Result")
                if st.session_state.tool_result:
                    result_text = st.session_state.tool_result.get("result", "")
                    if "```" in result_text:
                        # Render code blocks natively
                        st.markdown(result_text)
                    else:
                        st.write(result_text)
                else:
                    st.caption("—")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    with st.expander(" Supported Voice Commands"):
        st.markdown("""
| Example Command | Intent |
|---|---|
| *"Create a file called meeting_notes.txt"* | `create_file` |
| *"Write a Python script that prints hello and save it as hello.py"* | `write_code` |
| *"Summarize the following text: ..."* | `summarize` |
| *"What is machine learning?"* | `chat` |
""")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    """Main entry point for the Streamlit application."""
    _init_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
