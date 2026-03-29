"""
Streamlit Web UI for Context-1 Data Generation Pipeline
Provides a graphical interface to run the pipeline using free LLM providers.
"""
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Any, Dict

import streamlit as st

from config import (
    DOMAIN_OPTIONS,
    MODEL_OPTIONS,
    PipelineConfig,
    build_command,
    validate_api_keys_from_env,
)


REPO_URL = "https://github.com/developerjeremylive/context-1-data-gen-etheroi.git"
REPO_NAME = "context-1-data-gen-etheroi"


def get_repo_path() -> Path:
    return Path(tempfile.gettempdir()) / REPO_NAME


def load_env_file(repo_path: Path) -> dict:
    env_path = repo_path / ".env"
    env_vars = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    else:
        env_vars["__ERROR__"] = f".env not found at {env_path}"
    return env_vars


def clone_repo_if_needed(token: str) -> tuple[bool, str, Path]:
    repo_path = get_repo_path()
    if repo_path.exists():
        return True, f"Repository already exists at {repo_path}", repo_path
    try:
        clone_url = f"https://x-access-token:{token}@github.com/developerjeremylive/context-1-data-gen-etheroi.git"
        subprocess.run(
            ["git", "clone", clone_url, str(repo_path)],
            check=True, capture_output=True, text=True,
        )
        return True, f"Successfully cloned to {repo_path}", repo_path
    except subprocess.CalledProcessError as e:
        return False, f"Failed to clone: {e.stderr}", repo_path


def find_venv_python(repo_path: Path) -> tuple[str, list[str]]:
    for venv_name in [".venv", "venv", ".venv37"]:
        venv_path = repo_path / venv_name
        if venv_path.exists():
            for exe_name in ["python.exe", "python3.exe", "python"]:
                python_exe = venv_path / ("Scripts" if os.name == "nt" else "bin") / exe_name
                if python_exe.exists():
                    sp_result = subprocess.run(
                        [str(python_exe), "-c", "import site; print('|'.join(site.getsitepackages()))"],
                        capture_output=True, text=True,
                    )
                    if sp_result.returncode == 0:
                        return str(python_exe), sp_result.stdout.strip().split("|")
    current_python = sys.executable
    venv_sp = repo_path / ".venv" / ("lib" if os.name != "nt" else "Lib") / "site-packages"
    if venv_sp.exists():
        return current_python, [str(venv_sp)]
    return current_python, []


def install_project_deps(repo_path: Path) -> tuple[bool, str, str, list[str]]:
    result = subprocess.run(
        ["uv", "sync", "--all-extras"],
        cwd=str(repo_path), capture_output=True, text=True, timeout=180,
    )
    if result.returncode == 0:
        python_exe, sp_paths = find_venv_python(repo_path)
        return True, "Dependencies installed via uv sync", python_exe, sp_paths
    python_exe, sp_paths = find_venv_python(repo_path)
    result = subprocess.run(
        [python_exe, "-m", "pip", "install", "-e", ".[all]"],
        cwd=str(repo_path), capture_output=True, text=True, timeout=300,
    )
    if result.returncode == 0:
        return True, "Dependencies installed via pip", python_exe, sp_paths
    return False, f"Failed to install deps: {result.stderr[-500:]}", python_exe, sp_paths


# ─── OpenAI-compatible client that works with Anthropic-style calls ───────────
class OpenAIMessagesClient:
    """
    A client wrapper that provides an `.messages.create()` interface
    but calls the OpenAI-compatible endpoint (Pollinations AI).

    The pipeline code expects Anthropic's `client.messages.create(...)` interface.
    This class translates those calls to OpenAI `chat.completions.create(...)` calls.
    """

    def __init__(self, base_url: str = "https://gen.pollinations.ai/v1",
                 api_key: str = "not-needed", model: str = "openai/gpt-oss-20b"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                # Fallback: use requests directly
                self._client = None
        return self._client

    def messages_create(self, model: str, system: str, max_tokens: int,
                        messages: List[Dict], tools: List[Dict],
                        tool_choice: Dict, thinking: Dict = None) -> Any:
        """
        Translate Anthropic-style `client.messages.create(...)` to
        OpenAI `client.chat.completions.create(...)`.

        Returns an object with `.content` (list of content blocks) compatible
        with what the pipeline expects from Anthropic.
        """
        # Build OpenAI messages
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Handle tool results ( Anthropic format with tool_result content blocks )
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_result":
                            text_parts.append(f"[TOOL RESULT]: {block.get('content', '')}")
                        elif block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "input":
                            text_parts.append(str(block))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)
            openai_messages.append({"role": role, "content": content})

        # Try OpenAI SDK first
        client = self._get_client()
        if client is not None:
            try:
                kwargs = {
                    "model": model,
                    "messages": openai_messages,
                    "max_tokens": max_tokens,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice.get("type", "auto")
                
                response = client.chat.completions.create(**kwargs)

                # Translate OpenAI response to Anthropic-like format
                return _OpenAIResponse(response)
            except Exception as e:
                # Fallback: use requests directly
                pass

        # Direct HTTP fallback
        return self._messages_create_http(model, system, max_tokens, openai_messages, tools, tool_choice)

    def _messages_create_http(self, model: str, system: str, max_tokens: int,
                               messages: list, tools: list, tool_choice: dict) -> Any:
        """Use requests directly as fallback."""
        import json, urllib.request

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice.get("type", "auto")

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return _OpenAIResponse(data)
        except Exception as e:
            raise RuntimeError(f"Pollination AI API call failed: {e}")


class _OpenAIResponse:
    """Wraps OpenAI response to look like Anthropic response for the pipeline."""

    def __init__(self, response):
        if isinstance(response, dict):
            self._data = response
            choice = response.get("choices", [{}])[0]
            self._message = choice.get("message", {})
            self.content = [_TextBlock(self._message.get("content", ""))]
            tool_calls = self._message.get("tool_calls", [])
            if tool_calls:
                # Convert tool_calls to tool_use blocks
                for tc in tool_calls:
                    self.content.append(_ToolUseBlock(tc))
        else:
            # OpenAI SDK response object
            self._data = {}
            self._message = response
            content = getattr(response, "choices", [None])[0].message if hasattr(response, "choices") else None
            text = getattr(content, "content", "") or ""
            self.content = [_TextBlock(text)]
            tool_calls = getattr(content, "tool_calls", []) or []
            for tc in tool_calls:
                self.content.append(_ToolUseBlock(tc))


class _TextBlock:
    type = "text"
    def __init__(self, text):
        self.text = text
        self.thinking = ""


class _ToolUseBlock:
    type = "tool_use"
    def __init__(self, tc):
        self.id = tc.get("id", "")
        self.name = tc.get("function", {}).get("name", "")
        self.input = tc.get("function", {}).get("arguments", {})
        if isinstance(self.input, str):
            try:
                import json
                self.input = json.loads(self.input)
            except Exception:
                pass


# ─── Patching functions ────────────────────────────────────────────────────────
def patch_pipeline_for_pollination(repo_path: Path) -> tuple[bool, str]:
    """
    Patch pipeline modules to use the OpenAIMessagesClient wrapper.
    This translates Anthropic SDK calls → OpenAI-compatible calls for Pollinations AI.
    """
    patches = []

    # ─── Patch core/utils.py — add get_pollination_client ───────────────────
    utils_file = repo_path / "agentic_search_data_gen" / "core" / "utils.py"
    if utils_file.exists():
        content = utils_file.read_text(encoding="utf-8")
        if "def get_pollination_client():" not in content:
            # Import the client class definition into the patched file
            client_code = '''
def get_pollination_client():
    """Get an OpenAI-compatible client that works with Pollinations AI.

    Provides `.messages.create(...)` interface (Anthropic-style) but calls
    OpenAI-compatible endpoint, translating calls automatically.
    """
    import os
    base_url = os.getenv("OPENAI_API_BASE", "https://gen.pollinations.ai/v1")
    api_key = os.getenv("OPENAI_API_KEY", "not-needed")
    model = os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b")
    from .client_wrapper import OpenAIMessagesClient
    return OpenAIMessagesClient(base_url=base_url, api_key=api_key, model=model)
'''
            if "def get_anthropic_client():" in content:
                idx = content.find("def get_anthropic_client():")
                content = content[:idx] + client_code + content[idx:]
            elif "def count_tokens" in content:
                idx = content.find("def count_tokens")
                content = content[:idx] + client_code + content[idx:]
            utils_file.write_text(content, encoding="utf-8")
            patches.append("  ✓ Patched core/utils.py (added get_pollination_client)")

    # ─── Patch __main__.py ──────────────────────────────────────────────────
    main_file = repo_path / "agentic_search_data_gen" / "domains" / "web" / "__main__.py"
    if main_file.exists():
        content = main_file.read_text(encoding="utf-8")
        # Add import for the client wrapper
        if "from .client_wrapper import" not in content:
            # Add at top of imports
            content = content.replace(
                "from ...core.utils import get_anthropic_client",
                "from ...core.utils import get_anthropic_client, get_pollination_client"
            )
        # Patch client = get_anthropic_client() → use pollination client
        patched = content.replace(
            "client = get_anthropic_client()",
            "baseten_key = os.getenv('BASETEN_API_KEY', '')\n    if baseten_key:\n        from openai import OpenAI\n        client = OpenAI(api_key=baseten_key, base_url='https://app.baseten.co')\n    else:\n        client = get_pollination_client()"
        )
        if patched != content:
            main_file.write_text(patched, encoding="utf-8")
            patches.append("  ✓ Patched __main__.py")

    # ─── Patch explore.py ────────────────────────────────────────────────────
    explore_file = repo_path / "agentic_search_data_gen" / "domains" / "web" / "explore.py"
    if explore_file.exists():
        content = explore_file.read_text(encoding="utf-8")
        patched = content.replace(
            "    def __init__(self, model: str = \"claude-sonnet-4-5\", max_iterations: int = 20):\n        client = get_anthropic_client()\n        super().__init__(client, model, max_iterations)",
            "    def __init__(self, model: str = \"claude-sonnet-4-5\", max_iterations: int = 20):\n        baseten_key = os.getenv('BASETEN_API_KEY', '')\n        if baseten_key:\n            from openai import OpenAI\n            client = OpenAI(api_key=baseten_key, base_url='https://app.baseten.co')\n        else:\n            from .client_wrapper import get_pollination_client\n            client = get_pollination_client()\n        super().__init__(client, model, max_iterations)"
        )
        if patched != content:
            explore_file.write_text(patched, encoding="utf-8")
            patches.append("  ✓ Patched explore.py")

    # ─── Create the client wrapper module ───────────────────────────────────
    wrapper_file = repo_path / "agentic_search_data_gen" / "domains" / "web" / "client_wrapper.py"
    wrapper_content = '''
"""OpenAI-compatible client wrapper for Pollinations AI.

Provides `.messages.create(...)` interface (Anthropic-style) but calls
OpenAI-compatible endpoint, translating calls automatically.

This allows the pipeline to work with Pollinations AI without modifying
the core agent code that uses Anthropic SDK patterns.
"""

import json
import urllib.request
from typing import List, Any, Dict


class OpenAIMessagesClient:
    """
    A client wrapper that provides an `.messages.create()` interface
    but calls the OpenAI-compatible endpoint (Pollinations AI).

    Translates Anthropic-style `client.messages.create(...)` calls to
    OpenAI `client.chat.completions.create(...)` calls.
    """

    def __init__(self, base_url: str = "https://gen.pollinations.ai/v1",
                 api_key: str = "not-needed", model: str = "openai/gpt-oss-20b"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_openai_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                self._client = None
        return self._client

    def messages_create(self, model: str, system: str, max_tokens: int,
                        messages: List[Dict], tools: List[Dict] = None,
                        tool_choice: Dict = None, thinking: Dict = None) -> Any:
        """
        Translate Anthropic-style `client.messages.create(...)` to
        OpenAI `client.chat.completions.create(...)`.
        """

        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_result":
                            text_parts.append(f"[TOOL RESULT]: {block.get('content', '')}")
                        elif block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\\n".join(text_parts)

            openai_messages.append({"role": role, "content": content})

        # Try OpenAI SDK
        client = self._get_openai_client()
        if client is not None:
            try:
                kwargs = {"model": model or self.model, "messages": openai_messages, "max_tokens": max_tokens}
                if tools:
                    kwargs["tools"] = tools
                    tc_type = (tool_choice or {}).get("type", "auto")
                    kwargs["tool_choice"] = tc_type

                response = client.chat.completions.create(**kwargs)
                return _OpenAIResponse(response)
            except Exception:
                pass

        # Direct HTTP fallback
        return self._messages_create_http(model or self.model, openai_messages, max_tokens, tools, tool_choice)

    def _messages_create_http(self, model: str, messages: list, max_tokens: int,
                               tools: list, tool_choice: dict) -> Any:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = (tool_choice or {}).get("type", "auto")

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return _OpenAIResponse(data)
        except Exception as e:
            raise RuntimeError(f"Pollination AI API call failed: {e}")


class _OpenAIResponse:
    """Wraps OpenAI response to look like Anthropic response for the pipeline."""

    def __init__(self, response):
        if isinstance(response, dict):
            self._data = response
            choice = response.get("choices", [{}])[0]
            msg = choice.get("message", {})
            self.content = [_TextBlock(msg.get("content", ""))]
            for tc in msg.get("tool_calls", []):
                self.content.append(_ToolUseBlock(tc))
        else:
            self._data = {}
            choice = getattr(response, "choices", [None])[0] if hasattr(response, "choices") else None
            msg = getattr(choice, "message", None) if choice else None
            text = getattr(msg, "content", "") or ""
            self.content = [_TextBlock(text)]
            for tc in getattr(msg, "tool_calls", []) or []:
                self.content.append(_ToolUseBlock(tc))


class _TextBlock:
    type = "text"
    def __init__(self, text):
        self.text = text
        self.thinking = ""


class _ToolUseBlock:
    type = "tool_use"
    def __init__(self, tc):
        self.id = tc.get("id", "")
        self.name = tc.get("function", {}).get("name", "")
        raw_args = tc.get("function", {}).get("arguments", {})
        if isinstance(raw_args, str):
            try:
                self.input = json.loads(raw_args)
            except Exception:
                self.input = {"raw": raw_args}
        else:
            self.input = raw_args
'''
    wrapper_file.write_text(wrapper_content, encoding="utf-8")
    patches.append("  ✓ Created client_wrapper.py")

    return True, "\\n".join(patches) if patches else "No patches needed (files not found)"


def run_pipeline(config: PipelineConfig, output_placeholder, status_placeholder,
                 env: dict, python_exe: str, extra_python_paths: list[str]) -> tuple[bool, str]:
    repo_path = get_repo_path()

    clean_env = {}
    for key in ["PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"]:
        if key in os.environ:
            clean_env[key] = os.environ[key]

    clean_env["PYTHONPATH"] = str(repo_path)
    for sp in extra_python_paths:
        clean_env["PYTHONPATH"] += os.pathsep + sp

    clean_env.update(env)
    for key in [k for k, v in clean_env.items() if not v]:
        del clean_env[key]

    # Force UTF-8 on Windows
    if os.name == "nt":
        clean_env["PYTHONIOENCODING"] = "utf-8"
        clean_env["PYTHONUTF8"] = "1"

    cmd = [
        python_exe, "-m",
        f"agentic_search_data_gen.domains.{config.domain}",
        "--seeds", config.seeds_file,
        "--output", config.output_dir,
        "--collection", config.collection,
        "--explore-model", config.explore_model,
        "--verify-model", config.verify_model,
        "--distract-model", config.distract_model,
        "--extend-model", config.extend_model,
        "--explore-max-iterations", str(config.explore_max_iterations),
        "--verify-max-retries", str(config.verify_max_retries),
        "--distract-max-iterations", str(config.distract_max_iterations),
        "--extend-max-iterations", str(config.extend_max_iterations),
        "--extension-rounds", str(config.extension_rounds),
        "--max-workers", str(config.max_workers),
    ]

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(repo_path), env=clean_env, text=True, bufsize=1,
        )
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            output_placeholder.markdown(f"```\\n{''.join(output_lines[-100:])}\\n```")
        process.wait()
        return (True, "Pipeline completed successfully!") if process.returncode == 0 \
               else (False, f"Pipeline failed with return code {process.returncode}")
    except Exception as e:
        return False, f"Error running pipeline: {str(e)}"


def get_generated_files(output_dir: str) -> list[str]:
    output_path = Path(output_dir)
    return [f.name for f in output_path.glob("*.json")] if output_path.exists() else []


# ─── Session state init ────────────────────────────────────────────────────────
def _init_state():
    for key in ["use_env_file", "deps_info"]:
        if key not in st.session_state:
            st.session_state[key] = None


_init_state()


def main():
    st.set_page_config(page_title="Context-1 Data Generation", page_icon="🔍", layout="wide")
    st.title("🔍 Context-1 Data Generation")
    st.markdown("Generate synthetic multi-hop search tasks using free LLM providers")

    # ─── Sidebar ─────────────────────────────────────────────────────────────
    st.sidebar.title("🔍 Configuration")

    st.sidebar.markdown("### 🔐 Source de Keys")
    use_env = st.sidebar.checkbox(
        "Usar archivo .env",
        value=st.session_state.get("use_env_file", False),
        key="use_env_file_cb",
    )
    st.session_state.use_env_file = use_env

    if use_env:
        st.sidebar.success("🌸 **Pollination AI** — gpt-oss-20b — sin API key")
        st.sidebar.caption("Usa `OPENAI_API_BASE` y `OPENAI_MODEL` del `.env` si están definidos")
    else:
        st.sidebar.caption("📝 Ingresa las API keys manualmente abajo.")

    st.sidebar.markdown("### 🔑 GitHub")
    github_token = st.sidebar.text_input("GitHub Token", type="password",
        help="Token PAT para clonar el repositorio", placeholder="ghp_...")

    st.sidebar.markdown("### ⚙️ Pipeline")
    domain = st.sidebar.selectbox("Dominio", options=list(DOMAIN_OPTIONS.keys()),
        format_func=lambda x: DOMAIN_OPTIONS[x], index=0)

    seeds_file_map = {
        "web": "agentic_search_data_gen/domains/web/seeds.txt",
        "sec": "agentic_search_data_gen/domains/sec/seeds.txt",
        "patents": "agentic_search_data_gen/domains/patents/seeds.txt",
        "epstein": "agentic_search_data_gen/domains/epstein/seeds.txt",
    }
    seeds_file = st.sidebar.text_input("Seeds File", value=seeds_file_map.get(domain, "seeds.txt"))
    output_dir = st.sidebar.text_input("Output Directory", value="output")
    collection = st.sidebar.text_input("ChromaDB Collection", value="context1-data")

    # Models (manual mode only)
    if not use_env:
        st.sidebar.markdown("### 🤖 Modelos")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            explore_model = st.selectbox("Explore", options=MODEL_OPTIONS,
                index=MODEL_OPTIONS.index("claude-sonnet-4-5"))
            distract_model = st.selectbox("Distract", options=MODEL_OPTIONS,
                index=MODEL_OPTIONS.index("claude-sonnet-4-5"))
        with col2:
            verify_model = st.selectbox("Verify", options=MODEL_OPTIONS,
                index=MODEL_OPTIONS.index("claude-opus-4-5"))
            extend_model = st.selectbox("Extend", options=MODEL_OPTIONS,
                index=MODEL_OPTIONS.index("claude-sonnet-4-5"))
    else:
        explore_model = verify_model = distract_model = extend_model = "openai/gpt-oss-20b"

    st.sidebar.markdown("### 🔄 Límites")
    col_l1, col_l2 = st.sidebar.columns(2)
    with col_l1:
        explore_max = st.number_input("Explore", min_value=1, max_value=200, value=20, key="explore_iter")
        verify_max = st.number_input("Verify Retries", min_value=1, max_value=20, value=3, key="verify_retries")
        distract_max = st.number_input("Distract", min_value=1, max_value=200, value=15, key="distract_iter")
    with col_l2:
        extend_max = st.number_input("Extend", min_value=1, max_value=200, value=20, key="extend_iter")
        ext_rounds = st.number_input("Ext Rounds", min_value=0, max_value=20, value=0, key="ext_rounds")
        max_workers = st.number_input("Workers", min_value=1, max_value=32, value=8, key="max_workers")

    if not use_env:
        st.sidebar.markdown("### 🔑 API Keys")
        anthropic_key = st.sidebar.text_input("Anthropic", type="password", placeholder="sk-ant-...")
        openai_key = st.sidebar.text_input("OpenAI", type="password", placeholder="sk-...")
        serper_key = st.sidebar.text_input("Serper", type="password", placeholder="...")
        jina_key = st.sidebar.text_input("Jina", type="password", placeholder="...")
        chroma_key = st.sidebar.text_input("Chroma API", type="password", placeholder="...")
        chroma_db = st.sidebar.text_input("Chroma DB", type="password", placeholder="...")
    else:
        anthropic_key = openai_key = serper_key = jina_key = ""
        chroma_key = chroma_db = ""

    config = PipelineConfig(
        repo_url=REPO_URL, domain=domain,
        seeds_file=seeds_file, output_dir=output_dir, collection=collection,
        explore_model=explore_model, verify_model=verify_model,
        distract_model=distract_model, extend_model=extend_model,
        explore_max_iterations=st.session_state.explore_iter,
        verify_max_retries=st.session_state.verify_retries,
        distract_max_iterations=st.session_state.distract_iter,
        extend_max_iterations=st.session_state.extend_iter,
        extension_rounds=st.session_state.ext_rounds,
        max_workers=st.session_state.max_workers,
        anthropic_api_key=anthropic_key, openai_api_key=openai_key,
        serper_api_key=serper_key, jina_api_key=jina_key,
        chroma_api_key=chroma_key, chroma_database=chroma_db,
        use_env_file=use_env,
        pollination_model="openai/gpt-oss-20b",
    )

    # ─── Main area ─────────────────────────────────────────────────────────────
    use_env = st.session_state.use_env_file
    if use_env:
        st.success("**🌸 Modo Pollination AI** — gpt-oss-20b — sin API key necesaria")
    else:
        st.info("**📝 Modo manual** — Activa *Usar archivo .env* para usar Pollination AI.")

    tab1, tab2 = st.tabs(["🚀 Run Pipeline", "📊 Results"])

    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Pipeline Status")
            output_placeholder = st.empty()
            status_placeholder = st.empty()
        with col2:
            st.subheader("Quick Info")
            st.info(f"""
**Dominio:** {DOMAIN_OPTIONS.get(config.domain, config.domain)}
**Explore:** `{config.explore_model}`
**Verify:** `{config.verify_model}`
**Output:** `{config.output_dir}`
**Collection:** `{config.collection}`
""")
        run_clicked = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

        if run_clicked:
            if not github_token:
                st.error("⚠️ Ingresa tu GitHub Token en el menú izquierdo.")
                return

            # Step 1: Clone
            with st.spinner("📦 Cloning repository..."):
                success, message, repo_path = clone_repo_if_needed(github_token)
            if not success:
                st.error(f"❌ {message}")
                return
            st.success(f"✅ {message}")

            # Step 2: Install deps
            deps_info = st.session_state.get("deps_info")
            if deps_info is None:
                with st.spinner("📦 Installing project dependencies..."):
                    deps_ok, deps_msg, python_exe, sp_paths = install_project_deps(repo_path)
                if deps_ok:
                    st.success(f"✅ {deps_msg}")
                    st.session_state.deps_info = (python_exe, sp_paths)
                else:
                    st.warning(f"⚠️ {deps_msg}")
                    python_exe = sys.executable
                    sp_paths = []
                    st.session_state.deps_info = (python_exe, sp_paths)
            else:
                python_exe, sp_paths = deps_info

            # Step 3: Load env from .env
            env_vars = load_env_file(repo_path)
            st.markdown("**🔍 Debug — Loaded env vars:**")
            if "__ERROR__" in env_vars:
                st.error(env_vars["__ERROR__"])
            else:
                for k, v in env_vars.items():
                    st.write(f"  `{k}` = `{v[:8]}...`")

            # Step 4: Patch pipeline for Pollination AI
            if use_env:
                with st.spinner("🔧 Patching pipeline for Pollination AI..."):
                    patch_ok, patch_msg = patch_pipeline_for_pollination(repo_path)
                st.info(f"LLM Provider patches:\\n{patch_msg}")

                # Set Pollination AI env vars (don't override existing OPENAI_API_BASE/OPENAI_MODEL if user set them)
                if "OPENAI_API_BASE" not in env_vars or not env_vars.get("OPENAI_API_BASE"):
                    env_vars["OPENAI_API_BASE"] = "https://gen.pollinations.ai/v1"
                if "OPENAI_API_KEY" not in env_vars:
                    env_vars["OPENAI_API_KEY"] = "not-needed"
                if "OPENAI_MODEL" not in env_vars:
                    env_vars["OPENAI_MODEL"] = "openai/gpt-oss-20b"
                # Remove Anthropic key so pipeline doesn't try to use it
                env_vars.pop("ANTHROPIC_API_KEY", None)
                env_vars.pop("BASETEN_API_KEY", None)
                st.success("🌸 **Pollination AI** activo — modelo `openai/gpt-oss-20b`")

            # Step 5: Validate
            validation_errors = validate_api_keys_from_env(env_vars, config.domain, use_env)
            if validation_errors:
                st.error("❌ **API Key Validation Failed:**")
                for error in validation_errors:
                    st.write(f"• {error}")
                return

            # Step 6: Run
            status_placeholder.info("🚀 Running pipeline...")
            with st.spinner("⚙️ Running..."):
                success, message = run_pipeline(
                    config, output_placeholder, status_placeholder,
                    env_vars, python_exe, sp_paths,
                )
            if success:
                status_placeholder.success(f"✅ {message}")
            else:
                status_placeholder.error(f"❌ {message}")

    with tab2:
        st.subheader("Generated Files")
        output_path = get_repo_path() / config.output_dir
        if output_path.exists():
            files = get_generated_files(str(output_path))
            if files:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Tasks", len(files))
                c2.metric("Domain", config.domain.upper())
                c3.metric("Output Dir", config.output_dir)
                st.markdown("---")
                for file in files:
                    with st.expander(f"📄 {file}"):
                        try:
                            import json
                            with open(output_path / file) as f:
                                st.json(json.load(f))
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.info("No JSON files found. Run the pipeline first.")
        else:
            st.info("Output directory not found. Run the pipeline first.")


if __name__ == "__main__":
    main()