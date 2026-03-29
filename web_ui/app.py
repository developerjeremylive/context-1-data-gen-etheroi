"""
Streamlit Web UI for Context-1 Data Generation Pipeline
Provides a graphical interface to configure and run the data generation pipeline.
"""
import os
import sys
import subprocess
import tempfile
import shutil
import site
from pathlib import Path

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
    """Load environment variables from .env file in the repo."""
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
    """Clone the repository if it doesn't exist locally."""
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
    """Find the Python executable and site-packages in the repo's venv."""
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
    """Install project dependencies using uv."""
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


def patch_pipeline_for_baseten(repo_path: Path) -> tuple[bool, str]:
    """
    Patch the pipeline modules to use Baseten (OpenAI-compatible) instead of Anthropic
    when BASETEN_API_KEY is present in the .env file.
    
    Baseten provides gpt-oss-20b and other models via OpenAI-compatible API.
    """
    patches = []
    
    # ─── Patch core/utils.py — add get_baseten_client ────────────────────────
    utils_file = repo_path / "agentic_search_data_gen" / "core" / "utils.py"
    if utils_file.exists():
        content = utils_file.read_text(encoding="utf-8")
        # Add get_baseten_client after get_anthropic_client
        if "def get_baseten_client():" not in content:
            baseten_func = '''
def get_baseten_client():
    """Get OpenAI-compatible client for Baseten API."""
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("BASETEN_API_KEY"), base_url="https://app.baseten.co")
'''
            # Insert after get_anthropic_client
            if "def get_anthropic_client():" in content:
                idx = content.find("def get_anthropic_client():")
                content = content[:idx] + baseten_func + content[idx:]
            elif "def count_tokens" in content:
                idx = content.find("def count_tokens")
                content = content[:idx] + baseten_func + content[idx:]
            utils_file.write_text(content, encoding="utf-8")
            patches.append("  ✓ Patched core/utils.py (added get_baseten_client)")
    
    # ─── Patch __main__.py ──────────────────────────────────────────────────
    main_file = repo_path / "agentic_search_data_gen" / "domains" / "web" / "__main__.py"
    if main_file.exists():
        content = main_file.read_text(encoding="utf-8")
        patched = content.replace(
            "from ...core.utils import get_anthropic_client",
            "from ...core.utils import get_anthropic_client, get_baseten_client"
        )
        patched = patched.replace(
            "client = get_anthropic_client()",
            "from dotenv import load_dotenv\n    load_dotenv()\n    baseten_key = os.getenv('BASETEN_API_KEY', '')\n    if baseten_key:\n        from openai import OpenAI\n        client = OpenAI(api_key=baseten_key, base_url='https://app.baseten.co')\n    else:\n        client = get_anthropic_client()"
        )
        if patched != content:
            main_file.write_text(patched, encoding="utf-8")
            patches.append("  ✓ Patched __main__.py")
    
    # ─── Patch explore.py ────────────────────────────────────────────────────
    explore_file = repo_path / "agentic_search_data_gen" / "domains" / "web" / "explore.py"
    if explore_file.exists():
        content = explore_file.read_text(encoding="utf-8")
        # Change client = get_anthropic_client() in __init__
        patched = content.replace(
            "from ...core.utils import get_anthropic_client, strip_links",
            "from ...core.utils import get_anthropic_client, strip_links, get_baseten_client"
        )
        patched = patched.replace(
            "    def __init__(self, model: str = \"claude-sonnet-4-5\", max_iterations: int = 20):\n        client = get_anthropic_client()\n        super().__init__(client, model, max_iterations)",
            "    def __init__(self, model: str = \"claude-sonnet-4-5\", max_iterations: int = 20):\n        from dotenv import load_dotenv\n        load_dotenv()\n        baseten_key = os.getenv('BASETEN_API_KEY', '')\n        if baseten_key:\n            from openai import OpenAI\n            client = OpenAI(api_key=baseten_key, base_url='https://app.baseten.co')\n        else:\n            client = get_anthropic_client()\n        super().__init__(client, model, max_iterations)"
        )
        if patched != content:
            explore_file.write_text(patched, encoding="utf-8")
            patches.append(f"  ✓ Patched explore.py")
    
    # ─── Patch verify.py ────────────────────────────────────────────────────
    verify_file = repo_path / "agentic_search_data_gen" / "domains" / "web" / "verify.py"
    if verify_file.exists():
        content = verify_file.read_text(encoding="utf-8")
        patched = content.replace(
            "from ...core.utils import (\n    count_matching_quotes,\n    min_required_matches,\n    parse_quotes,\n    get_anthropic_client\n)",
            "from ...core.utils import (\n    count_matching_quotes,\n    min_required_matches,\n    parse_quotes,\n    get_anthropic_client,\n    get_baseten_client\n)"
        )
        # Patch run_single_item_extraction in verify.py
        # Find the method and add baseten client detection
        patched = patched.replace(
            "def get_anthropic_client():",
            "def get_baseten_client():\n    from openai import OpenAI\n    import os\n    return OpenAI(api_key=os.getenv('BASETEN_API_KEY'), base_url='https://app.baseten.co')\n\ndef get_anthropic_client():"
        )
        if patched != content:
            verify_file.write_text(patched, encoding="utf-8")
            patches.append(f"  ✓ Patched verify.py (added get_baseten_client)")
    
    return True, "\n".join(patches) if patches else "No patches needed (files not found)"


def run_pipeline(
    config: PipelineConfig,
    output_placeholder,
    status_placeholder,
    env: dict,
    python_exe: str,
    extra_python_paths: list[str],
    use_env: bool = False,
    pollination_model: str = "",
) -> tuple[bool, str]:
    """Run the pipeline with the given environment and Python paths."""
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
    
    # Force UTF-8 encoding for all output on Windows
    if os.name == "nt":
        clean_env["PYTHONIOENCODING"] = "utf-8"
        clean_env["PYTHONUTF8"] = "1"
    
    cmd = [
        python_exe,
        "-m",
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
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(repo_path),
            env=clean_env,
            text=True,
            bufsize=1,
        )
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            output_placeholder.markdown(f"```\n{''.join(output_lines[-100:])}\n```")
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
    for key in ["use_env_file", "deps_info", "pollination_model"]:
        if key not in st.session_state:
            if key == "pollination_model":
                st.session_state[key] = "openai/gpt-oss-20b"
            else:
                st.session_state[key] = None


_init_state()


def main():
    st.set_page_config(page_title="Context-1 Data Generation",
                        page_icon="🔍", layout="wide")
    
    st.title("🔍 Context-1 Data Generation")
    st.markdown("Generate synthetic multi-hop search tasks")
    
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
        st.sidebar.markdown("### 🌸 LLM Provider (desde .env)")
        st.sidebar.caption(
            "Usa BASETEN_API_KEY del `.env` → gpt-oss-20b via Baseten\n"
            "Sin API key → intenta Pollination AI (gen.pollinations.ai)"
        )
        st.sidebar.caption("✅ `.env` mode activo")
    else:
        st.sidebar.caption("📝 Ingresa las API keys manualmente abajo.")
    
    st.sidebar.markdown("### 🔑 GitHub")
    github_token = st.sidebar.text_input(
        "GitHub Token",
        type="password",
        help="Token PAT para clonar el repositorio",
        placeholder="ghp_...",
    )
    
    st.sidebar.markdown("### ⚙️ Pipeline")
    domain = st.sidebar.selectbox(
        "Dominio",
        options=list(DOMAIN_OPTIONS.keys()),
        format_func=lambda x: DOMAIN_OPTIONS[x],
        index=0,
    )
    
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
        explore_model = "openai/gpt-oss-20b"
        verify_model = "openai/gpt-oss-20b"
        distract_model = "openai/gpt-oss-20b"
        extend_model = "openai/gpt-oss-20b"
    
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
    
    pollination_model = "openai/gpt-oss-20b"
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
        pollination_model=pollination_model,
    )
    
    # ─── Main area ─────────────────────────────────────────────────────────────
    use_env = st.session_state.use_env_file
    
    if use_env:
        st.success(
            "**🔐 Modo .env activo** — LLM desde `.env`:\n"
            "- Si `BASETEN_API_KEY` existe → usa **Baseten** (gpt-oss-20b)\n"
            "- Si no → usa **Pollination AI** (gen.pollinations.ai)"
        )
    else:
        st.info("**📝 Modo manual** — Activa *Usar archivo .env* para usar el `.env` del repo.")
    
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
**Distract:** `{config.distract_model}`
**Extend:** `{config.extend_model}`
**Output:** `{config.output_dir}`
**Collection:** `{config.collection}`
""")
        
        run_clicked = st.button("🚀 Run Pipeline", type="primary",
                                use_container_width=True)
        
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
            
            # Step 4: Patch pipeline for LLM provider
            if use_env:
                with st.spinner("🔧 Patching pipeline for LLM provider..."):
                    patch_ok, patch_msg = patch_pipeline_for_baseten(repo_path)
                st.info(f"LLM Provider patches:\n{patch_msg}")
                
                baseten_key = env_vars.get("BASETEN_API_KEY", "")
                if baseten_key:
                    env_vars["OPENAI_API_KEY"] = baseten_key
                    env_vars["OPENAI_API_BASE"] = "https://app.baseten.co"
                    st.success("🌸 **Baseten** activo — modelo `gpt-oss-20b` via Baseten")
                else:
                    env_vars["OPENAI_API_BASE"] = "https://gen.pollinations.ai/v1"
                    env_vars["OPENAI_API_KEY"] = "not-needed"
                    env_vars.pop("ANTHROPIC_API_KEY", None)
                    st.success("🌸 **Pollination AI** activo — sin API key")
            
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
                    use_env=use_env, pollination_model=pollination_model,
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