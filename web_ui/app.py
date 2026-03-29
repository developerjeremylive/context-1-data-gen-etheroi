"""
Streamlit Web UI for Context-1 Data Generation Pipeline
Provides a graphical interface to configure and run the data generation pipeline.
"""
import os
import sys
import subprocess
import tempfile
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


def run_pipeline(config: PipelineConfig, output_placeholder, status_placeholder, env: dict) -> tuple[bool, str]:
    """Run the pipeline with the given environment."""
    repo_path = get_repo_path()
    full_env = os.environ.copy()
    full_env["PYTHONPATH"] = str(repo_path)
    full_env.update(env)
    cmd = build_command(config, str(repo_path))
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(repo_path), env=full_env, text=True, bufsize=1,
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


# ─── Session state init ───────────────────────────────────────────────────────
def _init_state():
    for key in ["use_env_file", "debug_env", "debug_loaded"]:
        if key not in st.session_state:
            st.session_state[key] = False if key == "use_env_file" else None


_init_state()


def render_sidebar() -> tuple[str, PipelineConfig]:
    """Render sidebar and return (github_token, config)."""
    st.sidebar.title("🔍 Configuration")
    
    # ─── .env Toggle (persisted via session_state) ─────────────────────────
    st.sidebar.markdown("### 🔐 Source de Keys")
    
    # Use checkbox (more reliable than toggle for persistence across re-renders)
    use_env = st.sidebar.checkbox(
        "Usar archivo .env",
        value=st.session_state.use_env_file,
        help="Si está activo, las API keys se leen del archivo .env en el repositorio. "
             "Se ocultan los campos de API keys en el menú izquierdo.",
        key="use_env_file_cb",
    )
    # Sync to session state
    st.session_state.use_env_file = use_env
    
    if use_env:
        st.sidebar.caption("✅ Cargando keys desde `.env` — campos ocultos.")
    else:
        st.sidebar.caption("📝 Ingresa las API keys manualmente abajo.")
    
    # ─── GitHub Token ───────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔑 GitHub")
    github_token = st.sidebar.text_input(
        "GitHub Token",
        type="password",
        help="Token PAT para clonar el repositorio",
        placeholder="ghp_...",
    )
    
    # ─── Pipeline Settings ─────────────────────────────────────────────────
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
    default_seeds = seeds_file_map.get(domain, "seeds.txt")
    
    seeds_file = st.sidebar.text_input("Seeds File", value=default_seeds)
    output_dir = st.sidebar.text_input("Output Directory", value="output")
    collection = st.sidebar.text_input("ChromaDB Collection", value="context1-data")
    
    # ─── Models ─────────────────────────────────────────────────────────────
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
    
    # ─── Limits ─────────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔄 Límites")
    col_l1, col_l2 = st.sidebar.columns(2)
    with col_l1:
        st.number_input("Explore Iterations", min_value=1, max_value=200,
                         value=20, key="explore_iter")
        st.number_input("Verify Retries", min_value=1, max_value=20,
                         value=3, key="verify_retries")
        st.number_input("Distract Iterations", min_value=1, max_value=200,
                         value=15, key="distract_iter")
    with col_l2:
        st.number_input("Extend Iterations", min_value=1, max_value=200,
                         value=20, key="extend_iter")
        st.number_input("Extension Rounds", min_value=0, max_value=20,
                         value=0, key="ext_rounds")
        st.number_input("Max Workers", min_value=1, max_value=32,
                         value=8, key="max_workers")
    
    # Read from session state (key = the widget key)
    explore_max = st.session_state.get("explore_iter", 20)
    verify_max  = st.session_state.get("verify_retries", 3)
    distract_max = st.session_state.get("distract_iter", 15)
    extend_max  = st.session_state.get("extend_iter", 20)
    ext_rounds  = st.session_state.get("ext_rounds", 0)
    max_workers = st.session_state.get("max_workers", 8)
    
    # ─── API Keys (hidden when using .env) ─────────────────────────────────
    if not use_env:
        st.sidebar.markdown("### 🔑 API Keys")
        anthropic_api_key = st.sidebar.text_input(
            "Anthropic", type="password", placeholder="sk-ant-...")
        openai_api_key = st.sidebar.text_input(
            "OpenAI", type="password", placeholder="sk-...")
        serper_api_key = st.sidebar.text_input(
            "Serper", type="password", placeholder="...")
        jina_api_key = st.sidebar.text_input(
            "Jina", type="password", placeholder="...")
        chroma_api_key = st.sidebar.text_input(
            "Chroma API Key", type="password", placeholder="...")
        chroma_database = st.sidebar.text_input(
            "Chroma Database", type="password", placeholder="...")
    else:
        anthropic_api_key = openai_api_key = serper_api_key = ""
        jina_api_key = chroma_api_key = chroma_database = ""
    
    config = PipelineConfig(
        repo_url=REPO_URL, domain=domain,
        seeds_file=seeds_file, output_dir=output_dir, collection=collection,
        explore_model=explore_model, verify_model=verify_model,
        distract_model=distract_model, extend_model=extend_model,
        explore_max_iterations=explore_max, verify_max_retries=verify_max,
        distract_max_iterations=distract_max, extend_max_iterations=extend_max,
        extension_rounds=ext_rounds, max_workers=max_workers,
        anthropic_api_key=anthropic_api_key, openai_api_key=openai_api_key,
        serper_api_key=serper_api_key, jina_api_key=jina_api_key,
        chroma_api_key=chroma_api_key, chroma_database=chroma_database,
        use_env_file=use_env,
    )
    
    return github_token, config


def main():
    st.set_page_config(page_title="Context-1 Data Generation",
                        page_icon="🔍", layout="wide")
    
    st.title("🔍 Context-1 Data Generation")
    st.markdown("Generate synthetic multi-hop search tasks")
    
    github_token, config = render_sidebar()
    
    # Mode banner
    use_env = st.session_state.use_env_file
    if use_env:
        st.success("**🔐 Modo .env activo** — Keys desde `.env`, campos ocultos.")
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
**Seeds:** `{config.seeds_file}`
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
            
            # Step 2: Load env vars from .env
            env_vars = load_env_file(repo_path)
            
            # Show debug info so user can see what was loaded
            st.markdown("**🔍 Debug — Loaded env vars:**")
            if "__ERROR__" in env_vars:
                st.error(env_vars["__ERROR__"])
            else:
                for k, v in env_vars.items():
                    display_val = v[:8] + "..." if len(v) > 8 else v
                    st.write(f"  `{k}` = `{display_val}`")
            
            # Step 3: Validate
            validation_errors = validate_api_keys_from_env(env_vars, config.domain)
            if validation_errors:
                st.error("❌ **API Key Validation Failed:**")
                for error in validation_errors:
                    st.write(f"• {error}")
                return
            
            # Step 4: Run
            status_placeholder.info("🚀 Running pipeline...")
            with st.spinner("⚙️ Running..."):
                success, message = run_pipeline(config, output_placeholder,
                                               status_placeholder, env_vars)
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