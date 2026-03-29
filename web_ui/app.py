"""
Streamlit Web UI for Context-1 Data Generation Pipeline
Provides a graphical interface to configure and run the data generation pipeline.
"""
import os
import sys
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

import streamlit as st

from config import (
    DEFAULT_CONFIG,
    DOMAIN_OPTIONS,
    MODEL_OPTIONS,
    PipelineConfig,
    build_command,
    validate_api_keys,
)


# Repository configuration
REPO_URL = "https://github.com/developerjeremylive/context-1-data-gen-etheroi.git"
REPO_NAME = "context-1-data-gen-etheroi"


def get_repo_path() -> Path:
    """Get the local path to the cloned repository."""
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
    return env_vars


def clone_repo_if_needed(token: str) -> tuple[bool, str]:
    """Clone the repository if it doesn't exist locally."""
    repo_path = get_repo_path()
    
    if repo_path.exists():
        return True, f"Repository already exists at {repo_path}"
    
    try:
        # Use HTTPS with token for authentication
        clone_url = f"https://x-access-token:{token}@github.com/developerjeremylive/context-1-data-gen-etheroi.git"
        subprocess.run(
            ["git", "clone", clone_url, str(repo_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return True, f"Successfully cloned repository to {repo_path}"
    except subprocess.CalledProcessError as e:
        return False, f"Failed to clone repository: {e.stderr}"


def run_pipeline(
    config: PipelineConfig,
    output_placeholder,
    status_placeholder,
    use_env_file: bool = False,
) -> tuple[bool, str]:
    """Run the pipeline with the given configuration."""
    repo_path = get_repo_path()
    
    # Set PYTHONPATH to include the repo directory
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_path)
    
    # Load .env file if option is enabled
    if use_env_file:
        env_from_file = load_env_file(repo_path)
        for key, value in env_from_file.items():
            env[key] = value
    
    # Build and run the command
    cmd = build_command(config, str(repo_path))
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(repo_path),
            env=env,
            text=True,
            bufsize=1,
        )
        
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            output_placeholder.markdown(f"```\n{''.join(output_lines[-100:])}\n```")
        
        process.wait()
        
        if process.returncode == 0:
            return True, "Pipeline completed successfully!"
        else:
            return False, f"Pipeline failed with return code {process.returncode}"
            
    except Exception as e:
        return False, f"Error running pipeline: {str(e)}"


def get_generated_files(output_dir: str) -> list[str]:
    """Get list of generated JSON files in the output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    return [f.name for f in output_path.glob("*.json")]


def render_sidebar() -> tuple[str, PipelineConfig, bool]:
    """Render the sidebar configuration and return the config.
    
    Returns:
        Tuple of (github_token, config, use_env_file)
    """
    
    st.sidebar.title("🔍 Configuration")
    
    # ─── Use .env File Toggle ───────────────────────────────────────────────
    st.sidebar.markdown("### 🔐 Source de Keys")
    use_env_file = st.sidebar.toggle(
        "Usar archivo .env",
        value=False,
        help="Si está activo, las API keys se leen del archivo .env en el repositorio. "
             "Se ocultan los campos de API keys en el menú izquierdo.",
    )
    
    if use_env_file:
        st.sidebar.caption(
            "✅ Cargando keys desde `.env` — campos ocultos.",
            help="Las siguientes variables se usan: ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                 "SERPER_API_KEY, JINA_API_KEY, CHROMA_API_KEY, CHROMA_DATABASE",
        )
    else:
        st.sidebar.caption("📝 Ingresa las API keys manualmente abajo.")
    
    # ─── GitHub Token ───────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔑 GitHub")
    github_token = st.sidebar.text_input(
        "GitHub Token",
        type="password",
        help="Token PAT para clonar el repositorio (solo lectura está bien)",
        placeholder="ghp_...",
    )
    
    # ─── Basic Settings ────────────────────────────────────────────────────
    st.sidebar.markdown("### ⚙️ Pipeline")
    
    domain = st.sidebar.selectbox(
        "Dominio",
        options=list(DOMAIN_OPTIONS.keys()),
        format_func=lambda x: DOMAIN_OPTIONS[x],
        index=0,
    )
    
    repo_url = st.sidebar.text_input("Repository URL", REPO_URL)
    
    seeds_file_map = {
        "web": "agentic_search_data_gen/domains/web/seeds.txt",
        "sec": "agentic_search_data_gen/domains/sec/seeds.txt",
        "patents": "agentic_search_data_gen/domains/patents/seeds.txt",
        "epstein": "agentic_search_data_gen/domains/epstein/seeds.txt",
    }
    default_seeds = seeds_file_map.get(domain, "seeds.txt")
    seeds_file = st.sidebar.text_input(
        "Seeds File",
        value=default_seeds,
        help="Ruta al archivo de seeds (relativa al repo)",
    )
    
    output_dir = st.sidebar.text_input(
        "Output Directory",
        value="output",
        help="Directorio donde se guardan los JSON generados",
    )
    
    collection = st.sidebar.text_input(
        "ChromaDB Collection",
        value="context1-data",
        help="Nombre de la colección en ChromaDB",
    )
    
    # ─── Model Selection ───────────────────────────────────────────────────
    st.sidebar.markdown("### 🤖 Modelos")
    
    col_model1, col_model2 = st.sidebar.columns(2)
    
    with col_model1:
        explore_model = st.selectbox(
            "Explore",
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index("claude-sonnet-4-5"),
        )
        distract_model = st.selectbox(
            "Distract",
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index("claude-sonnet-4-5"),
        )
    
    with col_model2:
        verify_model = st.selectbox(
            "Verify",
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index("claude-opus-4-5"),
        )
        extend_model = st.selectbox(
            "Extend",
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index("claude-sonnet-4-5"),
        )
    
    # ─── Iteration Limits ──────────────────────────────────────────────────
    st.sidebar.markdown("### 🔄 Límites")
    
    col_lim1, col_lim2 = st.sidebar.columns(2)
    
    with col_lim1:
        explore_max_iterations = st.number_input(
            "Explore Iterations",
            min_value=1, max_value=200,
            value=20,
        )
        verify_max_retries = st.number_input(
            "Verify Retries",
            min_value=1, max_value=20,
            value=3,
        )
        distract_max_iterations = st.number_input(
            "Distract Iterations",
            min_value=1, max_value=200,
            value=15,
        )
    
    with col_lim2:
        extend_max_iterations = st.number_input(
            "Extend Iterations",
            min_value=1, max_value=200,
            value=20,
        )
        extension_rounds = st.number_input(
            "Extension Rounds",
            min_value=0, max_value=20,
            value=0,
        )
        max_workers = st.number_input(
            "Max Workers",
            min_value=1, max_value=32,
            value=8,
        )
    
    # ─── API Keys (only when NOT using .env) ────────────────────────────────
    if not use_env_file:
        st.sidebar.markdown("### 🔑 API Keys")
        
        anthropic_api_key = st.sidebar.text_input(
            "Anthropic API Key",
            type="password",
            help="Requerido para todos los dominios (console.anthropic.com)",
            placeholder="sk-ant-...",
        )
        
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Requerido para web, SEC, patents, epstein",
            placeholder="sk-...",
        )
        
        serper_api_key = st.sidebar.text_input(
            "Serper API Key",
            type="password",
            help="Requerido para dominio web (serper.dev)",
            placeholder="...",
        )
        
        jina_api_key = st.sidebar.text_input(
            "Jina API Key",
            type="password",
            help="Requerido para dominio web (jina.ai)",
            placeholder="...",
        )
        
        chroma_api_key = st.sidebar.text_input(
            "Chroma API Key",
            type="password",
            help="Requerido para indexing",
            placeholder="...",
        )
        
        chroma_database = st.sidebar.text_input(
            "Chroma Database",
            type="password",
            help="Nombre de la base de datos en ChromaDB",
            placeholder="...",
        )
    else:
        # Hidden but present (empty strings — validation reads from env)
        anthropic_api_key = ""
        openai_api_key = ""
        serper_api_key = ""
        jina_api_key = ""
        chroma_api_key = ""
        chroma_database = ""
    
    # Build config
    config = PipelineConfig(
        repo_url=repo_url,
        domain=domain,
        seeds_file=seeds_file,
        output_dir=output_dir,
        collection=collection,
        explore_model=explore_model,
        verify_model=verify_model,
        distract_model=distract_model,
        extend_model=extend_model,
        explore_max_iterations=explore_max_iterations,
        verify_max_retries=verify_max_retries,
        distract_max_iterations=distract_max_iterations,
        extend_max_iterations=extend_max_iterations,
        extension_rounds=extension_rounds,
        max_workers=max_workers,
        anthropic_api_key=anthropic_api_key,
        openai_api_key=openai_api_key,
        serper_api_key=serper_api_key,
        jina_api_key=jina_api_key,
        chroma_api_key=chroma_api_key,
        chroma_database=chroma_database,
        use_env_file=use_env_file,
    )
    
    return github_token, config, use_env_file


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Context-1 Data Generation",
        page_icon="🔍",
        layout="wide",
    )
    
    st.title("🔍 Context-1 Data Generation")
    st.markdown("Generate synthetic multi-hop search tasks across multiple domains")
    
    # Render sidebar and get config
    github_token, config, use_env_file = render_sidebar()
    
    # ─── Env file status banner ─────────────────────────────────────────────
    if use_env_file:
        st.success(
            "**🔐 Modo .env activo** — Las API keys se leen del archivo `.env` "
            "en el repositorio. Los campos de keys fueron ocultados del menú."
        )
    else:
        st.info(
            "**📝 Modo manual** — Ingresa las API keys en el menú izquierdo. "
            "Alternativamente, activa el toggle *Usar archivo .env* para leerlas "
            "desde el archivo `.env` del repositorio."
        )
    
    # Main content area with tabs
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

**Modelo Explore:** `{config.explore_model}`
**Modelo Verify:** `{config.verify_model}`
**Modelo Distract:** `{config.distract_model}`
**Modelo Extend:** `{config.extend_model}`

**Output:** `{config.output_dir}`
**Collection:** `{config.collection}`

**Seeds:** `{config.seeds_file}`
""")
        
        # Run Button
        run_clicked = st.button(
            "🚀 Run Pipeline",
            type="primary",
            use_container_width=True,
        )
        
        if run_clicked:
            if not github_token:
                st.error("⚠️ Ingresa tu GitHub Token en el menú izquierdo para clonar el repositorio.")
                return
            
            # Validate API keys
            validation_errors = validate_api_keys(config)
            if validation_errors:
                st.error("❌ **API Key Validation Failed:**")
                for error in validation_errors:
                    st.write(f"• {error}")
                return
            
            # Clone repo if needed
            with st.spinner("📦 Cloning repository..."):
                success, message = clone_repo_if_needed(github_token)
                if not success:
                    st.error(f"❌ Failed to clone repository: {message}")
                    return
                st.success(f"✅ {message}")
            
            # Run pipeline
            status_placeholder.info("🚀 Running pipeline... This may take a while.")
            
            with st.spinner("⚙️ Running pipeline..."):
                success, message = run_pipeline(
                    config, output_placeholder, status_placeholder,
                    use_env_file=use_env_file,
                )
            
            if success:
                status_placeholder.success(f"✅ {message}")
            else:
                status_placeholder.error(f"❌ {message}")
    
    with tab2:
        st.subheader("Generated Files")
        
        output_dir = config.output_dir if config.output_dir else "output"
        
        # Check if output directory exists
        output_path = get_repo_path() / output_dir
        
        if output_path.exists():
            files = get_generated_files(str(output_path))
            
            if files:
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Total Tasks", len(files))
                col_m2.metric("Domain", config.domain.upper())
                col_m3.metric("Output Dir", output_dir)
                
                st.markdown("---")
                
                for file in files:
                    with st.expander(f"📄 {file}"):
                        file_path = output_path / file
                        try:
                            import json
                            with open(file_path) as f:
                                content = json.load(f)
                            st.json(content)
                        except Exception as e:
                            st.error(f"Error reading file: {e}")
            else:
                st.info("No JSON files found. Run the pipeline first.")
        else:
            st.info("Output directory not found. Run the pipeline first.")


if __name__ == "__main__":
    main()