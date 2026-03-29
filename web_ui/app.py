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


def run_pipeline(config: PipelineConfig, output_placeholder) -> tuple[bool, str]:
    """Run the pipeline with the given configuration."""
    repo_path = get_repo_path()
    
    # Set PYTHONPATH to include the repo directory
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_path)
    
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


def render_sidebar() -> tuple[str, PipelineConfig]:
    """Render the sidebar configuration and return the config."""
    
    st.sidebar.title("🔍 Configuration")
    
    # GitHub Token
    github_token = st.sidebar.text_input(
        "GitHub Token",
        type="password",
        help="Personal access token for cloning the repository",
    )
    
    # Basic Configuration
    st.sidebar.subheader("Basic Settings")
    
    domain = st.sidebar.selectbox(
        "Domain",
        options=list(DOMAIN_OPTIONS.keys()),
        format_func=lambda x: DOMAIN_OPTIONS[x],
        index=0,
    )
    
    repo_url = st.sidebar.text_input("Repository URL", REPO_URL)
    
    seeds_file = st.sidebar.text_input(
        "Seeds File Path",
        value="seeds.txt",
        help="Path to the seeds file (relative to repo or absolute)",
    )
    
    output_dir = st.sidebar.text_input(
        "Output Directory",
        value="output",
        help="Directory to save generated JSON files",
    )
    
    collection = st.sidebar.text_input(
        "ChromaDB Collection Name",
        value="context1-data",
        help="Name of the ChromaDB collection for indexing",
    )
    
    # Model Configuration
    st.sidebar.subheader("Model Selection")
    
    explore_model = st.sidebar.selectbox(
        "Explore Model",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index("claude-sonnet-4-5"),
    )
    
    verify_model = st.sidebar.selectbox(
        "Verify Model",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index("claude-sonnet-4-5"),
    )
    
    distract_model = st.sidebar.selectbox(
        "Distract Model",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index("claude-sonnet-4-5"),
    )
    
    extend_model = st.sidebar.selectbox(
        "Extend Model",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index("claude-sonnet-4-5"),
    )
    
    # Iteration Limits
    st.sidebar.subheader("Iteration Limits")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        explore_max_iterations = st.number_input(
            "Explore Max Iterations",
            min_value=1,
            max_value=100,
            value=20,
        )
        verify_max_retries = st.number_input(
            "Verify Max Retries",
            min_value=1,
            max_value=20,
            value=3,
        )
        distract_max_iterations = st.number_input(
            "Distract Max Iterations",
            min_value=1,
            max_value=100,
            value=15,
        )
    
    with col2:
        extend_max_iterations = st.number_input(
            "Extend Max Iterations",
            min_value=1,
            max_value=100,
            value=20,
        )
        extension_rounds = st.number_input(
            "Extension Rounds",
            min_value=0,
            max_value=20,
            value=0,
        )
        max_workers = st.number_input(
            "Max Workers",
            min_value=1,
            max_value=32,
            value=8,
        )
    
    # API Keys
    st.sidebar.subheader("API Keys")
    
    anthropic_api_key = st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        help="Required for all domains",
    )
    
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Required for web, SEC, email, and patents domains",
    )
    
    serper_api_key = st.sidebar.text_input(
        "Serper API Key",
        type="password",
        help="Required for web domain",
    )
    
    jina_api_key = st.sidebar.text_input(
        "Jina API Key",
        type="password",
        help="Required for web domain",
    )
    
    chroma_api_key = st.sidebar.text_input(
        "Chroma API Key",
        type="password",
        help="Required for web, SEC, email, and patents domains",
    )
    
    chroma_database = st.sidebar.text_input(
        "Chroma Database",
        type="password",
        help="Required for web, SEC, email, and patents domains",
    )
    
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
    )
    
    return github_token, config


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
    github_token, config = render_sidebar()
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["🚀 Run Pipeline", "📊 Results"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Pipeline Status")
            output_placeholder = st.empty()
            status_placeholder = st.empty()
        
        with col2:
            st.subheader("Quick Info")
            st.info("""
            **Domains Available:**
            - 🌐 Web Search Tasks
            - 📈 SEC Filing Tasks  
            - 📋 Patent Prior-Art Tasks
            - 📧 Email Search Tasks
            
            **Models Available:**
            - Claude Sonnet 4.5
            - Claude Opus 4.5
            - Claude 3.5 Sonnet
            - Claude 3 Opus
            """)
        
        # Validation and Run Button
        if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
            if not github_token:
                st.error("Please enter a GitHub token in the sidebar")
                return
            
            # Validate API keys
            validation_errors = validate_api_keys(config)
            if validation_errors:
                st.error("API Key Validation Failed:")
                for error in validation_errors:
                    st.write(f"• {error}")
                return
            
            # Clone repo if needed
            with st.spinner("Cloning repository..."):
                success, message = clone_repo_if_needed(github_token)
                if not success:
                    st.error(f"Failed to clone repository: {message}")
                    return
                st.success(message)
            
            # Run pipeline
            status_placeholder.info("🚀 Running pipeline... This may take a while.")
            
            with st.spinner("Running pipeline..."):
                success, message = run_pipeline(config, output_placeholder)
            
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
                st.success(f"Found {len(files)} generated files")
                
                for file in files:
                    with st.expander(f"📄 {file}"):
                        file_path = output_path / file
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                            st.code(content, language="json")
                        except Exception as e:
                            st.error(f"Error reading file: {e}")
            else:
                st.info("No JSON files found in the output directory. Run the pipeline first.")
        else:
            st.info("Output directory not found. Run the pipeline first.")


if __name__ == "__main__":
    main()