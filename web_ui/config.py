"""
Configuration dataclass and validation for Context-1 Data Generation Pipeline UI.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List


# Domain options with display names
DOMAIN_OPTIONS = {
    "web": "Web Search Tasks",
    "sec": "SEC Filing Tasks",
    "patents": "Patent Prior-Art Tasks",
    "epstein": "Email Search Tasks",
}

# Model options
MODEL_OPTIONS = [
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
]

# Default configuration
DEFAULT_CONFIG = {
    "explore_model": "claude-sonnet-4-5",
    "verify_model": "claude-opus-4-5",
    "distract_model": "claude-sonnet-4-5",
    "extend_model": "claude-sonnet-4-5",
    "explore_max_iterations": 20,
    "verify_max_retries": 3,
    "distract_max_iterations": 15,
    "extend_max_iterations": 20,
    "extension_rounds": 0,
    "max_workers": 8,
}


@dataclass
class PipelineConfig:
    """Configuration for the data generation pipeline."""
    
    # Repository settings
    repo_url: str = "https://github.com/developerjeremylive/context-1-data-gen-etheroi.git"
    domain: str = "web"
    
    # Input/Output settings
    seeds_file: str = "seeds.txt"
    output_dir: str = "output"
    collection: str = "context1-data"
    
    # Model settings
    explore_model: str = "claude-sonnet-4-5"
    verify_model: str = "claude-sonnet-4-5"
    distract_model: str = "claude-sonnet-4-5"
    extend_model: str = "claude-sonnet-4-5"
    
    # Iteration limits
    explore_max_iterations: int = 20
    verify_max_retries: int = 3
    distract_max_iterations: int = 15
    extend_max_iterations: int = 20
    extension_rounds: int = 0
    max_workers: int = 8
    
    # API Keys (manual entry or .env mode)
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    serper_api_key: str = ""
    jina_api_key: str = ""
    chroma_api_key: str = ""
    chroma_database: str = ""
    
    # Use .env file flag
    use_env_file: bool = False
    
    def to_env_vars(self) -> dict:
        """Convert API keys to environment variables.
        
        When use_env_file is True, reads from actual environment variables
        (which will be populated by load_env_file in app.py).
        """
        env = {}
        if self.use_env_file:
            # Read from actual environment (loaded from .env by app.py)
            keys = [
                "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SERPER_API_KEY",
                "JINA_API_KEY", "CHROMA_API_KEY", "CHROMA_DATABASE",
            ]
            for key in keys:
                val = os.getenv(key, "")
                if val:
                    env[key] = val
        else:
            # Use manually entered keys
            if self.anthropic_api_key:
                env["ANTHROPIC_API_KEY"] = self.anthropic_api_key
            if self.openai_api_key:
                env["OPENAI_API_KEY"] = self.openai_api_key
            if self.serper_api_key:
                env["SERPER_API_KEY"] = self.serper_api_key
            if self.jina_api_key:
                env["JINA_API_KEY"] = self.jina_api_key
            if self.chroma_api_key:
                env["CHROMA_API_KEY"] = self.chroma_api_key
            if self.chroma_database:
                env["CHROMA_DATABASE"] = self.chroma_database
        return env


def validate_api_keys(config: PipelineConfig) -> List[str]:
    """
    Validate that required API keys are present for the selected domain.
    
    When use_env_file is True, reads from actual environment variables
    (which are loaded from .env by app.py and set in the subprocess env).
    
    Returns:
        List of error messages for missing keys. Empty list if all keys are present.
    """
    errors = []
    
    # Helper to get key value (from env if use_env_file, else from config fields)
    def get_key(env_name: str, config_val: str = "") -> str:
        if config.use_env_file:
            return os.getenv(env_name, "")
        return config_val
    
    anthropic_key = get_key("ANTHROPIC_API_KEY", config.anthropic_api_key)
    openai_key = get_key("OPENAI_API_KEY", config.openai_api_key)
    serper_key = get_key("SERPER_API_KEY", config.serper_api_key)
    jina_key = get_key("JINA_API_KEY", config.jina_api_key)
    chroma_key = get_key("CHROMA_API_KEY", config.chroma_api_key)
    chroma_db = get_key("CHROMA_DATABASE", config.chroma_database)
    
    # Required for all domains
    if not anthropic_key:
        errors.append("Anthropic API key is required for all domains")
    
    # Domain-specific validation
    if config.domain == "web":
        if not serper_key:
            errors.append("Serper API key is required for web domain")
        if not jina_key:
            errors.append("Jina API key is required for web domain")
        if not openai_key:
            errors.append("OpenAI API key is required for web domain")
        if not chroma_key:
            errors.append("Chroma API key is required for web domain")
        if not chroma_db:
            errors.append("Chroma Database is required for web domain")
    
    elif config.domain == "sec":
        if not openai_key:
            errors.append("OpenAI API key is required for SEC domain")
        if not chroma_key:
            errors.append("Chroma API key is required for SEC domain")
        if not chroma_db:
            errors.append("Chroma Database is required for SEC domain")
    
    elif config.domain == "patents":
        if not openai_key:
            errors.append("OpenAI API key is required for patents domain")
        if not chroma_key:
            errors.append("Chroma API key is required for patents domain")
        if not chroma_db:
            errors.append("Chroma Database is required for patents domain")
    
    elif config.domain == "epstein":
        if not openai_key:
            errors.append("OpenAI API key is required for email domain")
        if not chroma_key:
            errors.append("Chroma API key is required for email domain")
        if not chroma_db:
            errors.append("Chroma Database is required for email domain")
    
    return errors


def build_command(config: PipelineConfig, repo_path: str = None) -> List[str]:
    """
    Build the command to run the pipeline.
    
    Args:
        config: Pipeline configuration
        repo_path: Path to the repository (optional)
    
    Returns:
        List of command arguments
    """
    cmd = [
        sys.executable,
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
    
    return cmd