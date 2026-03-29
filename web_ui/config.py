"""
Configuration dataclass and validation for Context-1 Data Generation Pipeline UI.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

from typing import List


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
    "verify_model": "claude-sonnet-4-5",
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
    
    # API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    serper_api_key: str = ""
    jina_api_key: str = ""
    chroma_api_key: str = ""
    chroma_database: str = ""
    
    def to_env_vars(self) -> dict:
        """Convert API keys to environment variables."""
        env = {}
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
    
    Returns:
        List of error messages for missing keys. Empty list if all keys are present.
    """
    errors = []
    
    # Required for all domains
    if not config.anthropic_api_key:
        errors.append("Anthropic API key is required for all domains")
    
    # Domain-specific validation
    if config.domain == "web":
        if not config.serper_api_key:
            errors.append("Serper API key is required for web domain")
        if not config.jina_api_key:
            errors.append("Jina API key is required for web domain")
        if not config.openai_api_key:
            errors.append("OpenAI API key is required for web domain")
        if not config.chroma_api_key:
            errors.append("Chroma API key is required for web domain")
        if not config.chroma_database:
            errors.append("Chroma Database is required for web domain")
    
    elif config.domain == "sec":
        if not config.openai_api_key:
            errors.append("OpenAI API key is required for SEC domain")
        if not config.chroma_api_key:
            errors.append("Chroma API key is required for SEC domain")
        if not config.chroma_database:
            errors.append("Chroma Database is required for SEC domain")
    
    elif config.domain == "patents":
        if not config.openai_api_key:
            errors.append("OpenAI API key is required for patents domain")
        if not config.chroma_api_key:
            errors.append("Chroma API key is required for patents domain")
        if not config.chroma_database:
            errors.append("Chroma Database is required for patents domain")
    
    elif config.domain == "epstein":
        if not config.openai_api_key:
            errors.append("OpenAI API key is required for email domain")
        if not config.chroma_api_key:
            errors.append("Chroma API key is required for email domain")
        if not config.chroma_database:
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
        sys.executable,  # Use current Python executable
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


# Import sys for build_command
import sys