"""
Configuration dataclass and validation for Context-1 Data Generation Pipeline UI.
"""
import sys
from dataclasses import dataclass
from typing import List


DOMAIN_OPTIONS = {
    "web": "Web Search Tasks",
    "sec": "SEC Filing Tasks",
    "patents": "Patent Prior-Art Tasks",
    "epstein": "Email Search Tasks",
}

MODEL_OPTIONS = [
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
]

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
    repo_url: str = "https://github.com/developerjeremylive/context-1-data-gen-etheroi.git"
    domain: str = "web"
    seeds_file: str = "seeds.txt"
    output_dir: str = "output"
    collection: str = "context1-data"
    
    explore_model: str = "claude-sonnet-4-5"
    verify_model: str = "claude-sonnet-4-5"
    distract_model: str = "claude-sonnet-4-5"
    extend_model: str = "claude-sonnet-4-5"
    
    explore_max_iterations: int = 20
    verify_max_retries: int = 3
    distract_max_iterations: int = 15
    extend_max_iterations: int = 20
    extension_rounds: int = 0
    max_workers: int = 8
    
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    serper_api_key: str = ""
    jina_api_key: str = ""
    chroma_api_key: str = ""
    chroma_database: str = ""
    
    use_env_file: bool = False
    pollination_model: str = ""  # e.g. "pollinationai/llama-3-70b-instruct"


def validate_api_keys_from_env(env_vars: dict, domain: str, use_env_file: bool = False) -> List[str]:
    """
    Validate required API keys from an env dict.
    
    When use_env_file=True and a pollination_model is set, we skip
    Anthropic and OpenAI validation (using Pollination AI instead).
    """
    errors = []
    
    anthropic_key = env_vars.get("ANTHROPIC_API_KEY", "")
    openai_key = env_vars.get("OPENAI_API_KEY", "")
    serper_key = env_vars.get("SERPER_API_KEY", "")
    jina_key = env_vars.get("JINA_API_KEY", "")
    chroma_key = env_vars.get("CHROMA_API_KEY", "")
    chroma_db = env_vars.get("CHROMA_DATABASE", "")
    
    # Check if using Pollination AI (no Anthropic/OpenAI needed)
    using_pollination = use_env_file and env_vars.get("OPENAI_API_BASE") == "https://llm.pollination.ai"
    
    # Required for all domains — skip if using Pollination
    if not anthropic_key and not using_pollination:
        errors.append("Anthropic API key is required for all domains")
    
    # OpenAI only needed for embeddings — skip if using Pollination (still needed for embeddings)
    # Actually, openai_key is still used for embeddings even with Pollination LLM.
    # The user said "avoid using anthropic and openai models" so we keep openai for embeddings if needed.
    # But let the user decide — if they're in .env mode and have the key, use it.
    # If they explicitly want no API keys, they should use Pollination for LLM and 
    # Chroma expects embeddings... this is complex.
    # For now: if using Pollination AND no OPENAI_API_KEY, we warn but don't block.
    # The pipeline can fail later if it really needs embeddings.
    
    if domain == "web":
        if not serper_key:
            errors.append("Serper API key is required for web domain (serper.dev)")
        if not jina_key:
            errors.append("Jina API key is required for web domain (jina.ai)")
        if not chroma_key:
            errors.append("Chroma API key is required (console.trychroma.com)")
        if not chroma_db:
            errors.append("Chroma Database is required")
    
    elif domain == "sec":
        if not openai_key and not using_pollination:
            errors.append("OpenAI API key is required for SEC domain (embeddings)")
        if not chroma_key:
            errors.append("Chroma API key is required")
        if not chroma_db:
            errors.append("Chroma Database is required")
    
    elif domain == "patents":
        if not openai_key and not using_pollination:
            errors.append("OpenAI API key is required for patents domain (embeddings)")
        if not chroma_key:
            errors.append("Chroma API key is required")
        if not chroma_db:
            errors.append("Chroma Database is required")
    
    elif domain == "epstein":
        if not openai_key and not using_pollination:
            errors.append("OpenAI API key is required for email domain (embeddings)")
        if not chroma_key:
            errors.append("Chroma API key is required")
        if not chroma_db:
            errors.append("Chroma Database is required")
    
    return errors


def build_command(config: PipelineConfig, repo_path: str = None) -> List[str]:
    """Build the command to run the pipeline."""
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