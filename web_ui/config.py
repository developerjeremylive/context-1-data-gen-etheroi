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
    pollination_model: str = ""  # e.g. "openai/gpt-oss-20b"


def validate_api_keys_from_env(env_vars: dict, domain: str, use_env_file: bool = False) -> List[str]:
    """
    Validate required API keys from an env dict.
    
    When use_env_file=True and a pollination_model is set, we skip
    Anthropic and OpenAI key validation (using Pollination AI instead).
    """
    errors = []
    
    serper_key = env_vars.get("SERPER_API_KEY", "")
    jina_key = env_vars.get("JINA_API_KEY", "")
    chroma_key = env_vars.get("CHROMA_API_KEY", "")
    chroma_db = env_vars.get("CHROMA_DATABASE", "")
    
    # Using Pollination means no Anthropic/OpenAI needed for LLM
    using_pollination = use_env_file and env_vars.get("OPENAI_API_BASE", "").startswith("https://gen.pollinations.ai")
    
    # Required for all domains when NOT using Pollination
    if not using_pollination:
        anthropic_key = env_vars.get("ANTHROPIC_API_KEY", "")
        if not anthropic_key:
            errors.append("Anthropic API key is required (unless using Pollination AI)")
    
    # Domain-specific (web domain needs Serper and Jina regardless of LLM provider)
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
        if not chroma_key:
            errors.append("Chroma API key is required")
        if not chroma_db:
            errors.append("Chroma Database is required")
    
    elif domain == "patents":
        if not chroma_key:
            errors.append("Chroma API key is required")
        if not chroma_db:
            errors.append("Chroma Database is required")
    
    elif domain == "epstein":
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