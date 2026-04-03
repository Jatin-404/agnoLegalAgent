from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="LEGAL_AGENT_",
    )

    app_name: str = "Legal Agent API"
    ollama_host: str = "http://localhost:11434"
    max_text_chars: int = 24000
    ollama_timeout_seconds: int = 180
    ollama_keep_alive: str = "30m"
    ollama_num_ctx: int = 4096
    structured_retries: int = 1
    use_native_output_schema: bool = False
    extraction_num_predict: int = 360
    enable_classifier_agent: bool = True
    classifier_model_id: str = "qwen2.5:7b"
    classifier_num_predict: int = 96
    classifier_agent_call_timeout_seconds: int = 12
    classifier_confidence_threshold: float = 0.84
    v2_chunk_concurrency: int = 3
    v2_chunk_size: int = 6500
    v2_max_chunks: int = 12
    v2_max_model_chunks: int = 4
    v2_model_id: str = "qwen2.5:7b"
    v2_merge_model_id: str = "qwen2.5:7b"
    v2_financial_model_id: str = "qwen2.5:7b"
    v2_exhibit_model_id: str = "qwen2.5:7b"
    v2_relationship_model_id: str = "qwen2.5:7b"
    v2_taxonomy_model_id: str = "qwen2.5:7b"
    v2_agent_call_timeout_seconds: int = 45
    v2_section_num_predict: int = 220
    v2_merge_num_predict: int = 220
    v2_financial_num_predict: int = 180
    v2_exhibit_num_predict: int = 160
    v2_relationship_num_predict: int = 180
    v2_taxonomy_num_predict: int = 48
    v2_enable_section_agent: bool = True
    v2_enable_merge_agent: bool = False
    v2_enable_financial_agent: bool = False
    v2_enable_exhibit_agent: bool = False
    v2_enable_relationship_agent: bool = False
    v2_enable_taxonomy_agent: bool = False
    langsmith_enabled: bool = False
    langsmith_api_key: str | None = None
    langsmith_api_url: str | None = None
    langsmith_project: str = "legal-agent"
    langsmith_workspace_id: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
