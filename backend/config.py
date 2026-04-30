from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env", env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "development"
    app_name: str = "vForge"
    app_port: int = 8000
    frontend_url: str = "http://localhost:3000"
    log_level: str = "info"

    database_url: str = (
        "postgresql+psycopg://vforge:vforge@localhost:5432/vforge"
    )

    ollama_base_url: str = "http://localhost:11434"
    together_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    custom_llm_base_url: str = ""
    custom_llm_api_key: str = ""
    custom_llm_model: str = ""

    default_generator_model: str = "llama3.1:8b"
    default_generator_provider: str = "ollama"
    default_base_model: str = "google/gemma-2-2b"

    hf_token: str = ""
    hf_default_org: str = ""
    gcs_bucket: str = ""
    google_application_credentials: str = ""

    gcp_project_id: str = ""
    gcp_region: str = "us-central1"
    tpu_name: str = ""
    tpu_zone: str = "us-central1-a"
    tpu_type: str = "v5e-4"


@lru_cache
def get_settings() -> Settings:
    return Settings()
