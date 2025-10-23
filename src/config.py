""" Configuration and Settings"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""

    # Langsmith config
    langsmith_tracing: bool = Field(default=True, alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field(alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="rss-qa-chatbot", alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com",
        alias="LANGSMITH_ENDPOINT")
    
    # Google Gemini API
    google_api_key: str = Field(alias="GOOGLE_API_KEY")

    # OpenRouter API for GPT and LLama model
    openrouter_api_key: str = Field(alias="OPENROUTER_API_KEY")

    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Application settings
    default_response_model: str = Field(default="gemini", alias="DEFAULT_RESPONSE_MODEL")
    message_summary_threshold: int = Field(default=10, alias="MESSAGE_SUMMARY_THRESHOLD")
    max_blog_titles: int = Field(default=5, alias="MAX_BLOG_TITLES")

    # Memory Settings
    sqlite_file_path: str = Field(default="langgraph.sqlite", alias="SQLITE_FILE_PATH")
    
    # Vector Store Settings
    blog_collection_name: str = "blog_content"
    user_memory_collection_name: str = "user_memory"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_size: int = 384

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

os.environ["LANGSMITH_TRACING"] = str(settings.langsmith_tracing).lower()
os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
