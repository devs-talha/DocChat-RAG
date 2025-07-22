from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    # OpenAI configuration
    openai_api_key: str
    openai_model: str = "gpt-4.1-nano"
    openai_temperature: float = 0.0
    openai_max_tokens: int = 1000

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"

    # Text splitter configuration
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Retriever configuration
    retriever_search_type: str = "similarity"
    retriever_k: int = 4

    # File upload configuration
    max_file_size_mb: int = 10
    max_files: int = 20

    # Retrieval QA Chat Prompt
    retrieval_qa_chat_prompt: str = "langchain-ai/retrieval-qa-chat"

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )


# Create a global settings instance
settings = Settings()
