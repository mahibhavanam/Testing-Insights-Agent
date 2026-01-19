import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./local.db")
    long_term_db_path: str = os.getenv("LONG_TERM_DB_PATH", "./long_term_memory.sqlite")
    short_term_checkpoint_path: str = os.getenv("SHORT_TERM_CHECKPOINT_PATH", "./short_term_checkpoints.sqlite")

def get_settings() -> Settings:
    s = Settings()
    if not s.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    if not s.database_url:
        raise RuntimeError("Missing DATABASE_URL in environment.")
    return s
