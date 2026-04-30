from collections.abc import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables on startup. Replace with Alembic migrations for prod."""
    from models import (  # noqa: F401  (register models with Base)
        project,
        dataset,
        chat_message,
        training_run,
        benchmark_result,
        setting,
    )

    Base.metadata.create_all(bind=engine)
