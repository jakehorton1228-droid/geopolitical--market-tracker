"""Database connection and session management."""

from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.config.settings import DATABASE_URL
from src.db.models import Base


# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=5,  # Number of connections to keep open
    max_overflow=10,  # Additional connections allowed when pool is full
    pool_pre_ping=True,  # Verify connections before use (handles stale connections)
    echo=False,  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def init_db():
    """
    Create all database tables.

    This uses SQLAlchemy's create_all which is safe to call multiple times -
    it only creates tables that don't exist.

    For production, you should use Alembic migrations instead.
    """
    Base.metadata.create_all(bind=engine)


def drop_db():
    """
    Drop all database tables.

    WARNING: This deletes all data! Only use for testing/development.
    """
    Base.metadata.drop_all(bind=engine)


@contextmanager
def get_session() -> Session:
    """
    Context manager for database sessions.

    Usage:
        with get_session() as session:
            events = session.query(Event).all()
            session.add(new_event)
            # Commits automatically on success, rolls back on exception

    Yields:
        SQLAlchemy Session object
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """
    Dependency for FastAPI endpoints.

    Usage in FastAPI:
        @app.get("/events")
        def get_events(db: Session = Depends(get_db)):
            return db.query(Event).all()

    Yields:
        SQLAlchemy Session object
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
