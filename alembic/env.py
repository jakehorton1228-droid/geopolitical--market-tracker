"""
Alembic migration environment configuration.

WHAT THIS FILE DOES:
--------------------
This is the "brain" of Alembic. When you run migration commands like:
    alembic upgrade head
    alembic revision --autogenerate -m "add new column"

Alembic reads this file to know:
1. How to connect to your database (DATABASE_URL)
2. What your models look like (target_metadata) - so it can detect changes
3. How to run migrations (online vs offline mode)

KEY CONCEPTS:
-------------
- target_metadata: SQLAlchemy's representation of your models. Alembic compares
  this to your actual database to auto-generate migrations.

- Online mode: Connects directly to the database to run migrations (normal use)

- Offline mode: Generates SQL scripts without connecting (useful for review
  or when you can't connect directly to production)
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# =============================================================================
# STEP 1: Import our project's configuration and models
# =============================================================================

# Import our database URL from settings (keeps credentials in one place)
from src.config.settings import DATABASE_URL

# Import our models' metadata - this tells Alembic what tables should exist
# Alembic will compare this metadata against the actual DB to find differences
from src.db.models import Base

# =============================================================================
# STEP 2: Configure Alembic
# =============================================================================

# This is the Alembic Config object - reads from alembic.ini
config = context.config

# Override the sqlalchemy.url from alembic.ini with our settings
# This way we don't hardcode credentials in alembic.ini
config.set_main_option("sqlalchemy.url", DATABASE_URL)

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# THIS IS THE KEY LINE:
# Point Alembic to our models' metadata so it can auto-detect schema changes
target_metadata = Base.metadata

# =============================================================================
# STEP 3: Migration runners
# =============================================================================


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode - generates SQL without DB connection.

    Useful when:
    - You want to review the SQL before running it
    - You need to hand off SQL scripts to a DBA
    - You can't connect directly to production from your machine

    Usage: alembic upgrade head --sql > migration.sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode - connects directly to database.

    This is the normal way to run migrations during development.
    Creates a connection, runs all pending migrations, then closes.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


# Determine which mode to run based on how Alembic was invoked
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
