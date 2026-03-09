"""Add pgvector extension and embedding columns to events and news_headlines

Revision ID: 9d23c73f6b52
Revises: c3d4e5f6a7b8
Create Date: 2026-03-09 12:29:23.903756

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '9d23c73f6b52'
down_revision: Union[str, Sequence[str], None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Enable pgvector and add 384-dim embedding columns."""
    # Enable the vector extension (idempotent — safe to re-run)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Add embedding columns (NULL until embedding pipeline processes each row)
    op.add_column('events', sa.Column('embedding', Vector(384), nullable=True))
    op.add_column('news_headlines', sa.Column('embedding', Vector(384), nullable=True))


def downgrade() -> None:
    """Remove embedding columns (extension left in place)."""
    op.drop_column('news_headlines', 'embedding')
    op.drop_column('events', 'embedding')
