"""Add news_headlines table

Revision ID: a1b2c3d4e5f6
Revises: fb13caaa6309
Create Date: 2026-03-05 22:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'fb13caaa6309'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create news_headlines table for RSS feed ingestion."""
    op.create_table('news_headlines',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('source', sa.String(length=20), nullable=False),
    sa.Column('headline', sa.Text(), nullable=False),
    sa.Column('url', sa.Text(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('published_at', sa.DateTime(), nullable=False),
    sa.Column('sentiment_score', sa.Float(), nullable=True),
    sa.Column('sentiment_label', sa.String(length=10), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('url', name='uq_news_headline_url')
    )
    op.create_index(op.f('ix_news_headlines_source'), 'news_headlines', ['source'], unique=False)
    op.create_index(op.f('ix_news_headlines_published_at'), 'news_headlines', ['published_at'], unique=False)
    op.create_index('ix_news_headlines_source_published', 'news_headlines', ['source', 'published_at'], unique=False)


def downgrade() -> None:
    """Drop news_headlines table."""
    op.drop_index('ix_news_headlines_source_published', table_name='news_headlines')
    op.drop_index(op.f('ix_news_headlines_published_at'), table_name='news_headlines')
    op.drop_index(op.f('ix_news_headlines_source'), table_name='news_headlines')
    op.drop_table('news_headlines')
