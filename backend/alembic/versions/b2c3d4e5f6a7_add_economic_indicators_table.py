"""Add economic_indicators table

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-08 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create economic_indicators table for FRED data."""
    op.create_table('economic_indicators',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('series_id', sa.String(length=20), nullable=False),
    sa.Column('series_name', sa.String(length=100), nullable=False),
    sa.Column('date', sa.Date(), nullable=False),
    sa.Column('value', sa.Float(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('series_id', 'date', name='uq_economic_indicator_series_date')
    )
    op.create_index(op.f('ix_economic_indicators_series_id'), 'economic_indicators', ['series_id'], unique=False)
    op.create_index(op.f('ix_economic_indicators_date'), 'economic_indicators', ['date'], unique=False)
    op.create_index('ix_economic_indicators_series_date', 'economic_indicators', ['series_id', 'date'], unique=False)


def downgrade() -> None:
    """Drop economic_indicators table."""
    op.drop_index('ix_economic_indicators_series_date', table_name='economic_indicators')
    op.drop_index(op.f('ix_economic_indicators_date'), table_name='economic_indicators')
    op.drop_index(op.f('ix_economic_indicators_series_id'), table_name='economic_indicators')
    op.drop_table('economic_indicators')
