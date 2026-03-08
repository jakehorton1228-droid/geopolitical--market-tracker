"""Add prediction_markets table

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-03-08 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, Sequence[str], None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create prediction_markets table for Polymarket data."""
    op.create_table('prediction_markets',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('market_id', sa.String(length=20), nullable=False),
    sa.Column('question', sa.Text(), nullable=False),
    sa.Column('event_title', sa.String(length=255), nullable=True),
    sa.Column('yes_price', sa.Float(), nullable=False),
    sa.Column('volume_24h', sa.Float(), nullable=True),
    sa.Column('total_volume', sa.Float(), nullable=True),
    sa.Column('liquidity', sa.Float(), nullable=True),
    sa.Column('end_date', sa.String(length=30), nullable=True),
    sa.Column('snapshot_at', sa.DateTime(), nullable=False),
    sa.Column('snapshot_date', sa.Date(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('market_id', 'snapshot_date', name='uq_prediction_market_snapshot')
    )
    op.create_index(op.f('ix_prediction_markets_market_id'), 'prediction_markets', ['market_id'], unique=False)
    op.create_index(op.f('ix_prediction_markets_snapshot_at'), 'prediction_markets', ['snapshot_at'], unique=False)
    op.create_index(op.f('ix_prediction_markets_snapshot_date'), 'prediction_markets', ['snapshot_date'], unique=False)
    op.create_index('ix_prediction_markets_market_date', 'prediction_markets', ['market_id', 'snapshot_date'], unique=False)


def downgrade() -> None:
    """Drop prediction_markets table."""
    op.drop_index('ix_prediction_markets_market_date', table_name='prediction_markets')
    op.drop_index(op.f('ix_prediction_markets_snapshot_date'), table_name='prediction_markets')
    op.drop_index(op.f('ix_prediction_markets_snapshot_at'), table_name='prediction_markets')
    op.drop_index(op.f('ix_prediction_markets_market_id'), table_name='prediction_markets')
    op.drop_table('prediction_markets')
