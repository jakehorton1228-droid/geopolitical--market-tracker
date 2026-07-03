"""Add Silver layer tables for medallion architecture

Revision ID: d4e5f6a7b8c9
Revises: fb13caaa6309
Create Date: 2026-06-14

Creates the Silver layer tables that sit between Bronze (raw ingestion)
and Gold (dbt models). DuckDB transforms populate these tables.

Tables:
- silver_events: Cleaned, classified GDELT events
- silver_market: Market data with rolling metrics
- silver_headlines: Headlines with normalized sentiment
- silver_event_market: Cross-domain event-to-market join
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd4e5f6a7b8c9'
down_revision: str = '9d23c73f6b52'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # silver_events — cleaned, classified GDELT events
    op.create_table(
        'silver_events',
        sa.Column('event_id', sa.String(50), nullable=False),
        sa.Column('event_date', sa.Date, nullable=False),
        sa.Column('country_code', sa.String(3)),
        sa.Column('event_group', sa.String(30), nullable=False),
        sa.Column('event_root_code', sa.String(2), nullable=False),
        sa.Column('cameo_label', sa.String(30)),
        sa.Column('goldstein_scale', sa.Float),
        sa.Column('num_mentions', sa.Integer),
        sa.Column('num_sources', sa.Integer),
        sa.Column('avg_tone', sa.Float),
        sa.Column('actor1_name', sa.String(255)),
        sa.Column('actor1_country', sa.String(3)),
        sa.Column('actor2_name', sa.String(255)),
        sa.Column('actor2_country', sa.String(3)),
        sa.Column('geo_name', sa.String(255)),
        sa.Column('geo_lat', sa.Float),
        sa.Column('geo_long', sa.Float),
        sa.Column('is_significant', sa.Boolean, default=False),
    )
    op.create_index('ix_silver_events_date_country', 'silver_events', ['event_date', 'country_code'])
    op.create_index('ix_silver_events_event_group', 'silver_events', ['event_group'])

    # silver_market — market data with rolling metrics
    op.create_table(
        'silver_market',
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('date', sa.Date, nullable=False),
        sa.Column('open', sa.Float),
        sa.Column('high', sa.Float),
        sa.Column('low', sa.Float),
        sa.Column('close', sa.Float, nullable=False),
        sa.Column('volume', sa.BigInteger),
        sa.Column('daily_return', sa.Float),
        sa.Column('log_return', sa.Float),
        sa.Column('return_5d', sa.Float),
        sa.Column('return_20d', sa.Float),
        sa.Column('volatility_20d', sa.Float),
        sa.Column('volume_zscore', sa.Float),
        sa.Column('is_trading_day', sa.Boolean, default=True),
    )
    op.create_index('ix_silver_market_symbol_date', 'silver_market', ['symbol', 'date'])

    # silver_headlines — headlines with normalized sentiment
    op.create_table(
        'silver_headlines',
        sa.Column('headline_id', sa.Integer, nullable=False),
        sa.Column('source', sa.String(20), nullable=False),
        sa.Column('headline', sa.Text, nullable=False),
        sa.Column('url', sa.Text, nullable=False),
        sa.Column('published_date', sa.Date, nullable=False),
        sa.Column('sentiment_score', sa.Float),
        sa.Column('sentiment_label', sa.String(10)),
    )
    op.create_index('ix_silver_headlines_date_source', 'silver_headlines', ['published_date', 'source'])

    # silver_event_market — cross-domain join
    op.create_table(
        'silver_event_market',
        sa.Column('event_date', sa.Date, nullable=False),
        sa.Column('country_code', sa.String(3), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('event_count', sa.Integer),
        sa.Column('avg_goldstein', sa.Float),
        sa.Column('min_goldstein', sa.Float),
        sa.Column('max_goldstein', sa.Float),
        sa.Column('total_mentions', sa.Integer),
        sa.Column('avg_tone', sa.Float),
        sa.Column('violent_count', sa.Integer),
        sa.Column('conflict_count', sa.Integer),
        sa.Column('cooperation_count', sa.Integer),
        sa.Column('dominant_event_group', sa.String(30)),
        sa.Column('close', sa.Float),
        sa.Column('daily_return', sa.Float),
        sa.Column('log_return', sa.Float),
        sa.Column('return_5d', sa.Float),
        sa.Column('volatility_20d', sa.Float),
    )
    op.create_index('ix_silver_event_market_date', 'silver_event_market', ['event_date'])
    op.create_index('ix_silver_event_market_symbol', 'silver_event_market', ['symbol'])
    op.create_index(
        'ix_silver_event_market_date_symbol',
        'silver_event_market',
        ['event_date', 'country_code', 'symbol'],
    )


def downgrade() -> None:
    op.drop_table('silver_event_market')
    op.drop_table('silver_headlines')
    op.drop_table('silver_market')
    op.drop_table('silver_events')
