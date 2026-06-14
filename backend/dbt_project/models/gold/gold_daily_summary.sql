{{
    config(
        materialized='incremental',
        unique_key='summary_date'
    )
}}

-- Gold Daily Summary: one row per date.
-- Cross-domain view combining geopolitical, market, and sentiment signals.
-- Computes a composite risk_level for the day.
-- The Supervisor agent's overview table.

with geo as (
    select
        event_date,
        sum(total_events) as total_global_events,
        sum(conflict_events) as total_conflict_events,
        sum(violent_events) as total_violent_events,
        -- Top conflict country = highest conflict event count that day
        (array_agg(country_code order by conflict_events desc))[1] as top_conflict_country
    from {{ ref('gold_geopolitical_daily') }}
    group by event_date
),

market as (
    select
        date,
        avg(daily_return) as avg_market_return,
        max(case when symbol = '^VIX' then close end) as vix_close,
        sum(case when is_anomaly then 1 else 0 end) as anomaly_count
    from {{ ref('gold_market_daily') }}
    group by date
),

sentiment as (
    select
        published_date,
        avg_sentiment as avg_news_sentiment,
        headline_count
    from {{ ref('gold_sentiment_daily') }}
    where source = 'all'
)

select
    coalesce(g.event_date, m.date, s.published_date) as summary_date,
    coalesce(g.total_global_events, 0) as total_global_events,
    coalesce(g.total_conflict_events, 0) as total_conflict_events,
    coalesce(g.total_violent_events, 0) as total_violent_events,
    g.top_conflict_country,
    m.avg_market_return,
    m.vix_close,
    coalesce(m.anomaly_count, 0) as market_anomaly_count,
    s.avg_news_sentiment,
    coalesce(s.headline_count, 0) as headline_count,
    {{ risk_level(
        'coalesce(g.total_conflict_events, 0)',
        'coalesce(s.avg_news_sentiment, 0)',
        'coalesce(m.vix_close, 0)'
    ) }} as risk_level,
    current_timestamp as computed_at

from geo g
full outer join market m on g.event_date = m.date
full outer join sentiment s on coalesce(g.event_date, m.date) = s.published_date

{% if is_incremental() %}
where coalesce(g.event_date, m.date, s.published_date) > (select max(summary_date) from {{ this }})
{% endif %}
