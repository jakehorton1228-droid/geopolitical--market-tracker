{{
    config(
        materialized='incremental',
        unique_key=['date', 'symbol']
    )
}}

-- Gold Market Daily: one row per (date, symbol).
-- Enriches Silver market data with anomaly flags from analysis_results.
-- The Market sub-agent's primary data source.

with market as (
    select
        symbol,
        date,
        close,
        daily_return,
        log_return,
        return_5d,
        return_20d,
        volatility_20d,
        volume_zscore
    from {{ ref('stg_market') }}
    {% if is_incremental() %}
    where date > (select max(date) from {{ this }})
    {% endif %}
),

anomalies as (
    select distinct
        symbol,
        -- Extract date from event window (use event_window_start as the anomaly date)
        event_window_start as anomaly_date,
        true as is_anomaly,
        anomaly_score
    from {{ source('bronze', 'analysis_results') }}
    where is_anomaly = true
)

select
    m.date,
    m.symbol,
    m.close,
    m.daily_return,
    m.log_return,
    m.return_5d,
    m.return_20d,
    m.volatility_20d,
    m.volume_zscore,
    coalesce(a.is_anomaly, false) as is_anomaly,
    a.anomaly_score,
    current_timestamp as computed_at

from market m
left join anomalies a
    on m.symbol = a.symbol
    and m.date = a.anomaly_date
