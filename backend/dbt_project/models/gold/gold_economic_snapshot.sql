{{
    config(
        materialized='incremental',
        unique_key=['date', 'series_id']
    )
}}

-- Gold Economic Snapshot: one row per (date, series_id).
-- Enriches raw FRED observations with previous-value deltas.
-- Reads directly from Bronze (economic_indicators is simple enough to skip Silver).
-- The Economic sub-agent's primary data source.

with indicators as (
    select
        date,
        series_id,
        series_name,
        value,
        lag(value) over (
            partition by series_id
            order by date
        ) as previous_value
    from {{ source('bronze', 'economic_indicators') }}
    {% if is_incremental() %}
    where date > (select max(date) from {{ this }})
    {% endif %}
)

select
    date,
    series_id,
    series_name,
    value,
    previous_value,
    value - previous_value as change,
    case
        when previous_value is not null and previous_value != 0
        then (value - previous_value) / abs(previous_value) * 100
    end as change_pct,
    current_timestamp as computed_at

from indicators
