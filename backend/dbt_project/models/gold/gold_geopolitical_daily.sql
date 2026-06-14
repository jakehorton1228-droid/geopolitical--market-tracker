{{
    config(
        materialized='incremental',
        unique_key=['event_date', 'country_code']
    )
}}

-- Gold Geopolitical Daily: one row per (date, country).
-- Aggregates event counts, conflict/cooperation splits, Goldstein metrics.
-- The Geopolitical sub-agent's primary data source.

select
    event_date,
    country_code,
    count(*) as total_events,
    sum(case when event_group in ('material_conflict', 'violent_conflict') then 1 else 0 end) as conflict_events,
    sum(case when event_group in ('verbal_cooperation', 'material_cooperation') then 1 else 0 end) as cooperation_events,
    sum(case when event_group = 'violent_conflict' then 1 else 0 end) as violent_events,
    avg(goldstein_scale) as avg_goldstein,
    min(goldstein_scale) as min_goldstein,
    max(goldstein_scale) as max_goldstein,
    sum(num_mentions) as total_mentions,
    avg(avg_tone) as avg_tone,
    mode() within group (order by event_root_code) as top_event_code,
    current_timestamp as computed_at

from {{ ref('stg_events') }}

{% if is_incremental() %}
where event_date > (select max(event_date) from {{ this }})
{% endif %}

group by event_date, country_code
