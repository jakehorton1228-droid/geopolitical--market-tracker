-- Staging view over silver_events.
-- Thin passthrough — Silver transform already handled cleaning.
-- Exists so Gold models use {{ ref('stg_events') }} instead of {{ source() }}
-- which gives dbt proper lineage tracking.

select
    event_id,
    event_date,
    country_code,
    event_group,
    event_root_code,
    cameo_label,
    goldstein_scale,
    num_mentions,
    num_sources,
    avg_tone,
    actor1_name,
    actor1_country,
    actor2_name,
    actor2_country,
    geo_name,
    geo_lat,
    geo_long,
    is_significant
from {{ source('silver', 'silver_events') }}
