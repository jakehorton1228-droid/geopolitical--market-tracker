-- Staging passthrough over the silver_events model.
-- Exists so Gold models ref a stable staging name and dbt tracks lineage
-- silver_events → stg_events → gold_*. Ephemeral: inlined as a CTE, not
-- materialized. Now refs the dbt model (was: source('silver', 'silver_events')).

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
from {{ ref('silver_events') }}
