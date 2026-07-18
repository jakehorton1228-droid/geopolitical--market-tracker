-- Silver events: Bronze GDELT events → cleaned, classified, deduplicated.
--
-- Ported one-to-one from src/transforms/silver_events.py (output-identical):
--   * dedup on global_event_id — keep the best-covered row (most mentions/sources)
--   * CAMEO root code → event_group + human label  (seeds: event_group_map, cameo_label_map)
--   * FIPS 10-4 → ISO 3166-1 alpha-3 country code  (seed: fips_to_iso)
--   * is_significant = |goldstein| >= 5 AND num_mentions >= 1000
--
-- Runs on DuckDB, materializes into Postgres (see dbt_project.yml / profiles.yml).

with deduped as (
    select
        *,
        row_number() over (
            partition by global_event_id
            order by num_mentions desc nulls last,
                     num_sources  desc nulls last
        ) as _rn
    from {{ source('bronze', 'events') }}
    where global_event_id is not null
      and event_date      is not null
      and event_root_code is not null
)

select
    d.global_event_id                          as event_id,
    d.event_date                               as event_date,
    coalesce(f.iso, d.action_geo_country_code) as country_code,
    coalesce(g.event_group, 'other')           as event_group,
    d.event_root_code                          as event_root_code,
    coalesce(c.cameo_label, 'other')           as cameo_label,
    d.goldstein_scale                          as goldstein_scale,
    d.num_mentions                             as num_mentions,
    d.num_sources                              as num_sources,
    d.avg_tone                                 as avg_tone,
    d.actor1_name                              as actor1_name,
    d.actor1_country_code                      as actor1_country,
    d.actor2_name                              as actor2_name,
    d.actor2_country_code                      as actor2_country,
    d.action_geo_name                          as geo_name,
    d.action_geo_lat                           as geo_lat,
    d.action_geo_long                          as geo_long,
    (abs(d.goldstein_scale) >= 5.0 and d.num_mentions >= 1000) as is_significant
from deduped d
left join {{ ref('event_group_map') }} g
       on substr(lpad(d.event_root_code, 2, '0'), 1, 2) = g.code
left join {{ ref('cameo_label_map') }} c
       on substr(lpad(d.event_root_code, 2, '0'), 1, 2) = c.code
left join {{ ref('fips_to_iso') }} f
       on d.action_geo_country_code = f.fips
where d._rn = 1
