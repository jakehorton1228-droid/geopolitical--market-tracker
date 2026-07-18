-- Silver event-market: the cross-domain linking table.
--
-- Ported one-to-one from src/transforms/silver_event_market.py (output-identical).
-- Joins geopolitical events to market reactions via the country_asset_map seed —
-- "when something happens in Russia, check Ruble, Gas, Oil." One row per
-- (event_date, country, symbol): what happened + how that symbol moved that day.
--
-- Depends on the silver_events and silver_market models (dbt orders it after both).

with events_agg as (
    select
        event_date,
        country_code,
        count(*)              as event_count,
        avg(goldstein_scale)  as avg_goldstein,
        min(goldstein_scale)  as min_goldstein,
        max(goldstein_scale)  as max_goldstein,
        sum(num_mentions)     as total_mentions,
        avg(avg_tone)         as avg_tone,
        sum(case when event_group = 'violent_conflict'
                 then 1 else 0 end) as violent_count,
        sum(case when event_group in ('material_conflict', 'violent_conflict')
                 then 1 else 0 end) as conflict_count,
        sum(case when event_group in ('verbal_cooperation', 'material_cooperation')
                 then 1 else 0 end) as cooperation_count
    from {{ ref('silver_events') }}
    where country_code is not null
    group by event_date, country_code
),

-- Dominant event group = most frequent per (date, country). Ties are broken
-- DETERMINISTICALLY by severity (most impactful group wins) so the column is
-- stable across runs and engines.
grp_counts as (
    select event_date, country_code, event_group, count(*) as gc
    from {{ ref('silver_events') }}
    where country_code is not null
    group by event_date, country_code, event_group
),

dominant as (
    select event_date, country_code, event_group as dominant_event_group
    from (
        select
            event_date,
            country_code,
            event_group,
            row_number() over (
                partition by event_date, country_code
                order by gc desc,
                         case event_group
                             when 'violent_conflict'     then 0
                             when 'material_conflict'    then 1
                             when 'verbal_conflict'      then 2
                             when 'material_cooperation' then 3
                             when 'verbal_cooperation'   then 4
                             else 5
                         end
            ) as rn
        from grp_counts
    )
    where rn = 1
)

select
    ea.event_date          as event_date,
    ea.country_code        as country_code,
    ca.symbol              as symbol,
    ea.event_count         as event_count,
    ea.avg_goldstein       as avg_goldstein,
    ea.min_goldstein       as min_goldstein,
    ea.max_goldstein       as max_goldstein,
    ea.total_mentions      as total_mentions,
    ea.avg_tone            as avg_tone,
    ea.violent_count       as violent_count,
    ea.conflict_count      as conflict_count,
    ea.cooperation_count   as cooperation_count,
    d.dominant_event_group as dominant_event_group,
    m.close                as close,
    m.daily_return         as daily_return,
    m.log_return           as log_return,
    m.return_5d            as return_5d,
    m.volatility_20d       as volatility_20d
from events_agg ea
join dominant d           on ea.event_date = d.event_date
                         and ea.country_code = d.country_code
join {{ ref('country_asset_map') }} ca on ea.country_code = ca.country_code
join {{ ref('silver_market') }} m       on m.symbol = ca.symbol
                                       and m.date   = ea.event_date
