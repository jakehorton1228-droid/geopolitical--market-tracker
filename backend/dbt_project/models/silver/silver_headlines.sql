-- Silver headlines: Bronze RSS headlines → deduped, source-normalized, scored.
--
-- Ported one-to-one from src/transforms/silver_headlines.py (output-identical):
--   * dedup on url (natural key) — keep the latest by published_at, then id
--   * normalize source name via the source_normalization seed
--   * published_at → published_date
--   * keep only rows already scored by FinBERT (sentiment_score not null)

with deduped as (
    select
        *,
        row_number() over (
            partition by url
            order by published_at desc nulls last, id desc
        ) as _rn
    from {{ source('bronze', 'news_headlines') }}
)

select
    d.id                                    as headline_id,
    coalesce(sn.canonical, lower(d.source)) as source,
    d.headline                              as headline,
    d.url                                   as url,
    cast(d.published_at as date)            as published_date,
    d.sentiment_score                       as sentiment_score,
    d.sentiment_label                       as sentiment_label
from deduped d
left join {{ ref('source_normalization') }} sn
       on lower(d.source) = sn.raw
where d._rn = 1
  and d.headline        is not null
  and d.url             is not null
  and d.published_at    is not null
  and d.sentiment_score is not null
