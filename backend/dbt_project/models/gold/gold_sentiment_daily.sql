{{
    config(
        materialized='incremental',
        unique_key=['published_date', 'source']
    )
}}

-- Gold Sentiment Daily: one row per (date, source) + an "all" aggregate.
-- Summarizes headline counts and sentiment by source and day.
-- The Sentiment sub-agent's primary data source.

with per_source as (
    select
        published_date,
        source,
        count(*) as headline_count,
        avg(sentiment_score) as avg_sentiment,
        sum(case when sentiment_label = 'positive' then 1 else 0 end) as positive_count,
        sum(case when sentiment_label = 'negative' then 1 else 0 end) as negative_count,
        sum(case when sentiment_label = 'neutral' then 1 else 0 end) as neutral_count,
        -- Most extreme headlines
        (array_agg(headline order by sentiment_score asc))[1] as most_negative_headline,
        (array_agg(headline order by sentiment_score desc))[1] as most_positive_headline

    from {{ ref('stg_headlines') }}
    {% if is_incremental() %}
    where published_date > (select max(published_date) from {{ this }})
    {% endif %}
    group by published_date, source
),

all_sources as (
    select
        published_date,
        'all' as source,
        count(*) as headline_count,
        avg(sentiment_score) as avg_sentiment,
        sum(case when sentiment_label = 'positive' then 1 else 0 end) as positive_count,
        sum(case when sentiment_label = 'negative' then 1 else 0 end) as negative_count,
        sum(case when sentiment_label = 'neutral' then 1 else 0 end) as neutral_count,
        (array_agg(headline order by sentiment_score asc))[1] as most_negative_headline,
        (array_agg(headline order by sentiment_score desc))[1] as most_positive_headline

    from {{ ref('stg_headlines') }}
    {% if is_incremental() %}
    where published_date > (select max(published_date) from {{ this }})
    {% endif %}
    group by published_date
)

select *, current_timestamp as computed_at from per_source
union all
select *, current_timestamp as computed_at from all_sources
