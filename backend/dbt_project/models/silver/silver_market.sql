-- Silver market: Bronze OHLCV → cleaned, with rolling return/volatility metrics.
--
-- Ported one-to-one from src/transforms/silver_market.py (output-identical):
--   * dedup on (symbol, date) — keep the highest-volume row
--   * rolling 5-day / 20-day returns via LAG
--   * rolling 20-day volatility (sample stddev of daily_return)
--   * volume z-score vs 20-day rolling mean/std
--
-- Window functions (LAG, rolling AVG/STDDEV over a ROWS frame) are standard SQL —
-- the same partition/order/frame semantics any analytical warehouse exposes.

with deduped as (
    select
        *,
        row_number() over (
            partition by symbol, date
            order by volume desc nulls last
        ) as _rn
    from {{ source('bronze', 'market_data') }}
    where symbol is not null
      and date   is not null
      and close  is not null
),

base as (
    select
        symbol,
        date,
        cast(open   as double) as open,
        cast(high   as double) as high,
        cast(low    as double) as low,
        cast(close  as double) as close,
        cast(volume as bigint) as volume
    from deduped
    where _rn = 1
),

-- Returns are computed HERE (not at ingestion) over the full per-symbol history,
-- so they're correct even when Bronze was loaded in incremental batches. NULL
-- prev-close (first row per symbol) or a zero prev-close yields NULL, matching
-- the old pandas pct_change / log-ratio behavior.
returns as (
    select
        *,
        (close - lag(close) over w) / nullif(lag(close) over w, 0) as daily_return,
        ln(close / nullif(lag(close) over w, 0))                   as log_return
    from base
    window w as (partition by symbol order by date)
),

windowed as (
    select
        *,
        lag(close, 5)  over w   as close_5d_ago,
        lag(close, 20) over w   as close_20d_ago,
        stddev_samp(daily_return) over w20 as volatility_20d,
        avg(volume)               over w20 as vol_mean_20d,
        stddev_samp(volume)       over w20 as vol_std_20d
    from returns
    window
        w   as (partition by symbol order by date),
        w20 as (partition by symbol order by date
                rows between 19 preceding and current row)
)

select
    symbol,
    date,
    open,
    high,
    low,
    close,
    volume,
    daily_return,
    log_return,
    case when close_5d_ago is not null and close_5d_ago <> 0
         then (close - close_5d_ago) / close_5d_ago end  as return_5d,
    case when close_20d_ago is not null and close_20d_ago <> 0
         then (close - close_20d_ago) / close_20d_ago end as return_20d,
    volatility_20d,
    case when vol_std_20d is not null and vol_std_20d > 0
         then (cast(volume as double) - vol_mean_20d) / vol_std_20d
         end                                              as volume_zscore,
    true as is_trading_day
from windowed
