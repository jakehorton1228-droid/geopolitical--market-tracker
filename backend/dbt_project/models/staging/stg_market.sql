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
    return_5d,
    return_20d,
    volatility_20d,
    volume_zscore,
    is_trading_day
from {{ source('silver', 'silver_market') }}
