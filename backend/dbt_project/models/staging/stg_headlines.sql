select
    headline_id,
    source,
    headline,
    url,
    published_date,
    sentiment_score,
    sentiment_label
from {{ source('silver', 'silver_headlines') }}
