select
    headline_id,
    source,
    headline,
    url,
    published_date,
    sentiment_score,
    sentiment_label
from {{ ref('silver_headlines') }}
