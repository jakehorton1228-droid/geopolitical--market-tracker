-- Mention counts should never be negative — a negative value
-- indicates a bug in the aggregation logic.

select count(*) as failures
from {{ ref('gold_geopolitical_daily') }}
where total_mentions < 0
having count(*) > 0
