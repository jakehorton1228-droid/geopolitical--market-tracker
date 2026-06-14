-- Ensure no Gold model has null dates — they're the partition key
-- for all downstream queries.

select count(*) as failures
from {{ ref('gold_daily_summary') }}
where summary_date is null
having count(*) > 0
