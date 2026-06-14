-- Risk level must be one of the three defined values.
-- Catches any edge case where the macro produces something unexpected.

select count(*) as failures
from {{ ref('gold_daily_summary') }}
where risk_level not in ('low', 'elevated', 'high')
having count(*) > 0
