{% macro risk_level(conflict_events, avg_sentiment, vix_close) %}
-- Composite risk scoring: combines conflict intensity, news sentiment, and VIX.
-- Returns 'low', 'elevated', or 'high'.
case
    when {{ conflict_events }} >= 50 and {{ avg_sentiment }} < -0.2 then 'high'
    when {{ conflict_events }} >= 50 or {{ avg_sentiment }} < -0.2 then 'elevated'
    when {{ vix_close }} > 30 then 'elevated'
    when {{ conflict_events }} >= 20 and {{ vix_close }} > 25 then 'elevated'
    else 'low'
end
{% endmacro %}
