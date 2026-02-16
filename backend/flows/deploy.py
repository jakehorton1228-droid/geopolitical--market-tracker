"""
Register Prefect deployments with cron schedules.

Run this script to create the daily pipeline deployment:
    python -m flows.deploy

The deployment will appear in the Prefect UI and the worker
will pick it up according to the cron schedule.
"""

from flows.daily_pipeline import daily_pipeline


if __name__ == "__main__":
    daily_pipeline.serve(
        name="daily-pipeline-deployment",
        cron="0 6 * * *",  # 6:00 AM UTC daily
        tags=["daily", "production"],
        description="Daily ingestion (GDELT + Yahoo Finance) followed by analysis (correlations + patterns)",
    )
