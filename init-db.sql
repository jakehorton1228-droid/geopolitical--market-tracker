-- Create the Prefect database (separate from the app database).
-- This script runs automatically on first container startup via
-- docker-entrypoint-initdb.d. It's a no-op if the database already exists.
SELECT 'CREATE DATABASE prefect'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'prefect')\gexec
