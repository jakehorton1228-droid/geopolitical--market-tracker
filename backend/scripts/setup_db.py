#!/usr/bin/env python3
"""
Database setup script.

This script helps set up the database for development. Run it after:
1. Starting PostgreSQL (docker start gmt-postgres)
2. Activating your virtual environment (source venv/bin/activate)

Usage:
    python scripts/setup_db.py          # Check connection and show status
    python scripts/setup_db.py --init   # Initialize tables (alternative to alembic)
    python scripts/setup_db.py --reset  # Drop and recreate all tables (DANGER!)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from src.db.connection import engine, init_db, drop_db, get_session
from src.db.models import Event, MarketData, AnalysisResult, EventMarketLink


def check_connection() -> bool:
    """Test database connection."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"✓ Connected to PostgreSQL")
            print(f"  Version: {version[:50]}...")
            return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


def show_table_counts():
    """Show row counts for all tables."""
    print("\nTable row counts:")
    with get_session() as session:
        tables = [
            ("events", Event),
            ("market_data", MarketData),
            ("analysis_results", AnalysisResult),
            ("event_market_links", EventMarketLink),
        ]
        for name, model in tables:
            count = session.query(model).count()
            print(f"  {name}: {count:,} rows")


def init_tables():
    """Create all tables using SQLAlchemy (not Alembic)."""
    print("Creating tables...")
    init_db()
    print("✓ Tables created")


def reset_database():
    """Drop and recreate all tables. WARNING: Deletes all data!"""
    confirm = input("This will DELETE ALL DATA. Type 'yes' to confirm: ")
    if confirm.lower() != 'yes':
        print("Aborted.")
        return

    print("Dropping all tables...")
    drop_db()
    print("Creating tables...")
    init_db()
    print("✓ Database reset complete")


def main():
    parser = argparse.ArgumentParser(description="Database setup utility")
    parser.add_argument("--init", action="store_true", help="Initialize tables")
    parser.add_argument("--reset", action="store_true", help="Reset database (DANGER!)")
    args = parser.parse_args()

    print("=" * 50)
    print("Geopolitical Market Tracker - Database Setup")
    print("=" * 50)
    print()

    if not check_connection():
        print("\nMake sure PostgreSQL is running:")
        print("  docker start gmt-postgres")
        sys.exit(1)

    if args.reset:
        reset_database()
    elif args.init:
        init_tables()

    show_table_counts()
    print("\n✓ Database is ready!")


if __name__ == "__main__":
    main()
