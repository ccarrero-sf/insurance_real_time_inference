"""
Run the setup_infrastructure.sql script against Snowflake.

Executes each SQL statement from setup_infrastructure.sql individually,
handling SET variables by substituting them inline since Snowpark sessions
don't persist session variables across separate sql() calls.

Usage:
    python run_setup.py
"""

import os
import sys

from snowflake.snowpark import Session


# Configuration - matches the SET variables in setup_infrastructure.sql
ROLE_NAME = os.getenv("SNOWFLAKE_ROLE", "SPCS_PSE_ROLE")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
COMPUTE_POOL = os.getenv("SNOWFLAKE_COMPUTE_POOL", "DEMO_POOL")
DB_NAME = os.getenv("PIPELINE_DB", "CC_INSURANCE_PIPELINE")


def run_setup(session: Session) -> None:
    """Execute all infrastructure setup statements."""
    statements = [
        # Database
        f"CREATE DATABASE IF NOT EXISTS {DB_NAME}",

        # Schemas
        f"CREATE SCHEMA IF NOT EXISTS {DB_NAME}.PIPELINE",
        f"CREATE SCHEMA IF NOT EXISTS {DB_NAME}.DATA",

        # Stages
        f"CREATE STAGE IF NOT EXISTS {DB_NAME}.PIPELINE.CODE_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')",
        f"CREATE STAGE IF NOT EXISTS {DB_NAME}.PIPELINE.JOB_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')",
        f"CREATE STAGE IF NOT EXISTS {DB_NAME}.PIPELINE.ARTIFACTS_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')",

        # Image Repository
        f"CREATE IMAGE REPOSITORY IF NOT EXISTS {DB_NAME}.PIPELINE.ML_IMAGE_REPO",

        # Grants - database and schemas
        f"GRANT USAGE ON DATABASE {DB_NAME} TO ROLE {ROLE_NAME}",
        f"GRANT ALL ON SCHEMA {DB_NAME}.PIPELINE TO ROLE {ROLE_NAME}",
        f"GRANT ALL ON SCHEMA {DB_NAME}.DATA TO ROLE {ROLE_NAME}",

        # Grants - stages
        f"GRANT READ, WRITE ON STAGE {DB_NAME}.PIPELINE.CODE_STAGE TO ROLE {ROLE_NAME}",
        f"GRANT READ, WRITE ON STAGE {DB_NAME}.PIPELINE.JOB_STAGE TO ROLE {ROLE_NAME}",
        f"GRANT READ, WRITE ON STAGE {DB_NAME}.PIPELINE.ARTIFACTS_STAGE TO ROLE {ROLE_NAME}",

        # Grants - warehouse
        f"GRANT USAGE ON WAREHOUSE {WAREHOUSE} TO ROLE {ROLE_NAME}",

        # Verify
        f"SHOW SCHEMAS IN DATABASE {DB_NAME}",
        f"SHOW STAGES IN SCHEMA {DB_NAME}.PIPELINE",
    ]

    print(f"Setting up infrastructure for {DB_NAME}")
    print(f"  Role: {ROLE_NAME}")
    print(f"  Warehouse: {WAREHOUSE}")
    print(f"  Compute Pool: {COMPUTE_POOL}")
    print()

    for stmt in statements:
        label = stmt.split("(")[0].strip() if "(" in stmt else stmt
        if len(label) > 80:
            label = label[:77] + "..."
        try:
            session.sql(stmt).collect()
            print(f"  OK: {label}")
        except Exception as e:
            print(f"  WARN: {label}")
            print(f"        {e}")

    print()
    print("Infrastructure setup complete.")


def get_session() -> Session:
    """Create a Snowpark session from the default connection."""
    return Session.builder.config("connection_name", "keypair").create()


if __name__ == "__main__":
    session = get_session()
    print(f"Connected as: {session.get_current_role()}")
    print()

    try:
        run_setup(session)
    finally:
        session.close()
