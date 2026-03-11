"""
Run the infrastructure setup for the submit_from_stage pipeline version.

Creates all Snowflake objects needed: database, schemas, internal stages,
Git integration (API integration, Git repository), and grants.

The Git integration is read-only (Snowflake only fetches from the remote
repository). No credentials/secrets are needed — commits are always pushed
from the local development environment, not from Snowflake.

Update GIT_REPO_ORIGIN and GIT_ALLOWED_PREFIXES to match your repository.

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
SCHEMA_NAME = "PIPELINE_STAGE"

# Git configuration - CHANGE THESE
GIT_REPO_ORIGIN = os.getenv("GIT_REPO_ORIGIN", "https://github.com/ccarrero-sf/insurance_real_time_inference.git")
GIT_ALLOWED_PREFIXES = os.getenv("GIT_ALLOWED_PREFIXES", "https://github.com/ccarrero-sf")
GIT_API_INTEGRATION_NAME = os.getenv("GIT_API_INTEGRATION_NAME", "ML_PIPELINE_GIT_API_INTEGRATION")
GIT_REPO_NAME = os.getenv("GIT_REPO_NAME", "ML_PIPELINE_GIT_REPO")


def run_setup(session: Session) -> None:
    """Execute all infrastructure setup statements."""
    fqn_schema = f"{DB_NAME}.{SCHEMA_NAME}"
    fqn_repo = f"{fqn_schema}.{GIT_REPO_NAME}"

    statements = [
        # Database
        f"CREATE DATABASE IF NOT EXISTS {DB_NAME}",

        # Schemas
        f"CREATE SCHEMA IF NOT EXISTS {fqn_schema}",
        f"CREATE SCHEMA IF NOT EXISTS {DB_NAME}.DATA",

        # Internal stages
        f"CREATE STAGE IF NOT EXISTS {fqn_schema}.CODE_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')",
        f"CREATE STAGE IF NOT EXISTS {fqn_schema}.JOB_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')",
        f"CREATE STAGE IF NOT EXISTS {fqn_schema}.DAG_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')",
        f"CREATE STAGE IF NOT EXISTS {fqn_schema}.ARTIFACTS_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')",

        # Image Repository
        f"CREATE IMAGE REPOSITORY IF NOT EXISTS {fqn_schema}.ML_IMAGE_REPO",

        # Git API Integration (no credentials needed — public/read-only repo)
        f"CREATE API INTEGRATION IF NOT EXISTS {GIT_API_INTEGRATION_NAME} API_PROVIDER = git_https_api API_ALLOWED_PREFIXES = ('{GIT_ALLOWED_PREFIXES}') ENABLED = TRUE",

        # Git Repository (no GIT_CREDENTIALS — read-only access)
        f"CREATE GIT REPOSITORY IF NOT EXISTS {fqn_repo} API_INTEGRATION = {GIT_API_INTEGRATION_NAME} ORIGIN = '{GIT_REPO_ORIGIN}'",

        # Initial fetch
        f"ALTER GIT REPOSITORY {fqn_repo} FETCH",

        # Grants - database and schemas
        f"GRANT USAGE ON DATABASE {DB_NAME} TO ROLE {ROLE_NAME}",
        f"GRANT ALL ON SCHEMA {fqn_schema} TO ROLE {ROLE_NAME}",
        f"GRANT ALL ON SCHEMA {DB_NAME}.DATA TO ROLE {ROLE_NAME}",

        # Grants - stages
        f"GRANT READ, WRITE ON STAGE {fqn_schema}.CODE_STAGE TO ROLE {ROLE_NAME}",
        f"GRANT READ, WRITE ON STAGE {fqn_schema}.JOB_STAGE TO ROLE {ROLE_NAME}",
        f"GRANT READ, WRITE ON STAGE {fqn_schema}.DAG_STAGE TO ROLE {ROLE_NAME}",
        f"GRANT READ, WRITE ON STAGE {fqn_schema}.ARTIFACTS_STAGE TO ROLE {ROLE_NAME}",

        # Grants - Git repository
        f"GRANT READ ON GIT REPOSITORY {fqn_repo} TO ROLE {ROLE_NAME}",

        # Grants - warehouse
        f"GRANT USAGE ON WAREHOUSE {WAREHOUSE} TO ROLE {ROLE_NAME}",

        # Verify
        f"SHOW SCHEMAS IN DATABASE {DB_NAME}",
        f"SHOW STAGES IN SCHEMA {fqn_schema}",
        f"SHOW GIT REPOSITORIES IN SCHEMA {fqn_schema}",
    ]

    print(f"Setting up infrastructure for {DB_NAME}")
    print(f"  Role: {ROLE_NAME}")
    print(f"  Warehouse: {WAREHOUSE}")
    print(f"  Compute Pool: {COMPUTE_POOL}")
    print(f"  Git Repo: {GIT_REPO_ORIGIN}")
    print(f"  Git Repo Name: {GIT_REPO_NAME}")
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
