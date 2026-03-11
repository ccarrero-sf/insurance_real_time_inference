"""
Refresh the Git Repository stage in Snowflake.

Runs ALTER GIT REPOSITORY ... FETCH to pull the latest commits from the
remote Git repository into the Snowflake Git Repository stage.

This replaces upload_code.py from the submit_directory version. Instead of
uploading files from the local filesystem, the code now lives in a Git
repository that is synced via this fetch command.

Can be run:
  - Standalone before deployment to ensure latest code is available.
  - Automatically as the first task (REFRESH_GIT) in the DAG before any
    pipeline execution.

Usage:
    python refresh_git_repo.py [--verify]

    --verify: After fetching, list the Git repo stage contents to verify.
"""

import argparse
import os

from snowflake.snowpark import Session


# Configuration
DB_NAME = os.getenv("PIPELINE_DB", "CC_INSURANCE_PIPELINE")
SCHEMA_NAME = os.getenv("PIPELINE_SCHEMA", "PIPELINE_STAGE")
GIT_REPO_NAME = os.getenv("GIT_REPO_NAME", "ML_PIPELINE_GIT_REPO")
GIT_BRANCH = os.getenv("GIT_BRANCH", "main")

FQN_REPO = f"{DB_NAME}.{SCHEMA_NAME}.{GIT_REPO_NAME}"
REPO_STAGE = f"@{FQN_REPO}"


def refresh_git_repo(session: Session, verify: bool = False) -> None:
    """Fetch the latest commits from the remote Git repository."""
    session.use_database(DB_NAME)
    session.use_schema(SCHEMA_NAME)

    print(f"Refreshing Git repository: {FQN_REPO}")
    session.sql(f"ALTER GIT REPOSITORY {FQN_REPO} FETCH").collect()
    print("Git repository refreshed successfully.")

    if verify:
        branch_path = f"{REPO_STAGE}/branches/{GIT_BRANCH}"
        print(f"\nRepository contents ({branch_path}):")
        files = session.sql(f"LIST {branch_path}").collect()
        for f in files:
            print(f"  {f['name']}  (size: {f['size']})")

        print(f"\nBranches:")
        branches = session.sql(f"SHOW GIT BRANCHES IN {FQN_REPO}").collect()
        for b in branches:
            print(f"  {b['name']} ({b['commit_hash'][:8]})")

    print("\nRefresh complete. The DAG will use the latest code from Git.")


def get_session() -> Session:
    """Create a Snowpark session from the default connection."""
    return Session.builder.config("connection_name", "keypair").create()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refresh the Git Repository stage in Snowflake"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="List repository contents after refresh to verify",
    )
    args = parser.parse_args()

    session = get_session()
    print(f"Connected as: {session.get_current_role()}")

    try:
        refresh_git_repo(session, verify=args.verify)
    finally:
        session.close()
