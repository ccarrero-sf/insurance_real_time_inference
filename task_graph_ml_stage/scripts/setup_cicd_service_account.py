"""
Set up a CI/CD service account for GitHub Actions → Snowflake DAG deployment.

This script must be run by an ACCOUNTADMIN (or a role with CREATE USER,
CREATE ROLE, CREATE NETWORK POLICY privileges). It creates:

  1. A dedicated CICD_DEPLOY_RL role with minimum required permissions.
  2. A CICD_DEPLOY_USER service account (key-pair auth only, no password).
  3. A network policy that allows GitHub Actions runner IPs.
  4. Grants so the service account can deploy the ML pipeline DAG.

Prerequisites:
  1. Generate a key pair for the service account:

       openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out cicd_rsa_key.p8 -nocrypt
       openssl rsa -in cicd_rsa_key.p8 -pubout -out cicd_rsa_key.pub

  2. Extract the public key (single line, no BEGIN/END headers):

       grep -v "BEGIN\|END" cicd_rsa_key.pub | tr -d '\\n'

  3. Set the RSA_PUBLIC_KEY environment variable before running:

       export RSA_PUBLIC_KEY="MIIBIjANBgkq..."

  4. Configure these GitHub Secrets in your repository
     (Settings > Secrets and variables > Actions):

       SNOWFLAKE_ACCOUNT   = your Snowflake account identifier
       SNOWFLAKE_USER      = CICD_DEPLOY_USER
       SNOWFLAKE_PRIVATE_KEY = content of cicd_rsa_key.p8

  5. Optionally set these GitHub Variables:

       SNOWFLAKE_ROLE      = CICD_DEPLOY_RL  (default in workflow)
       SNOWFLAKE_WAREHOUSE = COMPUTE_WH      (default in workflow)

Usage:
    # Must connect as ACCOUNTADMIN
    export RSA_PUBLIC_KEY="MIIBIjANBgkq..."
    python setup_cicd_service_account.py

Environment variables:
    RSA_PUBLIC_KEY          (required) Public key for the service account
    SNOWFLAKE_WAREHOUSE     (optional, default: COMPUTE_WH)
    PIPELINE_DB             (optional, default: CC_INSURANCE_PIPELINE)
    SNOWFLAKE_COMPUTE_POOL  (optional, default: DEMO_POOL)
"""

import os
import sys

from snowflake.snowpark import Session


# Configuration
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
DB_NAME = os.getenv("PIPELINE_DB", "CC_INSURANCE_PIPELINE")
COMPUTE_POOL = os.getenv("SNOWFLAKE_COMPUTE_POOL", "DEMO_POOL")
SCHEMA_NAME = "PIPELINE_STAGE"
GIT_REPO_NAME = os.getenv("GIT_REPO_NAME", "ML_PIPELINE_GIT_REPO")

# CI/CD role and user names
CICD_ROLE = "CICD_DEPLOY_RL"
CICD_USER = "CICD_DEPLOY_USER"
NETWORK_POLICY = "CICD_NETWORK_POLICY"
NETWORK_RULE = "GITHUB_ACTIONS_NETWORK_RULE"

# GitHub Actions runner IP ranges.
# GitHub runners use IPs from broad Azure ranges that change frequently and
# sometimes fall outside the published list at https://api.github.com/meta.
# We allow all IPs (0.0.0.0/0) for the service account since it uses key-pair
# auth only (no password) and is a TYPE=SERVICE account with no interactive login.
GITHUB_ACTIONS_IP_RANGES = [
    "0.0.0.0/0",
]


def setup_cicd_service_account(session: Session, rsa_public_key: str) -> None:
    """Create the CI/CD service account, role, network policy, and grants."""
    fqn_schema = f"{DB_NAME}.{SCHEMA_NAME}"
    fqn_repo = f"{fqn_schema}.{GIT_REPO_NAME}"
    ip_list = ", ".join(f"'{ip}'" for ip in GITHUB_ACTIONS_IP_RANGES)

    statements = [
        # -- Role --
        (
            f"CREATE ROLE IF NOT EXISTS {CICD_ROLE} "
            f"COMMENT = 'Role for GitHub Actions CI/CD pipeline deployments'",
            "Create CI/CD role",
        ),

        # -- User (key-pair auth only) --
        (
            f"CREATE USER IF NOT EXISTS {CICD_USER} "
            f"DEFAULT_ROLE = {CICD_ROLE} "
            f"DEFAULT_WAREHOUSE = {WAREHOUSE} "
            f"TYPE = SERVICE "
            f"COMMENT = 'Service account for GitHub Actions DAG deployment'",
            "Create service account user",
        ),

        # -- Assign role to user --
        (
            f"GRANT ROLE {CICD_ROLE} TO USER {CICD_USER}",
            "Grant role to user",
        ),

        # -- Set public key --
        (
            f"ALTER USER {CICD_USER} SET RSA_PUBLIC_KEY = '{rsa_public_key}'",
            "Set RSA public key on user",
        ),

        # -- Network rule for GitHub Actions --
        (
            f"CREATE OR REPLACE NETWORK RULE {NETWORK_RULE} "
            f"MODE = INGRESS TYPE = IPV4 VALUE_LIST = ({ip_list})",
            "Create GitHub Actions network rule",
        ),

        # -- Network policy (applied to user only) --
        (
            f"CREATE OR REPLACE NETWORK POLICY {NETWORK_POLICY} "
            f"ALLOWED_NETWORK_RULE_LIST = ('{NETWORK_RULE}')",
            "Create CI/CD network policy",
        ),
        (
            f"ALTER USER {CICD_USER} SET NETWORK_POLICY = {NETWORK_POLICY}",
            "Apply network policy to service account",
        ),

        # -- Grants: warehouse --
        (
            f"GRANT USAGE ON WAREHOUSE {WAREHOUSE} TO ROLE {CICD_ROLE}",
            "Grant warehouse usage",
        ),

        # -- Grants: database and schemas --
        (
            f"GRANT USAGE ON DATABASE {DB_NAME} TO ROLE {CICD_ROLE}",
            "Grant database usage",
        ),
        (
            f"GRANT ALL ON SCHEMA {fqn_schema} TO ROLE {CICD_ROLE}",
            "Grant schema PIPELINE_STAGE",
        ),
        (
            f"GRANT ALL ON SCHEMA {DB_NAME}.DATA TO ROLE {CICD_ROLE}",
            "Grant schema DATA",
        ),

        # -- Grants: stages --
        (
            f"GRANT READ, WRITE ON STAGE {fqn_schema}.CODE_STAGE TO ROLE {CICD_ROLE}",
            "Grant CODE_STAGE",
        ),
        (
            f"GRANT READ, WRITE ON STAGE {fqn_schema}.JOB_STAGE TO ROLE {CICD_ROLE}",
            "Grant JOB_STAGE",
        ),
        (
            f"GRANT READ, WRITE ON STAGE {fqn_schema}.DAG_STAGE TO ROLE {CICD_ROLE}",
            "Grant DAG_STAGE",
        ),
        (
            f"GRANT READ, WRITE ON STAGE {fqn_schema}.ARTIFACTS_STAGE TO ROLE {CICD_ROLE}",
            "Grant ARTIFACTS_STAGE",
        ),

        # -- Grants: Git repository (WRITE needed for ALTER ... FETCH) --
        (
            f"GRANT READ, WRITE ON GIT REPOSITORY {fqn_repo} TO ROLE {CICD_ROLE}",
            "Grant Git repository read/write",
        ),

        # -- Grants: compute pool --
        (
            f"GRANT USAGE ON COMPUTE POOL {COMPUTE_POOL} TO ROLE {CICD_ROLE}",
            "Grant compute pool usage",
        ),

        # -- Grants: task execution --
        (
            f"GRANT EXECUTE TASK ON ACCOUNT TO ROLE {CICD_ROLE}",
            "Grant execute task",
        ),
        (
            f"GRANT EXECUTE MANAGED TASK ON ACCOUNT TO ROLE {CICD_ROLE}",
            "Grant execute managed task",
        ),

        # -- Grants: create objects --
        (
            f"GRANT CREATE TABLE ON SCHEMA {fqn_schema} TO ROLE {CICD_ROLE}",
            "Grant create table",
        ),
        (
            f"GRANT CREATE TASK ON SCHEMA {fqn_schema} TO ROLE {CICD_ROLE}",
            "Grant create task",
        ),
        (
            f"GRANT CREATE PROCEDURE ON SCHEMA {fqn_schema} TO ROLE {CICD_ROLE}",
            "Grant create procedure",
        ),
    ]

    print(f"Setting up CI/CD service account")
    print(f"  Role:     {CICD_ROLE}")
    print(f"  User:     {CICD_USER}")
    print(f"  Policy:   {NETWORK_POLICY}")
    print(f"  Database: {DB_NAME}")
    print(f"  Pool:     {COMPUTE_POOL}")
    print()

    for stmt, label in statements:
        try:
            session.sql(stmt).collect()
            print(f"  OK: {label}")
        except Exception as e:
            print(f"  WARN: {label}")
            print(f"        {e}")

    print()
    print("CI/CD service account setup complete.")
    print()
    print("Next steps:")
    print("  1. Store the private key as GitHub Secret: SNOWFLAKE_PRIVATE_KEY")
    print(f"  2. Store the account ID as GitHub Secret:  SNOWFLAKE_ACCOUNT")
    print(f"  3. Store '{CICD_USER}' as GitHub Secret:   SNOWFLAKE_USER")
    print(f"  4. Optionally set GitHub Variable:         SNOWFLAKE_ROLE = {CICD_ROLE}")


def get_session() -> Session:
    """Create a Snowpark session. Must connect as ACCOUNTADMIN."""
    return Session.builder.config("connection_name", "keypair").create()


if __name__ == "__main__":
    rsa_public_key = os.getenv("RSA_PUBLIC_KEY", "").strip()
    if not rsa_public_key:
        print("ERROR: RSA_PUBLIC_KEY environment variable is required.")
        print()
        print("Generate a key pair first:")
        print("  openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out cicd_rsa_key.p8 -nocrypt")
        print("  openssl rsa -in cicd_rsa_key.p8 -pubout -out cicd_rsa_key.pub")
        print()
        print("Then export the public key (single line, no headers):")
        print('  export RSA_PUBLIC_KEY="$(grep -v "BEGIN\\|END" cicd_rsa_key.pub | tr -d \'\\n\')"')
        sys.exit(1)

    session = get_session()
    print(f"Connected as: {session.get_current_role()}")
    print()

    try:
        setup_cicd_service_account(session, rsa_public_key)
    finally:
        session.close()
