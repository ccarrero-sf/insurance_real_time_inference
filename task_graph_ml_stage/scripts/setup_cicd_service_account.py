"""
Set up a CI/CD service account for GitHub Actions → Snowflake DAG deployment.

This script must be run by an ACCOUNTADMIN (or a role with CREATE USER,
CREATE ROLE, CREATE NETWORK POLICY privileges). It creates:

  1. A dedicated CICD_DEPLOY_RL role with minimum required permissions.
  2. A CICD_DEPLOY_USER service account (key-pair auth only, no password).
  3. A network policy that allows GitHub Actions runner IPs.
  4. Grants so the service account can deploy the ML pipeline DAG.
  5. A PIPELINE_OPERATOR_RL role for humans to monitor and execute the DAG
     (without deploy/alter privileges).

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
    # First time: create user + grants
    export RSA_PUBLIC_KEY="MIIBIjANBgkq..."
    export OPERATOR_USER=CCARRERO
    python setup_cicd_service_account.py

    # Re-run after schema rebuild: grants only (no RSA_PUBLIC_KEY needed)
    export OPERATOR_USER=CCARRERO
    python setup_cicd_service_account.py

Environment variables:
    RSA_PUBLIC_KEY          (optional) Public key for the service account.
                            If omitted, skips user/network policy creation
                            and runs grants + operator role setup only.
    OPERATOR_USER           (optional) Username to grant PIPELINE_OPERATOR_RL to
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

# Operator role (for humans to monitor/execute the DAG)
OPERATOR_ROLE = "PIPELINE_OPERATOR_RL"

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

    # Run grants
    setup_cicd_grants(session)

    print()
    print("CI/CD service account setup complete.")
    print()
    print("Next steps:")
    print("  1. Store the private key as GitHub Secret: SNOWFLAKE_PRIVATE_KEY")
    print(f"  2. Store the account ID as GitHub Secret:  SNOWFLAKE_ACCOUNT")
    print(f"  3. Store '{CICD_USER}' as GitHub Secret:   SNOWFLAKE_USER")
    print(f"  4. Optionally set GitHub Variable:         SNOWFLAKE_ROLE = {CICD_ROLE}")


def setup_cicd_grants(session: Session) -> None:
    """Grant all required privileges to the CI/CD role.

    Can be run independently when the user/network policy already exist
    (e.g., after recreating a schema).
    """
    fqn_schema = f"{DB_NAME}.{SCHEMA_NAME}"
    fqn_repo = f"{fqn_schema}.{GIT_REPO_NAME}"

    statements = [
        # -- Ensure role exists --
        (
            f"CREATE ROLE IF NOT EXISTS {CICD_ROLE} "
            f"COMMENT = 'Role for GitHub Actions CI/CD pipeline deployments'",
            "Ensure CI/CD role exists",
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

        # -- Grants: task execution (needed to deploy tasks) --
        (
            f"GRANT EXECUTE TASK ON ACCOUNT TO ROLE {CICD_ROLE}",
            "Grant execute task",
        ),
        (
            f"GRANT EXECUTE MANAGED TASK ON ACCOUNT TO ROLE {CICD_ROLE}",
            "Grant execute managed task",
        ),

        # -- Grants: MANAGE GRANTS (needed to claim/release task ownership) --
        # The deploy script uses a claim→deploy→release pattern: it temporarily
        # takes OWNERSHIP of tasks owned by SPCS_PSE_ROLE, deploys with CREATE
        # OR REPLACE, then transfers OWNERSHIP back. MANAGE GRANTS is required
        # to take ownership from another role.
        (
            f"GRANT MANAGE GRANTS ON ACCOUNT TO ROLE {CICD_ROLE}",
            "Grant manage grants (for ownership transfer)",
        ),

        # -- Grants: create objects --
        (
            f"GRANT CREATE TABLE ON SCHEMA {fqn_schema} TO ROLE {CICD_ROLE}",
            "Grant create table",
        ),
        (
            f"GRANT CREATE TABLE ON SCHEMA {DB_NAME}.DATA TO ROLE {CICD_ROLE}",
            "Grant create table in DATA",
        ),
        (
            f"GRANT CREATE TASK ON SCHEMA {fqn_schema} TO ROLE {CICD_ROLE}",
            "Grant create task",
        ),
        (
            f"GRANT CREATE PROCEDURE ON SCHEMA {fqn_schema} TO ROLE {CICD_ROLE}",
            "Grant create procedure",
        ),

        # -- Grants: existing objects (DML on tables, not OWNERSHIP) --
        (
            f"GRANT ALL ON ALL TABLES IN SCHEMA {fqn_schema} TO ROLE {CICD_ROLE}",
            "Grant all existing tables in PIPELINE_STAGE",
        ),
        (
            f"GRANT ALL ON ALL TABLES IN SCHEMA {DB_NAME}.DATA TO ROLE {CICD_ROLE}",
            "Grant all existing tables in DATA",
        ),
        (
            f"GRANT ALL ON ALL PROCEDURES IN SCHEMA {fqn_schema} TO ROLE {CICD_ROLE}",
            "Grant all existing procedures in PIPELINE_STAGE",
        ),

        # -- Future grants (for objects created by other roles later) --
        (
            f"GRANT ALL ON FUTURE TABLES IN SCHEMA {fqn_schema} TO ROLE {CICD_ROLE}",
            "Future grant on tables in PIPELINE_STAGE",
        ),
        (
            f"GRANT ALL ON FUTURE TABLES IN SCHEMA {DB_NAME}.DATA TO ROLE {CICD_ROLE}",
            "Future grant on tables in DATA",
        ),

        # -- Task execution grants for the runtime owner role --
        # Tasks run as their owner (SPCS_PSE_ROLE). That role needs EXECUTE TASK
        # so the Snowflake scheduler can run them.
        (
            f"GRANT EXECUTE TASK ON ACCOUNT TO ROLE SPCS_PSE_ROLE",
            "Grant execute task to runtime owner (SPCS_PSE_ROLE)",
        ),
        (
            f"GRANT EXECUTE MANAGED TASK ON ACCOUNT TO ROLE SPCS_PSE_ROLE",
            "Grant execute managed task to runtime owner (SPCS_PSE_ROLE)",
        ),
    ]

    print(f"Setting up CI/CD grants for role: {CICD_ROLE}")
    print()

    for stmt, label in statements:
        try:
            session.sql(stmt).collect()
            print(f"  OK: {label}")
        except Exception as e:
            print(f"  WARN: {label}")
            print(f"        {e}")


def setup_operator_role(session: Session, operator_user: str | None = None) -> None:
    """Create a read-only operator role for monitoring and executing the DAG.

    This role can resume/suspend/execute tasks and read pipeline data, but
    cannot create, alter, or drop any objects. Intended for human operators.
    """
    fqn_schema = f"{DB_NAME}.{SCHEMA_NAME}"

    statements = [
        # -- Role --
        (
            f"CREATE ROLE IF NOT EXISTS {OPERATOR_ROLE} "
            f"COMMENT = 'Operator role for monitoring and executing the ML pipeline DAG "
            f"(no deploy/alter privileges)'",
            "Create operator role",
        ),

        # -- Warehouse --
        (
            f"GRANT USAGE ON WAREHOUSE {WAREHOUSE} TO ROLE {OPERATOR_ROLE}",
            "Grant warehouse usage",
        ),

        # -- Database and schemas --
        (
            f"GRANT USAGE ON DATABASE {DB_NAME} TO ROLE {OPERATOR_ROLE}",
            "Grant database usage",
        ),
        (
            f"GRANT USAGE ON SCHEMA {fqn_schema} TO ROLE {OPERATOR_ROLE}",
            "Grant schema PIPELINE_STAGE usage",
        ),
        (
            f"GRANT USAGE ON SCHEMA {DB_NAME}.DATA TO ROLE {OPERATOR_ROLE}",
            "Grant schema DATA usage",
        ),

        # -- Task execution (account-level) --
        (
            f"GRANT EXECUTE TASK ON ACCOUNT TO ROLE {OPERATOR_ROLE}",
            "Grant execute task",
        ),
        (
            f"GRANT EXECUTE MANAGED TASK ON ACCOUNT TO ROLE {OPERATOR_ROLE}",
            "Grant execute managed task",
        ),

        # -- MONITOR + OPERATE on tasks (can resume/suspend/execute, not alter/drop) --
        (
            f"GRANT MONITOR, OPERATE ON ALL TASKS IN SCHEMA {fqn_schema} TO ROLE {OPERATOR_ROLE}",
            "Grant operate on existing tasks",
        ),
        (
            f"GRANT MONITOR, OPERATE ON FUTURE TASKS IN SCHEMA {fqn_schema} TO ROLE {OPERATOR_ROLE}",
            "Grant operate on future tasks",
        ),

        # -- Read-only on tables --
        (
            f"GRANT SELECT ON ALL TABLES IN SCHEMA {fqn_schema} TO ROLE {OPERATOR_ROLE}",
            "Grant select on PIPELINE_STAGE tables",
        ),
        (
            f"GRANT SELECT ON ALL TABLES IN SCHEMA {DB_NAME}.DATA TO ROLE {OPERATOR_ROLE}",
            "Grant select on DATA tables",
        ),
        (
            f"GRANT SELECT ON FUTURE TABLES IN SCHEMA {fqn_schema} TO ROLE {OPERATOR_ROLE}",
            "Future grant select on PIPELINE_STAGE tables",
        ),
        (
            f"GRANT SELECT ON FUTURE TABLES IN SCHEMA {DB_NAME}.DATA TO ROLE {OPERATOR_ROLE}",
            "Future grant select on DATA tables",
        ),
    ]

    # Optionally assign the role to a user
    if operator_user:
        statements.append((
            f"GRANT ROLE {OPERATOR_ROLE} TO USER {operator_user}",
            f"Grant operator role to {operator_user}",
        ))

    print(f"Setting up operator role: {OPERATOR_ROLE}")
    print()

    for stmt, label in statements:
        try:
            session.sql(stmt).collect()
            print(f"  OK: {label}")
        except Exception as e:
            print(f"  WARN: {label}")
            print(f"        {e}")

    print()
    print("Operator role setup complete.")
    if operator_user:
        print(f"  User '{operator_user}' can now USE ROLE {OPERATOR_ROLE}")
    print(f"  Allowed: EXECUTE TASK, ALTER TASK ... RESUME/SUSPEND, SELECT on tables")
    print(f"  Denied:  CREATE/DROP/REPLACE tasks or other objects")


def get_session() -> Session:
    """Create a Snowpark session using the keypair connection, then switch to ACCOUNTADMIN."""
    session = Session.builder.config("connection_name", "keypair").create()
    session.sql("USE ROLE ACCOUNTADMIN").collect()
    return session


if __name__ == "__main__":
    rsa_public_key = os.getenv("RSA_PUBLIC_KEY", "").strip() or None
    operator_user = os.getenv("OPERATOR_USER", "").strip() or None

    session = get_session()
    print(f"Connected as: {session.get_current_role()}")
    print()

    try:
        if rsa_public_key:
            setup_cicd_service_account(session, rsa_public_key)
        else:
            print("RSA_PUBLIC_KEY not set -- skipping user/network policy creation.")
            print("Running grants-only setup (user and network policy must already exist).")
            print()
            setup_cicd_grants(session)

        print()
        print("=" * 60)
        print()
        setup_operator_role(session, operator_user)
    finally:
        session.close()
