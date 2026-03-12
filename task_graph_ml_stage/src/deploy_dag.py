"""
Deploy the Car Insurance ML Pipeline as a Snowflake Task Graph.

This script is designed to run from CI/CD (GitHub Actions) on every push to
main, as well as locally for development. It registers MLJobDefinitions from
the Snowflake Git Repository stage and deploys the DAG.

At deploy time the script refreshes the Git repo stage and syncs source files
to CODE_STAGE so that all tasks use the code from the commit that triggered
the deployment. There is no runtime REFRESH_GIT task — the DAG always runs
with the code that was current at deploy time.

Lightweight tasks (prepare_data, check_quality, promote_model, send_alert,
cleanup) use StoredProcedureCall on the warehouse. Because sproc.register()
needs PUT (write) access to its stage_location and Git repository stages are
read-only, these SPs import from an internal CODE_STAGE. At deploy time, source
files are copied from the Git stage into CODE_STAGE via COPY FILES.

DAG structure:

    INGEST_DATA (MLJobDefinition) >> PREPARE_DATA (SP)
    >> TRAIN_MODEL (MLJobDefinition) >> CHECK_QUALITY (SP Branch)
    >> [PROMOTE_MODEL (SP), SEND_ALERT (SP)]
    PROMOTE_MODEL >> RUN_INFERENCE (MLJobDefinition)
    CLEANUP (SP finalizer - always runs)

Prerequisites:
    1. Run setup_infrastructure.sql to create DB, schemas, stages, and Git integration.
    2. Ensure compute pool (DEMO_POOL) exists and is running.
    3. Ensure the Git repository has the pipeline code committed and pushed.

Usage:
    python deploy_dag.py [--execute]

    --execute: Deploy and immediately execute the DAG once.
               Without this flag, the DAG is created in suspended state.

    Environment variables (for CI/CD):
        SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PRIVATE_KEY_FILE
        Optional: SNOWFLAKE_ROLE, SNOWFLAKE_WAREHOUSE
"""

import argparse
import os
import sys

from snowflake.core import Root
from snowflake.core._common import CreateMode
from snowflake.core.task import StoredProcedureCall, Cron
from snowflake.core.task.dagv1 import DAG, DAGTask, DAGTaskBranch, DAGOperation
from snowflake.ml.jobs import MLJobDefinition
from snowflake.snowpark import Session

sys.path.insert(0, ".")
from constants import (
    PIPELINE_DB,
    PIPELINE_SCHEMA,
    DATA_SCHEMA,
    CODE_STAGE,
    JOB_STAGE,
    GIT_SRC_PATH,
    GIT_JOBS_PATH,
    GIT_REPO_NAME,
    WAREHOUSE,
    ROLE_NAME,
    COMPUTE_POOL,
)

# Import SP-based task functions for warehouse tasks
from pipeline_tasks import (
    task_prepare_data,
    task_check_quality,
    task_promote_model,
    task_send_alert,
    task_cleanup,
)

# =============================================================================
# DAG Configuration
# =============================================================================

DAG_NAME = "CAR_INSURANCE_ML_PIPELINE"
DAG_SCHEDULE = Cron("0 9 * * *", "UTC")

# Stage imports for StoredProcedureCall tasks (warehouse tasks only)
# These point to the internal writable CODE_STAGE (synced from Git at deploy time).
# COPY FILES preserves the relative directory structure from the Git stage, so
# files end up under CODE_STAGE/src/ (mirroring the repo layout).
CODE_STAGE_SRC = f"{CODE_STAGE}/src"
STAGE_IMPORTS = [
    f"{CODE_STAGE_SRC}/constants.py",
    f"{CODE_STAGE_SRC}/data_ops.py",
    f"{CODE_STAGE_SRC}/feature_ops.py",
    f"{CODE_STAGE_SRC}/modeling.py",
    f"{CODE_STAGE_SRC}/pipeline_tasks.py",
    f"{CODE_STAGE_SRC}/helpers/__init__.py",
    f"{CODE_STAGE_SRC}/helpers/data_generation.py",
    f"{CODE_STAGE_SRC}/helpers/feature_engineering.py",
    f"{CODE_STAGE_SRC}/helpers/model_artifacts.py",
    f"{CODE_STAGE_SRC}/helpers/inference_input.py",
]

# Packages required by the stored procedures
PACKAGES = [
    "snowflake-snowpark-python",
    "snowflake-ml-python",
    "xgboost",
    "scikit-learn",
    "pandas",
    "numpy",
    "cloudpickle",
]


def ensure_environment(session: Session) -> None:
    """Set up the Snowflake environment for DAG deployment."""
    session.use_role(ROLE_NAME)
    session.use_warehouse(WAREHOUSE)
    session.use_database(PIPELINE_DB)
    session.use_schema(PIPELINE_SCHEMA)


def refresh_git_repo(session: Session) -> None:
    """Fetch the latest code from the remote Git repository."""
    fqn_repo = f"{PIPELINE_DB}.{PIPELINE_SCHEMA}.{GIT_REPO_NAME}"
    print(f"Refreshing Git repository: {fqn_repo}")
    session.sql(f"ALTER GIT REPOSITORY {fqn_repo} FETCH").collect()
    print("Git repository refreshed.\n")


def sync_git_to_code_stage(session: Session) -> None:
    """
    Copy source files from the Git repo stage to the internal CODE_STAGE.

    StoredProcedureCall uses sproc.register(), which needs PUT (write) access
    to the stage_location. Git repository stages are read-only, so we keep
    CODE_STAGE as a writable internal stage and sync files into it from Git.
    """
    print(f"Syncing source files from Git stage to CODE_STAGE...")
    session.sql(f"COPY FILES INTO {CODE_STAGE} FROM {GIT_SRC_PATH}").collect()
    print("Source files synced to CODE_STAGE.\n")


def register_job_definitions(session: Session) -> dict:
    """
    Register MLJobDefinitions for each ML Job task using submit_from_stage pattern.

    Each job is a standalone script inside the jobs/ directory in the Git
    Repository stage. The Git repo stage path is used as the source, and each
    script serves as the entrypoint for its respective job definition.

    Returns a dict of {task_name: MLJobDefinition}.
    """
    print(f"Registering MLJobDefinitions from Git stage: {GIT_JOBS_PATH}...")

    # Register ingest_data job definition
    ingest_def = MLJobDefinition.register(
        source=GIT_JOBS_PATH,
        entrypoint="ingest_data.py",
        compute_pool=COMPUTE_POOL,
        stage_name=JOB_STAGE,
        session=session,
    )
    print(f"  Registered: ingest_data (entrypoint=ingest_data.py)")

    # Register train_model job definition
    train_def = MLJobDefinition.register(
        source=GIT_JOBS_PATH,
        entrypoint="train_model.py",
        compute_pool=COMPUTE_POOL,
        stage_name=JOB_STAGE,
        session=session,
    )
    print(f"  Registered: train_model (entrypoint=train_model.py)")

    # Register run_inference job definition
    inference_def = MLJobDefinition.register(
        source=GIT_JOBS_PATH,
        entrypoint="run_inference.py",
        compute_pool=COMPUTE_POOL,
        stage_name=JOB_STAGE,
        session=session,
    )
    print(f"  Registered: run_inference (entrypoint=run_inference.py)")

    return {
        "ingest_data": ingest_def,
        "train_model": train_def,
        "run_inference": inference_def,
    }


def create_dag(session: Session, job_definitions: dict) -> DAG:
    """
    Define the Car Insurance ML Pipeline DAG.

    ML Job tasks (ingest, train, inference) use MLJobDefinition objects.
    Warehouse tasks (prepare_data, check_quality, promote, alert, cleanup)
    use StoredProcedureCall. Code is synced from Git at deploy time (CI/CD),
    so there is no runtime Git refresh task.
    """
    dag = DAG(
        name=DAG_NAME,
        schedule=DAG_SCHEDULE,
        warehouse=WAREHOUSE,
        stage_location=CODE_STAGE,
        use_func_return_value=True,
        comment="Car Insurance ML pipeline: CI/CD deployed, ingest, prepare, train, evaluate, promote, infer",
    )

    with dag:
        # -- Task 1: Ingest Data (MLJobDefinition on compute pool) --
        ingest_data = DAGTask(
            name="INGEST_DATA",
            definition=job_definitions["ingest_data"],
        )

        # -- Task 2: Prepare Data (StoredProcedure on warehouse) --
        prepare_data = DAGTask(
            name="PREPARE_DATA",
            definition=StoredProcedureCall(
                func=task_prepare_data,
                stage_location=CODE_STAGE,
                imports=STAGE_IMPORTS,
                packages=PACKAGES,
            ),
            warehouse=WAREHOUSE,
        )

        # -- Task 3: Train Model (MLJobDefinition on compute pool) --
        train_model = DAGTask(
            name="TRAIN_MODEL",
            definition=job_definitions["train_model"],
        )

        # -- Task 4: Check Quality (Branch, StoredProcedure on warehouse) --
        check_quality = DAGTaskBranch(
            name="CHECK_QUALITY",
            definition=StoredProcedureCall(
                func=task_check_quality,
                stage_location=CODE_STAGE,
                imports=STAGE_IMPORTS,
                packages=PACKAGES,
            ),
            warehouse=WAREHOUSE,
        )

        # -- Task 5a: Promote Model (conditional, SP on warehouse) --
        promote_model = DAGTask(
            name="PROMOTE_MODEL",
            definition=StoredProcedureCall(
                func=task_promote_model,
                stage_location=CODE_STAGE,
                imports=STAGE_IMPORTS,
                packages=PACKAGES,
            ),
            warehouse=WAREHOUSE,
        )

        # -- Task 5b: Send Alert (conditional, SP on warehouse) --
        send_alert = DAGTask(
            name="SEND_ALERT",
            definition=StoredProcedureCall(
                func=task_send_alert,
                stage_location=CODE_STAGE,
                imports=STAGE_IMPORTS,
                packages=PACKAGES,
            ),
            warehouse=WAREHOUSE,
        )

        # -- Task 6: Run Inference (MLJobDefinition on compute pool) --
        run_inference = DAGTask(
            name="RUN_INFERENCE",
            definition=job_definitions["run_inference"],
        )

        # -- Finalizer: Cleanup (always runs, SP on warehouse) --
        cleanup = DAGTask(
            name="CLEANUP",
            definition=StoredProcedureCall(
                func=task_cleanup,
                stage_location=CODE_STAGE,
                imports=STAGE_IMPORTS,
                packages=PACKAGES,
            ),
            warehouse=WAREHOUSE,
            is_finalizer=True,
        )

        # -- Define Dependencies --
        # INGEST_DATA >> PREPARE_DATA >> TRAIN_MODEL >> CHECK_QUALITY
        # >> [PROMOTE_MODEL, SEND_ALERT]
        # PROMOTE_MODEL >> RUN_INFERENCE
        ingest_data >> prepare_data >> train_model >> check_quality >> [promote_model, send_alert]
        promote_model >> run_inference

    return dag


def deploy_dag(session: Session, execute: bool = False) -> None:
    """Deploy the DAG to Snowflake."""
    print(f"Building DAG: {DAG_NAME}")

    session.use_database(PIPELINE_DB)
    session.use_schema(PIPELINE_SCHEMA)

    # Refresh Git repo to ensure latest code is available for registration
    refresh_git_repo(session)

    # Copy source files from Git stage to writable CODE_STAGE (needed by sproc.register)
    sync_git_to_code_stage(session)

    # Ensure environment is set up
    ensure_environment(session)

    # Register MLJobDefinitions for compute-pool tasks (from Git stage)
    job_definitions = register_job_definitions(session)

    dag = create_dag(session, job_definitions)

    # Create an alerts table for the SEND_ALERT task
    session.sql(f"""
        CREATE TABLE IF NOT EXISTS {PIPELINE_DB}.{PIPELINE_SCHEMA}.PIPELINE_ALERTS (
            ALERT_TIME TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            ALERT_TYPE VARCHAR,
            MESSAGE VARCHAR
        )
    """).collect()
    print("Alerts table created/verified")

    # Deploy the DAG
    root = Root(session)
    schema = root.databases[PIPELINE_DB].schemas[PIPELINE_SCHEMA]

    print(f"Deploying DAG to {PIPELINE_DB}.{PIPELINE_SCHEMA}...")
    dag_op = DAGOperation(schema)
    dag_op.deploy(dag, mode=CreateMode.or_replace)
    print(f"DAG '{DAG_NAME}' deployed successfully")

    # List tasks in the DAG
    tasks = session.sql(f"""
        SHOW TASKS LIKE '%{DAG_NAME}%' IN SCHEMA {PIPELINE_DB}.{PIPELINE_SCHEMA}
    """).collect()
    print(f"\nTasks in DAG ({len(tasks)}):")
    for t in tasks:
        print(f"  - {t['name']} (state: {t['state']})")

    if execute:
        print(f"\nExecuting DAG '{DAG_NAME}'...")
        dag_op.run(dag)
        print("DAG execution triggered. Monitor progress in Snowsight or via:")
        print(f"  SELECT * FROM TABLE({PIPELINE_DB}.INFORMATION_SCHEMA.TASK_HISTORY()) ORDER BY SCHEDULED_TIME DESC;")
    else:
        print(f"\nDAG created in SUSPENDED state. To execute:")
        print(f"  ALTER TASK {PIPELINE_DB}.{PIPELINE_SCHEMA}.{DAG_NAME} RESUME;")
        print(f"  EXECUTE TASK {PIPELINE_DB}.{PIPELINE_SCHEMA}.{DAG_NAME};")


def get_session() -> Session:
    """Create a Snowpark session.

    In CI/CD (GitHub Actions), connection parameters come from environment
    variables (SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PRIVATE_KEY_FILE).
    Locally, the named 'keypair' connection from ~/.snowflake/connections.toml
    is used.
    """
    if os.getenv("SNOWFLAKE_ACCOUNT"):
        return Session.builder.configs({
            "account": os.environ["SNOWFLAKE_ACCOUNT"],
            "user": os.environ["SNOWFLAKE_USER"],
            "private_key_file": os.environ["SNOWFLAKE_PRIVATE_KEY_FILE"],
            "role": os.getenv("SNOWFLAKE_ROLE", ROLE_NAME),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", WAREHOUSE),
        }).create()
    return Session.builder.config("connection_name", "keypair").create()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy the Car Insurance ML Pipeline DAG")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the DAG immediately after deployment",
    )
    args = parser.parse_args()

    session = get_session()
    print(f"Connected as: {session.get_current_role()}")

    try:
        deploy_dag(session, execute=args.execute)
    finally:
        session.close()
