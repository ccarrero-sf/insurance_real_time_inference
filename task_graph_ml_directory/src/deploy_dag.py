"""
Deploy the Car Insurance ML Pipeline as a Snowflake Task Graph (submit_directory version).

This version uses MLJobDefinition.register() with submit_directory pattern
to register standalone job scripts as reusable DAG task definitions.
Each ML Job (ingest, train, inference) is a standalone Python script in the
jobs/ directory, registered as an MLJobDefinition that can be reused across
multiple DAG runs without re-uploading the payload.

Lightweight tasks (prepare_data, check_quality, promote_model, send_alert,
cleanup) still use StoredProcedureCall on the warehouse.

DAG structure:

    INGEST_DATA (MLJobDefinition) >> PREPARE_DATA (SP) >> TRAIN_MODEL (MLJobDefinition)
    >> CHECK_QUALITY (SP Branch) >> [PROMOTE_MODEL (SP), SEND_ALERT (SP)]
    PROMOTE_MODEL >> RUN_INFERENCE (MLJobDefinition)
    CLEANUP (SP finalizer - always runs)

Prerequisites:
    1. Run setup_infrastructure.sql to create DB, schemas, and stages.
    2. Ensure compute pool (DEMO_POOL) exists and is running.

Usage:
    python deploy_dag.py [--execute]

    --execute: Deploy and immediately execute the DAG once.
               Without this flag, the DAG is created in suspended state.
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
DAG_SCHEDULE = Cron("0 6 * * *", "UTC")

# Stage imports for StoredProcedureCall tasks (warehouse tasks only)
STAGE_IMPORTS = [
    f"{CODE_STAGE}/constants.py",
    f"{CODE_STAGE}/data_ops.py",
    f"{CODE_STAGE}/feature_ops.py",
    f"{CODE_STAGE}/modeling.py",
    f"{CODE_STAGE}/pipeline_tasks.py",
    f"{CODE_STAGE}/helpers/__init__.py",
    f"{CODE_STAGE}/helpers/data_generation.py",
    f"{CODE_STAGE}/helpers/feature_engineering.py",
    f"{CODE_STAGE}/helpers/model_artifacts.py",
    f"{CODE_STAGE}/helpers/inference_input.py",
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

# Directories to skip when uploading to CODE_STAGE
# (jobs/ is handled by MLJobDefinition.register automatically)
SKIP_DIRS = {"jobs", "__pycache__"}


def ensure_environment(session: Session) -> None:
    """Set up the Snowflake environment for DAG deployment."""
    session.use_role(ROLE_NAME)
    session.use_warehouse(WAREHOUSE)
    session.use_database(PIPELINE_DB)
    session.use_schema(PIPELINE_SCHEMA)


def upload_source_files(session: Session) -> None:
    """Upload the entire src/ directory tree to CODE_STAGE, preserving structure.

    Walks the directory recursively so new files are picked up automatically.
    Skips jobs/ (handled by MLJobDefinition.register) and __pycache__/.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Uploading source directory to {CODE_STAGE}...")

    uploaded = 0
    for dirpath, dirnames, filenames in os.walk(src_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            filepath = os.path.join(dirpath, filename)
            rel_dir = os.path.relpath(dirpath, src_dir)
            if rel_dir == ".":
                stage_target = CODE_STAGE
            else:
                stage_target = f"{CODE_STAGE}/{rel_dir}"

            session.file.put(filepath, stage_target, auto_compress=False, overwrite=True)
            rel_path = os.path.relpath(filepath, src_dir)
            print(f"  {rel_path} -> {stage_target}/{filename}")
            uploaded += 1

    print(f"Uploaded {uploaded} files\n")


def register_job_definitions(session: Session) -> dict:
    """
    Register MLJobDefinitions for each ML Job task using submit_directory pattern.

    Each job is a standalone script inside the jobs/ directory. The entire
    jobs/ directory (including helpers/) is uploaded as the payload, and each
    script serves as the entrypoint for its respective job definition.

    Returns a dict of {task_name: MLJobDefinition}.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    jobs_dir = os.path.join(src_dir, "jobs")

    print(f"Registering MLJobDefinitions from {jobs_dir}...")

    # Register ingest_data job definition
    ingest_def = MLJobDefinition.register(
        source=jobs_dir,
        entrypoint="ingest_data.py",
        compute_pool=COMPUTE_POOL,
        stage_name=JOB_STAGE,
        session=session,
    )
    print(f"  Registered: ingest_data (entrypoint=ingest_data.py)")

    # Register train_model job definition
    train_def = MLJobDefinition.register(
        source=jobs_dir,
        entrypoint="train_model.py",
        compute_pool=COMPUTE_POOL,
        stage_name=JOB_STAGE,
        session=session,
    )
    print(f"  Registered: train_model (entrypoint=train_model.py)")

    # Register run_inference job definition
    inference_def = MLJobDefinition.register(
        source=jobs_dir,
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
    use StoredProcedureCall.
    """
    dag = DAG(
        name=DAG_NAME,
        schedule=DAG_SCHEDULE,
        warehouse=WAREHOUSE,
        stage_location=CODE_STAGE,
        use_func_return_value=True,
        comment="Car Insurance ML pipeline (submit_directory version): ingest, prepare, train, evaluate, promote, infer",
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
        # INGEST_DATA >> PREPARE_DATA >> TRAIN_MODEL >> CHECK_QUALITY >> [PROMOTE_MODEL, SEND_ALERT]
        # PROMOTE_MODEL >> RUN_INFERENCE
        ingest_data >> prepare_data >> train_model >> check_quality >> [promote_model, send_alert]
        promote_model >> run_inference

    return dag


def deploy_dag(session: Session, execute: bool = False) -> None:
    """Deploy the DAG to Snowflake."""
    print(f"Building DAG: {DAG_NAME}")

    session.use_database(PIPELINE_DB)
    session.use_schema(PIPELINE_SCHEMA)

    # Upload source files for SP-based tasks
    upload_source_files(session)

    # Ensure environment is set up
    ensure_environment(session)

    # Register MLJobDefinitions for compute-pool tasks
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
    """Create a Snowpark session from the default connection."""
    return Session.builder.config("connection_name", "keypair").create()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy the Car Insurance ML Pipeline DAG (submit_directory version)")
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
