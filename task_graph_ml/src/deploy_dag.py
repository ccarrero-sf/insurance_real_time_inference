"""
Deploy the Car Insurance ML Pipeline as a Snowflake Task Graph (ML Jobs version).

This version uses @remote-decorated ML Job functions directly as DAG task
definitions for compute-intensive steps (ingest, train, inference), following
the pattern from the Snowflake ML Jobs e2e_task_graph example.

Lightweight tasks (prepare_data, check_quality, promote_model, send_alert,
cleanup) still use StoredProcedureCall on the warehouse.

DAG structure:

    INGEST_DATA (ML Job) >> PREPARE_DATA (SP) >> TRAIN_MODEL (ML Job)
    >> CHECK_QUALITY (SP Branch) >> [PROMOTE_MODEL (SP), SEND_ALERT (SP)]
    PROMOTE_MODEL >> RUN_INFERENCE (ML Job)
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
from snowflake.snowpark import Session

sys.path.insert(0, ".")
from constants import (
    PIPELINE_DB,
    PIPELINE_SCHEMA,
    DATA_SCHEMA,
    CODE_STAGE,
    WAREHOUSE,
    ROLE_NAME,
    COMPUTE_POOL,
)

# Import @remote ML Job functions - used directly as DAG task definitions
from data_ops import ingest_data_job
from modeling import train_model_job, run_inference_job

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

SOURCE_FILES = [
    "constants.py",
    "data_ops.py",
    "feature_ops.py",
    "modeling.py",
    "pipeline_tasks.py",
    os.path.join("helpers", "__init__.py"),
    os.path.join("helpers", "data_generation.py"),
    os.path.join("helpers", "feature_engineering.py"),
    os.path.join("helpers", "model_artifacts.py"),
    os.path.join("helpers", "inference_input.py"),
]


def ensure_environment(session: Session) -> None:
    """Set up the Snowflake environment for DAG deployment."""
    session.use_role(ROLE_NAME)
    session.use_warehouse(WAREHOUSE)
    session.use_database(PIPELINE_DB)
    session.use_schema(PIPELINE_SCHEMA)


def upload_source_files(session: Session) -> None:
    """Upload local source files to CODE_STAGE so SP tasks can import them at runtime."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Uploading source files to {CODE_STAGE}...")
    for filename in SOURCE_FILES:
        filepath = os.path.join(src_dir, filename)
        # Place files in matching subdirectory on the stage
        subdir = os.path.dirname(filename)
        stage_target = f"{CODE_STAGE}/{subdir}" if subdir else CODE_STAGE
        session.file.put(filepath, stage_target, auto_compress=False, overwrite=True)
        print(f"  {filename} -> {stage_target}/{os.path.basename(filename)}")
    print(f"Uploaded {len(SOURCE_FILES)} files\n")


def create_dag(session: Session) -> DAG:
    """
    Define the Car Insurance ML Pipeline DAG.

    ML Job tasks (ingest, train, inference) use @remote functions directly.
    Warehouse tasks (prepare_data, check_quality, promote, alert, cleanup)
    use StoredProcedureCall.
    """
    dag = DAG(
        name=DAG_NAME,
        schedule=DAG_SCHEDULE,
        warehouse=WAREHOUSE,
        stage_location=CODE_STAGE,
        use_func_return_value=True,
        comment="Car Insurance ML pipeline (ML Jobs version): ingest, prepare, train, evaluate, promote, infer",
    )

    with dag:
        # -- Task 1: Ingest Data (ML Job on compute pool) --
        # The @remote decorated function runs directly on SPCS (no warehouse needed)
        ingest_data = DAGTask(
            name="INGEST_DATA",
            definition=ingest_data_job,
        )

        # -- Task 2: Prepare Data (StoredProcedure on warehouse) --
        # Feature Store setup + dataset generation runs on warehouse
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

        # -- Task 3: Train Model (ML Job on compute pool) --
        # Training runs on SPCS via @remote (no warehouse needed)
        train_model = DAGTask(
            name="TRAIN_MODEL",
            definition=train_model_job,
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

        # -- Task 6: Run Inference (ML Job on compute pool) --
        # Batch inference runs on SPCS via @remote (no warehouse needed)
        run_inference = DAGTask(
            name="RUN_INFERENCE",
            definition=run_inference_job,
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

    # Ensure environment is set up for ML Job serialization
    ensure_environment(session)

    dag = create_dag(session)

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
    parser = argparse.ArgumentParser(description="Deploy the Car Insurance ML Pipeline DAG (ML Jobs version)")
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
