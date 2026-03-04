"""
Deploy the Car Insurance ML Pipeline as a Snowflake Task Graph (DAG).

This script creates and deploys the DAG with the following structure:

    INGEST_DATA >> TRAIN_MODEL >> CHECK_QUALITY >> [PROMOTE_MODEL, SEND_ALERT]
    PROMOTE_MODEL >> RUN_INFERENCE
    CLEANUP (finalizer - always runs)

Prerequisites:
    1. Run setup_infrastructure.sql to create DB, schemas, and stages.
    2. Run upload_code.py to upload source code to @CODE_STAGE.
    3. Ensure compute pool (DEMO_POOL) exists and is running.

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

# Import constants and task functions - assumes this runs locally with task_graph/src in PYTHONPATH
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
from pipeline_tasks import (
    task_ingest_data,
    task_train_model,
    task_check_quality,
    task_promote_model,
    task_send_alert,
    task_run_inference,
    task_cleanup,
)

# =============================================================================
# DAG Configuration
# =============================================================================

DAG_NAME = "CAR_INSURANCE_ML_PIPELINE"
# Schedule: every day at 6am UTC. Adjust as needed.
DAG_SCHEDULE = Cron("0 6 * * *", "UTC")

# Stage imports: all task stored procedures import code from CODE_STAGE
# so that the task functions can reference data_ops, feature_ops, modeling, etc.
STAGE_IMPORTS = [
    f"{CODE_STAGE}/constants.py",
    f"{CODE_STAGE}/data_ops.py",
    f"{CODE_STAGE}/feature_ops.py",
    f"{CODE_STAGE}/modeling.py",
    f"{CODE_STAGE}/pipeline_tasks.py",
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


SOURCE_FILES = ["constants.py", "data_ops.py", "feature_ops.py", "modeling.py", "pipeline_tasks.py"]


def upload_source_files(session: Session) -> None:
    """Upload local source files to CODE_STAGE so tasks can import them at runtime."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Uploading source files to {CODE_STAGE}...")
    for filename in SOURCE_FILES:
        filepath = os.path.join(src_dir, filename)
        session.file.put(filepath, CODE_STAGE, auto_compress=False, overwrite=True)
        print(f"  {filename} -> {CODE_STAGE}/{filename}")
    print(f"Uploaded {len(SOURCE_FILES)} files\n")


def create_dag(session: Session) -> DAG:
    """
    Define the Car Insurance ML Pipeline DAG.

    Returns a DAG object ready for deployment.
    """
    dag = DAG(
        name=DAG_NAME,
        schedule=DAG_SCHEDULE,
        warehouse=WAREHOUSE,
        stage_location=CODE_STAGE,
        use_func_return_value=True,
        comment="End-to-end Car Insurance ML pipeline: ingest, train, evaluate, promote, infer",
    )

    with dag:
        # -- Task 1: Ingest Data --
        ingest_data = DAGTask(
            name="INGEST_DATA",
            definition=StoredProcedureCall(
                func=task_ingest_data,
                stage_location=CODE_STAGE,
                imports=STAGE_IMPORTS,
                packages=PACKAGES,
            ),
            warehouse=WAREHOUSE,
        )

        # -- Task 2: Train Model --
        train_model = DAGTask(
            name="TRAIN_MODEL",
            definition=StoredProcedureCall(
                func=task_train_model,
                stage_location=CODE_STAGE,
                imports=STAGE_IMPORTS,
                packages=PACKAGES,
            ),
            warehouse=WAREHOUSE,
        )

        # -- Task 3: Check Quality (Branch) --
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

        # -- Task 4a: Promote Model (conditional) --
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

        # -- Task 4b: Send Alert (conditional) --
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

        # -- Task 5: Run Inference --
        run_inference = DAGTask(
            name="RUN_INFERENCE",
            definition=StoredProcedureCall(
                func=task_run_inference,
                stage_location=CODE_STAGE,
                imports=STAGE_IMPORTS,
                packages=PACKAGES,
            ),
            warehouse=WAREHOUSE,
        )

        # -- Finalizer: Cleanup (always runs) --
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
        # INGEST_DATA >> TRAIN_MODEL >> CHECK_QUALITY >> [PROMOTE_MODEL, SEND_ALERT]
        # PROMOTE_MODEL >> RUN_INFERENCE
        ingest_data >> train_model >> check_quality >> [promote_model, send_alert]
        promote_model >> run_inference

    return dag


def deploy_dag(session: Session, execute: bool = False) -> None:
    """Deploy the DAG to Snowflake."""
    print(f"Building DAG: {DAG_NAME}")

    session.use_database(PIPELINE_DB)
    session.use_schema(PIPELINE_SCHEMA)

    upload_source_files(session)

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
