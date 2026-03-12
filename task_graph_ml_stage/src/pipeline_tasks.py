"""
Pipeline Task Entry Points for Car Insurance ML DAG.

Heavy compute tasks (ingest, train, inference) run as ML Jobs via
MLJobDefinition.register() with source pointing to a Snowflake Git
Repository stage. The code is sourced from Git rather than uploaded
from local.

The tasks here (prepare_data, check_quality, promote_model, send_alert,
cleanup) run on the warehouse as StoredProcedure-based tasks since they
perform lightweight operations (Feature Store setup, registry lookups,
metric comparisons, etc.).

Code is synced from the Git stage to CODE_STAGE at deploy time by CI/CD.
"""

import json

from snowflake.core.task.context import TaskContext
from snowflake.snowpark import Session


# =============================================================================
# Task: Prepare Data (Feature Store + Dataset)
# =============================================================================

def task_prepare_data(session: Session) -> str:
    """
    Set up the Feature Store and prepare training datasets.

    Reads ingest results from predecessor (INGEST_DATA), sets up Feature Store
    if needed, generates training dataset, and passes dataset info downstream.

    Runs on the warehouse (not compute pool) since Feature Store operations
    use Snowpark DataFrames and SQL.
    """
    from feature_ops import setup_feature_store, prepare_datasets
    from constants import PIPELINE_DB, DATA_SCHEMA

    print("=== TASK: Prepare Data ===")

    # Read predecessor output
    ctx = TaskContext(session)
    ingest_result = json.loads(ctx.get_predecessor_return_value("INGEST_DATA"))
    print(f"Received ingest result: {ingest_result}")

    # Setup Feature Store (creates entity + feature view if not exists)
    print("Setting up Feature Store...")
    fv = setup_feature_store(session)

    # Prepare training dataset
    print("Preparing training dataset...")
    dataset_info = prepare_datasets(session, fv)
    print(f"Dataset: {dataset_info['dataset_name']} v{dataset_info['dataset_version']}")

    return_value = json.dumps(dataset_info)
    ctx.set_return_value(return_value)
    return return_value


# =============================================================================
# Task: Check Model Quality (Branch)
# =============================================================================

def task_check_quality(session: Session) -> str:
    """
    Compare new model quality against production model.

    This is a DAGTaskBranch: returns the name of the next task to execute.
    Returns "PROMOTE_MODEL" if new model is better, "SEND_ALERT" otherwise.
    """
    from modeling import check_model_quality

    print("=== TASK: Check Model Quality ===")

    ctx = TaskContext(session)
    train_result = json.loads(ctx.get_predecessor_return_value("TRAIN_MODEL"))
    metrics = train_result["metrics"]
    version_name = train_result["version_name"]

    print(f"Evaluating model {version_name}: {metrics}")

    decision = check_model_quality(session, metrics)
    print(f"Quality check decision: {decision}")

    # Store the full result for downstream tasks
    return_value = json.dumps({
        "decision": decision,
        "version_name": version_name,
        "metrics": metrics,
    })
    ctx.set_return_value(return_value)

    # The branch function must return the name of the next task
    if decision == "promote_model":
        return "PROMOTE_MODEL"
    else:
        return "SEND_ALERT"


# =============================================================================
# Task: Promote Model (conditional)
# =============================================================================

def task_promote_model(session: Session) -> str:
    """
    Promote the new model version to production (default).

    Only runs if check_quality decided to promote.
    """
    from modeling import get_registry, promote_model
    from constants import MODEL_NAME

    print("=== TASK: Promote Model ===")

    ctx = TaskContext(session)
    train_result = json.loads(ctx.get_predecessor_return_value("TRAIN_MODEL"))
    version_name = train_result["version_name"]

    registry = get_registry(session)
    base_model = registry.get_model(MODEL_NAME)
    mv = base_model.version(version_name)

    promote_model(session, mv)

    return_value = json.dumps({
        "promoted_version": version_name,
        "status": "promoted",
    })
    ctx.set_return_value(return_value)
    print(f"Model {version_name} promoted to production")
    return return_value


# =============================================================================
# Task: Send Alert (conditional)
# =============================================================================

def task_send_alert(session: Session) -> str:
    """
    Send alert that model quality did not meet promotion criteria.

    Only runs if check_quality decided NOT to promote.
    """
    print("=== TASK: Send Alert ===")

    ctx = TaskContext(session)
    train_result = json.loads(ctx.get_predecessor_return_value("TRAIN_MODEL"))
    version_name = train_result["version_name"]
    metrics = train_result["metrics"]

    alert_msg = (
        f"Model {version_name} did not meet promotion criteria. "
        f"Metrics: {json.dumps(metrics)}. "
        "Manual review recommended."
    )
    print(f"ALERT: {alert_msg}")

    from constants import PIPELINE_DB, PIPELINE_SCHEMA
    session.sql(
        "INSERT INTO IDENTIFIER(?) (ALERT_TIME, ALERT_TYPE, MESSAGE) "
        "SELECT CURRENT_TIMESTAMP(), 'MODEL_QUALITY', ?",
    ).bind([f"{PIPELINE_DB}.{PIPELINE_SCHEMA}.PIPELINE_ALERTS", alert_msg]).collect()

    return_value = json.dumps({
        "version_name": version_name,
        "status": "alert_sent",
        "message": alert_msg,
    })
    ctx.set_return_value(return_value)
    return return_value


# =============================================================================
# Finalizer: Cleanup
# =============================================================================

def task_cleanup(session: Session) -> str:
    """
    Cleanup old model versions and temporary artifacts.

    Runs as a DAG finalizer - always executes regardless of task success/failure.
    """
    from modeling import cleanup

    print("=== TASK: Cleanup ===")

    cleanup(session)

    return_value = json.dumps({"status": "cleanup_complete"})
    TaskContext(session).set_return_value(return_value)
    print("Cleanup complete")
    return return_value
