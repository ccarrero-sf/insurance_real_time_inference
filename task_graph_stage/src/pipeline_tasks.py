"""
Pipeline Task Entry Points for Car Insurance ML DAG.

Each function here is a task entry point that will be wrapped as a StoredProcedure
and executed by the Snowflake Task Graph. These functions use TaskContext for
inter-task communication and delegate heavy compute to ML Jobs via submit_from_stage.

The code is uploaded to @CODE_STAGE and referenced via imports in the
StoredProcedureCall definitions (see deploy_dag.py).
"""

import json
from datetime import datetime

from snowflake.core.task.context import TaskContext
from snowflake.snowpark import Session


# =============================================================================
# Task 1: Ingest Data
# =============================================================================

def task_ingest_data(session: Session) -> str:
    """
    Ingest (generate) new synthetic data into the pipeline database.

    Submits data generation as an ML Job on the compute pool, then passes
    table info downstream via TaskContext.set_return_value.
    """
    from data_ops import submit_ingest_job

    print("=" * 60)
    print("=== TASK: Ingest Data (ML Job) ===")
    print("=" * 60)
    print(f"[INGEST] Start time: {datetime.now().isoformat()}")
    print("[INGEST] Submitting data generation ML Job...")
    result = submit_ingest_job(session)

    return_value = json.dumps(result)
    TaskContext(session).set_return_value(return_value)
    print(f"[INGEST] Complete: {result['customers_count']} customers, {result['policies_count']} policies")
    print(f"[INGEST] Tables: {result.get('customers_table', 'N/A')}, {result.get('policies_table', 'N/A')}")
    print(f"[INGEST] End time: {datetime.now().isoformat()}")
    return return_value


# =============================================================================
# Task 2: Setup Feature Store & Train Model
# =============================================================================

def task_train_model(session: Session) -> str:
    """
    Set up the Feature Store, prepare datasets, and train the model via ML Job.

    Reads ingest results from predecessor, sets up Feature Store if needed,
    generates training dataset, submits training to SPCS compute pool,
    registers model, and saves artifacts to stage.
    """
    from feature_ops import setup_feature_store, prepare_datasets
    from modeling import train_model, save_model_artifacts, register_model

    from snowflake.ml.dataset import load_dataset
    from constants import PIPELINE_DB, DATA_SCHEMA

    print("=" * 60)
    print("=== TASK: Train Model ===")
    print("=" * 60)
    print(f"[TRAIN] Start time: {datetime.now().isoformat()}")

    # Read predecessor output
    ctx = TaskContext(session)
    ingest_result = json.loads(ctx.get_predecessor_return_value("INGEST_DATA"))
    print(f"[TRAIN] Predecessor INGEST_DATA result: {ingest_result}")

    # Setup Feature Store (creates entity + feature view if not exists)
    print("[TRAIN] Step 1/5: Setting up Feature Store...")
    fv = setup_feature_store(session)
    print("[TRAIN] Feature Store setup complete")

    # Prepare training dataset
    print("[TRAIN] Step 2/5: Preparing training dataset...")
    dataset_info = prepare_datasets(session, fv)
    print(f"[TRAIN] Dataset ready: {dataset_info['dataset_name']} v{dataset_info['dataset_version']}")

    # Train model via ML Job on SPCS
    print("[TRAIN] Step 3/5: Submitting training ML Job to SPCS compute pool...")
    training_result = train_model(
        session,
        dataset_name=dataset_info["dataset_name"],
        dataset_version=dataset_info["dataset_version"],
    )

    metrics = training_result["metrics"]
    print(f"[TRAIN] Training complete. Metrics: {metrics}")

    # Generate version name
    version_name = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[TRAIN] Generated version name: {version_name}")

    # Save artifacts to stage (backup/debugging)
    print("[TRAIN] Step 4/5: Saving model artifacts to stage...")
    run_id = version_name
    save_model_artifacts(session, training_result, run_id)

    # Register model in registry — the Pipeline includes preprocessing,
    # so no separate scaler/label_encoders needed
    print("[TRAIN] Step 5/5: Registering model in Snowflake Model Registry...")
    ds = load_dataset(session, dataset_info["dataset_name"], dataset_info["dataset_version"])
    mv = register_model(
        session,
        model_obj=training_result["model"],
        version_name=version_name,
        dataset=ds,
        metrics=metrics,
    )

    return_value = json.dumps({
        "version_name": version_name,
        "metrics": metrics,
        "dataset_name": dataset_info["dataset_name"],
        "dataset_version": dataset_info["dataset_version"],
    })
    ctx.set_return_value(return_value)
    print(f"[TRAIN] Model registered: {version_name}")
    print(f"[TRAIN] End time: {datetime.now().isoformat()}")
    return return_value


# =============================================================================
# Task 3: Check Model Quality (Branch)
# =============================================================================

def task_check_quality(session: Session) -> str:
    """
    Compare new model quality against production model.

    This is a DAGTaskBranch: returns the name of the next task to execute.
    Returns "PROMOTE_MODEL" if new model is better, "SEND_ALERT" otherwise.
    """
    from modeling import check_model_quality

    print("=" * 60)
    print("=== TASK: Check Model Quality (Branch) ===")
    print("=" * 60)
    print(f"[QUALITY] Start time: {datetime.now().isoformat()}")

    ctx = TaskContext(session)
    train_result = json.loads(ctx.get_predecessor_return_value("TRAIN_MODEL"))
    metrics = train_result["metrics"]
    version_name = train_result["version_name"]

    print(f"[QUALITY] Evaluating model version: {version_name}")
    print(f"[QUALITY] Model metrics: {metrics}")

    decision = check_model_quality(session, metrics)
    print(f"[QUALITY] Final decision: {decision}")

    # Return the task name to branch to
    # DAGTaskBranch expects the return value to be the downstream task name
    return_value = json.dumps({
        "decision": decision,
        "version_name": version_name,
        "metrics": metrics,
    })
    ctx.set_return_value(return_value)

    # The branch function must return the name of the next task
    if decision == "promote_model":
        print("[QUALITY] Branching to: PROMOTE_MODEL")
        print(f"[QUALITY] End time: {datetime.now().isoformat()}")
        return "PROMOTE_MODEL"
    else:
        print("[QUALITY] Branching to: SEND_ALERT")
        print(f"[QUALITY] End time: {datetime.now().isoformat()}")
        return "SEND_ALERT"


# =============================================================================
# Task 4a: Promote Model (conditional)
# =============================================================================

def task_promote_model(session: Session) -> str:
    """
    Promote the new model version to production (default).

    Only runs if check_quality decided to promote.
    """
    from modeling import get_registry, promote_model
    from constants import MODEL_NAME

    print("=" * 60)
    print("=== TASK: Promote Model ===")
    print("=" * 60)
    print(f"[PROMOTE] Start time: {datetime.now().isoformat()}")

    # Read from TRAIN_MODEL (not CHECK_QUALITY, whose return value is the
    # branch task name string due to use_func_return_value=True on the DAG).
    ctx = TaskContext(session)
    train_result = json.loads(ctx.get_predecessor_return_value("TRAIN_MODEL"))
    version_name = train_result["version_name"]
    print(f"[PROMOTE] Version to promote: {version_name}")

    print(f"[PROMOTE] Loading model '{MODEL_NAME}' from registry...")
    registry = get_registry(session)
    base_model = registry.get_model(MODEL_NAME)
    mv = base_model.version(version_name)
    print(f"[PROMOTE] Model version loaded: {mv.version_name}")

    print("[PROMOTE] Setting as default (production) version...")
    promote_model(session, mv)

    return_value = json.dumps({
        "promoted_version": version_name,
        "status": "promoted",
    })
    ctx.set_return_value(return_value)
    print(f"[PROMOTE] Model {version_name} promoted to production successfully")
    print(f"[PROMOTE] End time: {datetime.now().isoformat()}")
    return return_value


# =============================================================================
# Task 4b: Send Alert (conditional)
# =============================================================================

def task_send_alert(session: Session) -> str:
    """
    Send alert that model quality did not meet promotion criteria.

    Only runs if check_quality decided NOT to promote.
    """
    print("=" * 60)
    print("=== TASK: Send Alert ===")
    print("=" * 60)
    print(f"[ALERT] Start time: {datetime.now().isoformat()}")

    # Read from TRAIN_MODEL (not CHECK_QUALITY, whose return value is the
    # branch task name string due to use_func_return_value=True on the DAG).
    ctx = TaskContext(session)
    train_result = json.loads(ctx.get_predecessor_return_value("TRAIN_MODEL"))
    version_name = train_result["version_name"]
    metrics = train_result["metrics"]
    print(f"[ALERT] Model version: {version_name}")
    print(f"[ALERT] Model metrics: {metrics}")

    alert_msg = (
        f"Model {version_name} did not meet promotion criteria. "
        f"Metrics: {json.dumps(metrics)}. "
        "Manual review recommended."
    )
    print(f"[ALERT] Message: {alert_msg}")

    # In production, this could send an email, Slack message, etc.
    # For now, log the alert to a table
    from constants import PIPELINE_DB, PIPELINE_SCHEMA
    alert_table = f"{PIPELINE_DB}.{PIPELINE_SCHEMA}.PIPELINE_ALERTS"
    print(f"[ALERT] Inserting alert into {alert_table}...")
    # Escape single quotes in the message for safe SQL embedding
    safe_msg = alert_msg.replace("'", "''")
    session.sql(
        f"INSERT INTO {alert_table} (ALERT_TIME, ALERT_TYPE, MESSAGE) "
        f"SELECT CURRENT_TIMESTAMP(), 'MODEL_QUALITY', '{safe_msg}'"
    ).collect()
    print("[ALERT] Alert inserted successfully")

    return_value = json.dumps({
        "version_name": version_name,
        "status": "alert_sent",
        "message": alert_msg,
    })
    ctx.set_return_value(return_value)
    print(f"[ALERT] End time: {datetime.now().isoformat()}")
    return return_value


# =============================================================================
# Task 5: Run Inference
# =============================================================================

def task_run_inference(session: Session) -> str:
    """
    Run batch inference using the newly promoted production model.

    Uses the Feature Store to enrich data and calls mv.run() directly.
    The registered sklearn Pipeline handles all preprocessing internally.
    """
    from modeling import run_inference
    from constants import PIPELINE_DB, DATA_SCHEMA

    print("=" * 60)
    print("=== TASK: Run Inference ===")
    print("=" * 60)
    print(f"[INFERENCE] Start time: {datetime.now().isoformat()}")

    ctx = TaskContext(session)
    promote_result = json.loads(ctx.get_predecessor_return_value("PROMOTE_MODEL"))
    print(f"[INFERENCE] Predecessor PROMOTE_MODEL result: {promote_result}")
    print(f"[INFERENCE] Using promoted model version: {promote_result['promoted_version']}")

    input_table = f"{PIPELINE_DB}.{DATA_SCHEMA}.POLICIES"
    output_table = f"{PIPELINE_DB}.{DATA_SCHEMA}.PREDICTIONS"
    print(f"[INFERENCE] Input table: {input_table}")
    print(f"[INFERENCE] Output table: {output_table}")

    print("[INFERENCE] Starting batch inference...")
    pred_count = run_inference(session, input_table, output_table)

    return_value = json.dumps({
        "predictions_count": pred_count,
        "output_table": output_table,
        "model_version": promote_result["promoted_version"],
    })
    ctx.set_return_value(return_value)
    print(f"[INFERENCE] Inference complete: {pred_count} predictions written")
    print(f"[INFERENCE] End time: {datetime.now().isoformat()}")
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

    print("=" * 60)
    print("=== TASK: Cleanup (Finalizer) ===")
    print("=" * 60)
    print(f"[CLEANUP] Start time: {datetime.now().isoformat()}")
    print("[CLEANUP] Cleaning up old model versions and temporary artifacts...")

    cleanup(session)

    return_value = json.dumps({"status": "cleanup_complete"})
    TaskContext(session).set_return_value(return_value)
    print("[CLEANUP] Cleanup complete")
    print(f"[CLEANUP] End time: {datetime.now().isoformat()}")
    return return_value
