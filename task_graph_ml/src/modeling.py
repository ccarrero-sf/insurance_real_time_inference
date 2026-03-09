"""
Model Operations for Car Insurance ML Pipeline (ML Jobs version).

Training and inference run as ML Jobs on SPCS via @remote decorator.
These @remote functions are used directly as DAG task definitions
(no StoredProcedureCall wrapper). They use TaskContext for inter-task
communication within the DAG.
"""

import json
from datetime import datetime, timedelta
from typing import Optional

from snowflake.ml.jobs import remote
from snowflake.ml.model import ModelVersion
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session

from constants import (
    PIPELINE_DB,
    DATA_SCHEMA,
    PIPELINE_SCHEMA,
    COMPUTE_POOL,
    JOB_STAGE,
    MODEL_NAME,
    METRIC_NAME,
    METRIC_THRESHOLD,
)


# =============================================================================
# Training - ML Job DAG Task
# =============================================================================

@remote(
    COMPUTE_POOL,
    stage_name=JOB_STAGE,
    database=PIPELINE_DB,
    schema=PIPELINE_SCHEMA,
    imports=[("helpers", "helpers")],
)
def train_model_job() -> None:
    """
    ML Job DAG task: train XGBRegressor on SPCS compute pool.

    This function runs remotely on the compute pool as a DAG task.
    It reads dataset info from the predecessor task (PREPARE_DATA) via
    TaskContext, trains the model, evaluates it, saves artifacts to stage,
    registers the model, and passes results to downstream tasks.
    """
    import xgboost as xgb
    import json
    import numpy as np
    import pandas as pd
    from datetime import datetime
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from snowflake.ml.dataset import load_dataset
    from snowflake.ml.model.model_signature import infer_signature
    from snowflake.ml.registry import Registry
    from snowflake.core.task.context import TaskContext

    from helpers.feature_engineering import (
        encode_categoricals,
        scale_features,
        CATEGORICAL_COLS,
        NUMERIC_COLS,
        LABEL_COL,
    )
    from helpers.model_artifacts import (
        CarInsurancePricingModel,
        save_artifacts_to_stage,
        build_model_context,
    )

    session = Session.builder.getOrCreate()

    pipeline_db = "CC_INSURANCE_PIPELINE"
    data_schema = "DATA"
    pipeline_schema = "PIPELINE_MLJOBS"
    artifacts_stage = f"@{pipeline_db}.{pipeline_schema}.ARTIFACTS_STAGE"
    model_name = "CAR_INSURANCE_PRICING_MODEL"

    # Read predecessor output (PREPARE_DATA passes dataset info)
    ctx = TaskContext(session)
    prepare_result = json.loads(ctx.get_predecessor_return_value("PREPARE_DATA"))
    dataset_name = prepare_result["dataset_name"]
    dataset_version = prepare_result["dataset_version"]

    print(f"Loading dataset: {dataset_name} v{dataset_version}")
    ds = load_dataset(session, dataset_name, dataset_version)
    df = ds.read.to_pandas()

    # Feature preparation using helpers
    df, label_encoders, encoded_cols = encode_categoricals(df)
    feature_cols = NUMERIC_COLS + encoded_cols
    X, scaler = scale_features(df, feature_cols)
    y = df[LABEL_COL]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBRegressor
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = {
        "train_r2": round(float(r2_score(y_train, train_pred)), 4),
        "train_rmse": round(float(mean_squared_error(y_train, train_pred) ** 0.5), 2),
        "train_mae": round(float(mean_absolute_error(y_train, train_pred)), 2),
        "test_r2": round(float(r2_score(y_test, test_pred)), 4),
        "test_rmse": round(float(mean_squared_error(y_test, test_pred) ** 0.5), 2),
        "test_mae": round(float(mean_absolute_error(y_test, test_pred)), 2),
    }
    print(f"Training complete. Metrics: {metrics}")

    # Generate version name
    version_name = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Save model artifacts to stage (backup)
    artifacts = {
        "model": model,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_cols": feature_cols,
        "metrics": metrics,
    }
    artifact_path = save_artifacts_to_stage(session, artifacts, artifacts_stage, version_name)

    # Build CustomModel for registry
    pricing_model, _ = build_model_context(model, scaler, label_encoders, feature_cols)

    # =========================================================================
    # Register model in registry with both predict and transform signatures
    # =========================================================================

    input_cols = NUMERIC_COLS + CATEGORICAL_COLS
    sample_df = ds.read.to_snowpark_dataframe().limit(100)

    # Predict signature: raw input -> predicted premium
    signature = infer_signature(
        input_data=sample_df.select(input_cols),
        output_data=sample_df.select(LABEL_COL).with_column_renamed(
            LABEL_COL, "PREDICTED_PREMIUM"
        ),
    )

    # Transform signature: raw input -> scaled features
    sample_input_pd = sample_df.select(input_cols).to_pandas()
    transform_output = pricing_model.transform(sample_input_pd)
    transform_signature = infer_signature(
        input_data=sample_input_pd,
        output_data=transform_output,
    )

    # Ensure the module is in sys.modules so cloudpickle can find it
    import sys as _sys
    _module_name = CarInsurancePricingModel.__module__
    if _module_name not in _sys.modules:
        import types as _types
        _mod = _types.ModuleType(_module_name)
        _mod.CarInsurancePricingModel = CarInsurancePricingModel
        _sys.modules[_module_name] = _mod

    registry = Registry(
        session=session,
        database_name=pipeline_db,
        schema_name=data_schema,
    )
    mv = registry.log_model(
        pricing_model,
        model_name=model_name,
        version_name=version_name,
        signatures={"predict": signature, "transform": transform_signature},
        sample_input_data=sample_df.select(input_cols).limit(100),
        conda_dependencies=["xgboost", "scikit-learn", "pandas", "numpy"],
        target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
        comment=f"XGBoost model for car insurance premium prediction with embedded feature engineering. Version: {version_name}",
        metrics=metrics,
        options={"relax_version": True},
    )

    for metric_key, value in metrics.items():
        mv.set_metric(metric_name=metric_key, value=value)

    print(f"Registered model {mv.fully_qualified_model_name} version {mv.version_name}")

    # Pass results to downstream tasks via TaskContext
    result = {
        "version_name": version_name,
        "metrics": metrics,
        "artifact_path": artifact_path,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
    }
    ctx.set_return_value(json.dumps(result))


# =============================================================================
# Inference - ML Job DAG Task
# =============================================================================

@remote(
    COMPUTE_POOL,
    stage_name=JOB_STAGE,
    database=PIPELINE_DB,
    schema=PIPELINE_SCHEMA,
    imports=[("helpers", "helpers")],
)
def run_inference_job() -> None:
    """
    ML Job DAG task: run batch inference on SPCS compute pool.

    Reads the promoted model version from the predecessor task (PROMOTE_MODEL),
    runs predictions on the policies table, saves results, and passes
    prediction count to downstream tasks via TaskContext.
    """
    import json
    from snowflake.ml.registry import Registry
    from snowflake.core.task.context import TaskContext

    from helpers.inference_input import build_inference_input

    session = Session.builder.getOrCreate()

    pipeline_db = "CC_INSURANCE_PIPELINE"
    data_schema = "DATA"
    model_name = "CAR_INSURANCE_PRICING_MODEL"
    warehouse = "COMPUTE_WH"

    session.use_database(pipeline_db)
    session.use_schema(data_schema)

    # Read predecessor output
    ctx = TaskContext(session)
    promote_result = json.loads(ctx.get_predecessor_return_value("PROMOTE_MODEL"))
    print(f"Using promoted model: {promote_result['promoted_version']}")

    output_table = f"{pipeline_db}.{data_schema}.PREDICTIONS"

    registry = Registry(
        session=session,
        database_name=pipeline_db,
        schema_name=data_schema,
    )
    mv = registry.get_model(model_name).default

    # Build enriched input using helper
    input_df = build_inference_input(
        session=session,
        pipeline_db=pipeline_db,
        data_schema=data_schema,
        feature_store_db=pipeline_db,
        feature_store_schema=data_schema,
        warehouse=warehouse,
    )

    row_count = input_df.count()
    print(f"Running inference on {row_count} rows")

    predictions = mv.run(input_df, function_name="predict")
    predictions.write.save_as_table(output_table, mode="overwrite")

    pred_count = session.table(output_table).count()
    print(f"Predictions saved to {output_table}: {pred_count} rows")

    # Pass results to downstream tasks
    result = {
        "predictions_count": pred_count,
        "output_table": output_table,
        "model_version": promote_result["promoted_version"],
    }
    ctx.set_return_value(json.dumps(result))


# =============================================================================
# Registry helpers
# =============================================================================

def get_registry(session: Session) -> Registry:
    """Get the model registry."""
    return Registry(
        session=session,
        database_name=PIPELINE_DB,
        schema_name=DATA_SCHEMA,
    )


def promote_model(session: Session, mv: ModelVersion) -> None:
    """Set model version as the default (production) version."""
    registry = get_registry(session)
    base_model = registry.get_model(MODEL_NAME)
    base_model.default = mv
    print(f"Promoted {mv.version_name} to default (production)")


def get_production_model(session: Session) -> Optional[ModelVersion]:
    """Get the current production model version, or None."""
    registry = get_registry(session)
    try:
        base_model = registry.get_model(MODEL_NAME)
        return base_model.default
    except Exception:
        return None


def check_model_quality(session: Session, new_metrics: dict) -> str:
    """
    Compare new model metrics against current production model.

    Returns "promote_model" if new model is better, "send_alert" otherwise.
    """
    current_score = new_metrics.get(METRIC_NAME, 0)
    print(f"New model {METRIC_NAME}: {current_score}")

    prod_mv = get_production_model(session)
    if prod_mv is None:
        print("No production model exists. Promoting new model.")
        return "promote_model"

    try:
        prod_metrics = prod_mv.get_metric(METRIC_NAME)
        prod_score = prod_metrics if isinstance(prod_metrics, (int, float)) else 0
    except Exception:
        prod_score = 0

    print(f"Production model {METRIC_NAME}: {prod_score}")
    print(f"Threshold: {METRIC_THRESHOLD}")

    if current_score >= METRIC_THRESHOLD and current_score >= prod_score:
        print("New model meets threshold and is better than production. -> promote_model")
        return "promote_model"
    else:
        print("New model does not meet criteria. -> send_alert")
        return "send_alert"


# =============================================================================
# Cleanup
# =============================================================================

def cleanup(session: Session, expiry_days: int = 7) -> None:
    """Clean up old model versions and datasets beyond expiry."""
    registry = get_registry(session)
    try:
        base_model = registry.get_model(MODEL_NAME)
        versions = base_model.versions()
        default_version = base_model.default.version_name

        cutoff = datetime.now() - timedelta(days=expiry_days)
        for v in versions:
            if v.version_name == default_version:
                continue
            try:
                created = v.created_on
                if created and created < cutoff:
                    print(f"Deleting expired version: {v.version_name}")
                    base_model.delete_version(v.version_name)
            except Exception as e:
                print(f"Skipping version {v.version_name}: {e}")
    except Exception as e:
        print(f"Cleanup skipped: {e}")
