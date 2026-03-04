"""
Model Operations for Car Insurance ML Pipeline.

Handles model training (via ML Jobs on SPCS), evaluation, registration,
promotion, quality checking, batch inference, and cleanup.
"""

import io
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import cloudpickle as cp
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from snowflake.ml.dataset import Dataset, load_dataset
from snowflake.ml.jobs import remote
from snowflake.ml.model import ModelVersion
from snowflake.ml.model.model_signature import infer_signature
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session

from constants import (
    PIPELINE_DB,
    DATA_SCHEMA,
    PIPELINE_SCHEMA,
    COMPUTE_POOL,
    JOB_STAGE,
    ARTIFACTS_STAGE,
    MODEL_NAME,
    METRIC_NAME,
    METRIC_THRESHOLD,
    WAREHOUSE,
)


# =============================================================================
# Training
# =============================================================================

CATEGORICAL_COLS = [
    "CAR_MAKE", "CAR_MODEL", "FUEL_TYPE", "TRANSMISSION",
    "COVERAGE_TYPE", "GENDER", "STATE",
]

NUMERIC_COLS = [
    "CAR_AGE", "KILOMETERS", "ENGINE_SIZE", "ESTIMATED_CAR_VALUE",
    "AGE", "YEARS_LICENSED", "CLAIMS_HISTORY", "CREDIT_SCORE",
    "RISK_SCORE", "AVG_CLAIMS_PER_YEAR", "TOTAL_POLICIES", "AVG_CAR_AGE",
    "AVG_KILOMETERS", "TOTAL_CAR_VALUE", "AVG_DEDUCTIBLE",
]

LABEL_COL = "ANNUAL_PREMIUM"

EXCLUDE_COLS = [
    "CUSTOMER_ID", "TS", "CREDIT_TIER", "ANNUAL_PREMIUM",
]


def _prepare_features(df: pd.DataFrame):
    """
    Prepare features: encode categoricals, scale numerics, split X/y.

    Returns (X, y, label_encoders, scaler) for training, or
    (X, label_encoders, scaler) when y is not available.
    """
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col + "_ENCODED"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    encoded_cols = [c + "_ENCODED" for c in CATEGORICAL_COLS]
    feature_cols = NUMERIC_COLS + encoded_cols

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)

    if LABEL_COL in df.columns:
        y = df[LABEL_COL]
        return X, y, label_encoders, scaler
    return X, label_encoders, scaler


@remote(COMPUTE_POOL, stage_name=JOB_STAGE)
def train_model_remote(session: Session, dataset_name: str, dataset_version: str):
    """
    Train XGBRegressor on SPCS via ML Job.

    This function runs remotely on the compute pool. It loads the dataset,
    prepares features, trains the model, and returns the trained artifacts.
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    from snowflake.ml.dataset import load_dataset

    ds = load_dataset(session, dataset_name, dataset_version)
    df = ds.read.to_pandas()

    categorical_cols = [
        "CAR_MAKE", "CAR_MODEL", "FUEL_TYPE", "TRANSMISSION",
        "COVERAGE_TYPE", "GENDER", "STATE",
    ]
    numeric_cols = [
        "CAR_AGE", "KILOMETERS", "ENGINE_SIZE", "ESTIMATED_CAR_VALUE",
        "AGE", "YEARS_LICENSED", "CLAIMS_HISTORY", "CREDIT_SCORE",
        "RISK_SCORE", "AVG_CLAIMS_PER_YEAR", "TOTAL_POLICIES", "AVG_CAR_AGE",
        "AVG_KILOMETERS", "TOTAL_CAR_VALUE", "AVG_DEDUCTIBLE",
    ]
    label_col = "ANNUAL_PREMIUM"
    exclude_cols = ["CUSTOMER_ID", "TS", "CREDIT_TIER", "ANNUAL_PREMIUM"]

    # Encode categoricals
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + "_ENCODED"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    encoded_cols = [c + "_ENCODED" for c in categorical_cols]
    feature_cols = numeric_cols + encoded_cols

    # Scale features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)
    y = df[label_col]

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

    return {
        "model": model,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_cols": feature_cols,
        "metrics": metrics,
    }


def train_model(session: Session, dataset_name: str, dataset_version: str) -> dict:
    """
    Submit training as an ML Job and wait for the result.

    Returns dict with model artifacts and metrics.
    """
    print(f"Submitting training job to compute pool {COMPUTE_POOL}...")
    job = train_model_remote(session, dataset_name, dataset_version)
    print(f"ML Job submitted: {job.id}")

    result = job.result()
    print(f"Training complete. Test R2: {result['metrics']['test_r2']}")
    return result


def save_model_artifacts(session: Session, artifacts: dict, run_id: str) -> str:
    """Save model artifacts to stage. Returns the stage path."""
    model_pkl = cp.dumps(artifacts)
    artifact_path = f"{ARTIFACTS_STAGE}/{run_id}/model_artifacts.pkl"
    session.file.put_stream(io.BytesIO(model_pkl), artifact_path, overwrite=True)
    print(f"Model artifacts saved to {artifact_path}")
    return artifact_path


def load_model_artifacts(session: Session, artifact_path: str) -> dict:
    """Load model artifacts from stage."""
    with session.file.get_stream(artifact_path, decompress=True) as stream:
        return cp.loads(stream.read())


# =============================================================================
# Evaluation & Registry
# =============================================================================

def get_registry(session: Session) -> Registry:
    """Get the model registry."""
    return Registry(
        session=session,
        database_name=PIPELINE_DB,
        schema_name=DATA_SCHEMA,
    )


def register_model(
    session: Session,
    model_obj: Any,
    version_name: str,
    dataset: Dataset,
    metrics: dict,
    scaler: Any = None,
    label_encoders: dict = None,
) -> ModelVersion:
    """Register model in the Snowflake Model Registry."""
    registry = get_registry(session)

    input_cols = NUMERIC_COLS + CATEGORICAL_COLS
    sample_df = dataset.read.to_snowpark_dataframe().limit(100)
    signature = infer_signature(
        input_data=sample_df.select(input_cols),
        output_data=sample_df.select(LABEL_COL).with_column_renamed(LABEL_COL, "PREDICTED_PREMIUM"),
    )

    mv = registry.log_model(
        model_name=MODEL_NAME,
        model=model_obj,
        version_name=version_name,
        signatures={"predict": signature},
        comment=f"XGBRegressor for car insurance premium prediction. Version: {version_name}",
        options={"enable_explainability": False},
    )

    for metric_name, value in metrics.items():
        mv.set_metric(metric_name=metric_name, value=value)

    print(f"Registered model {mv.fully_qualified_model_name} version {mv.version_name}")
    return mv


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
# Inference
# =============================================================================

def run_inference(session: Session, input_table: str, output_table: str) -> int:
    """
    Run batch inference using the production model version.

    Returns the number of predictions made.
    """
    session.use_database(PIPELINE_DB)
    session.use_schema(DATA_SCHEMA)

    registry = get_registry(session)
    mv = registry.get_model(MODEL_NAME).default

    input_df = session.table(input_table)
    print(f"Running inference on {input_df.count()} rows from {input_table}")

    predictions = mv.run(input_df, function_name="predict")

    predictions.write.save_as_table(output_table, mode="overwrite")
    pred_count = session.table(output_table).count()
    print(f"Predictions saved to {output_table}: {pred_count} rows")
    return pred_count


@remote(COMPUTE_POOL, stage_name=JOB_STAGE)
def run_inference_remote(session: Session, input_table: str, output_table: str) -> int:
    """
    Run batch inference on the compute pool via ML Job.

    Self-contained: loads the production model from registry, runs predictions,
    and saves results. All logic inside the function body for clean serialization.
    """
    from snowflake.ml.registry import Registry

    pipeline_db = "CC_INSURANCE_PIPELINE"
    data_schema = "DATA"
    model_name = "CAR_INSURANCE_PRICING_MODEL"

    session.use_database(pipeline_db)
    session.use_schema(data_schema)

    registry = Registry(
        session=session,
        database_name=pipeline_db,
        schema_name=data_schema,
    )
    mv = registry.get_model(model_name).default

    input_df = session.table(input_table)
    row_count = input_df.count()
    print(f"Running inference on {row_count} rows from {input_table}")

    predictions = mv.run(input_df, function_name="predict")

    predictions.write.save_as_table(output_table, mode="overwrite")
    pred_count = session.table(output_table).count()
    print(f"Predictions saved to {output_table}: {pred_count} rows")
    return pred_count


def submit_inference_job(session: Session, input_table: str, output_table: str) -> int:
    """
    Submit batch inference as an ML Job on the compute pool and wait for result.

    Returns the number of predictions made.
    """
    print(f"Submitting inference job to compute pool {COMPUTE_POOL}...")
    job = run_inference_remote(session, input_table, output_table)
    print(f"ML Job submitted: {job.id}")

    result = job.result()
    print(f"Inference complete: {result} predictions")
    return result


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
