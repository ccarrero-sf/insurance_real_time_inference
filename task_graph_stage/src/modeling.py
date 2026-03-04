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
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from snowflake.ml.dataset import Dataset, load_dataset
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
    ARTIFACTS_STAGE,
    MODEL_NAME,
    METRIC_NAME,
    METRIC_THRESHOLD,
    METRIC_TOLERANCE,
    WAREHOUSE,
    FEATURE_STORE_DB,
    FEATURE_STORE_SCHEMA,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
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

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


@remote(COMPUTE_POOL, stage_name=JOB_STAGE)
def train_model_remote(session: Session, dataset_name: str, dataset_version: str):
    """
    Train an sklearn Pipeline (OrdinalEncoder + StandardScaler + XGBRegressor)
    on SPCS via ML Job.

    The pipeline accepts raw Feature Store columns (numeric + categorical)
    and handles all transformations internally, so mv.run() works on raw data.
    """
    import xgboost as xgb
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    from snowflake.ml.dataset import load_dataset

    ds = load_dataset(session, dataset_name, dataset_version)
    df = ds.read.to_pandas()
    print(f"[TRAIN] Loaded dataset: {dataset_name} v{dataset_version} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"[TRAIN] Columns: {list(df.columns)}")

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
    feature_cols = numeric_cols + categorical_cols

    # Build raw feature matrix (no manual encoding/scaling)
    X = df[feature_cols].copy()
    # Ensure categoricals are strings for OrdinalEncoder
    for col in categorical_cols:
        X[col] = X[col].astype(str)
    y = df[label_col]
    print(f"[TRAIN] Feature matrix shape: {X.shape}, target shape: {y.shape}")
    print(f"[TRAIN] Feature columns ({len(feature_cols)}): {feature_cols}")

    # Train/test split on raw data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    print(f"[TRAIN] Train/test split: train={len(X_train)}, test={len(X_test)}")

    # Build sklearn Pipeline with ColumnTransformer
    print("[TRAIN] Building sklearn Pipeline: ColumnTransformer(StandardScaler + OrdinalEncoder) -> XGBRegressor")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
        ],
        remainder="drop",
    )

    pipeline = SklearnPipeline([
        ("preprocessor", preprocessor),
        ("regressor", xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )),
    ])

    # Train the full pipeline on raw data
    print("[TRAIN] Fitting pipeline on training data...")
    pipeline.fit(
        X_train, y_train,
        regressor__eval_set=[(preprocessor.fit_transform(X_test), y_test)],
        regressor__verbose=False,
    )

    # Evaluate
    print("[TRAIN] Evaluating model on train and test sets...")
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

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
        "model": pipeline,
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
) -> ModelVersion:
    """
    Register model in the Snowflake Model Registry.

    The model_obj is an sklearn Pipeline that includes preprocessing
    (OrdinalEncoder + StandardScaler), so the sample_input_data should
    be raw Feature Store columns (no manual transformation needed).
    """
    registry = get_registry(session)
    print(f"[REGISTER] Registry: {PIPELINE_DB}.{DATA_SCHEMA}")

    # Use raw feature columns as sample input — the Pipeline handles
    # encoding and scaling internally.
    raw_sample = dataset.read.to_pandas().head(100)
    sample_input = raw_sample[FEATURE_COLS].copy()
    for col in CATEGORICAL_COLS:
        sample_input[col] = sample_input[col].astype(str)
    print(f"[REGISTER] Sample input shape: {sample_input.shape}, columns: {list(sample_input.columns)}")

    print(f"[REGISTER] Logging model '{MODEL_NAME}' version '{version_name}'...")
    mv = registry.log_model(
        model_name=MODEL_NAME,
        model=model_obj,
        version_name=version_name,
        sample_input_data=sample_input,
        comment=f"sklearn Pipeline (OrdinalEncoder+StandardScaler+XGBRegressor) for car insurance premium prediction. Version: {version_name}",
        options={"enable_explainability": False},
    )

    for metric_name, value in metrics.items():
        mv.set_metric(metric_name=metric_name, value=value)

    print(f"[REGISTER] Registered model {mv.fully_qualified_model_name} version {mv.version_name}")
    return mv


def promote_model(session: Session, mv: ModelVersion) -> None:
    """Set model version as the default (production) version."""
    registry = get_registry(session)
    base_model = registry.get_model(MODEL_NAME)
    print(f"[PROMOTE] Current default version: {base_model.default.version_name if base_model.default else 'None'}")
    base_model.default = mv
    print(f"[PROMOTE] Promoted {mv.version_name} to default (production)")


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

    Promotes if the new model meets the threshold AND is within METRIC_TOLERANCE
    of the production model (i.e., doesn't need to strictly beat it).
    Returns "promote_model" or "send_alert".
    """
    current_score = new_metrics.get(METRIC_NAME, 0)
    print(f"[QUALITY] New model {METRIC_NAME}: {current_score}")
    print(f"[QUALITY] All new model metrics: {new_metrics}")

    prod_mv = get_production_model(session)
    if prod_mv is None:
        print("[QUALITY] No production model exists. Decision: promote_model")
        return "promote_model"

    print(f"[QUALITY] Production model version: {prod_mv.version_name}")

    try:
        prod_metrics = prod_mv.get_metric(METRIC_NAME)
        prod_score = prod_metrics if isinstance(prod_metrics, (int, float)) else 0
    except Exception as e:
        print(f"[QUALITY] Could not read production metric: {e}")
        prod_score = 0

    diff = current_score - prod_score
    print(f"[QUALITY] Production model {METRIC_NAME}: {prod_score}")
    print(f"[QUALITY] Score difference (new - prod): {diff:.4f}")
    print(f"[QUALITY] Threshold: {METRIC_THRESHOLD}, Tolerance: {METRIC_TOLERANCE}")

    meets_threshold = current_score >= METRIC_THRESHOLD
    within_tolerance = current_score >= (prod_score - METRIC_TOLERANCE)

    print(f"[QUALITY] Meets threshold ({current_score} >= {METRIC_THRESHOLD}): {meets_threshold}")
    print(f"[QUALITY] Within tolerance ({current_score} >= {prod_score} - {METRIC_TOLERANCE} = {prod_score - METRIC_TOLERANCE:.4f}): {within_tolerance}")

    if meets_threshold and within_tolerance:
        print("[QUALITY] Decision: promote_model (meets threshold and within tolerance of production)")
        return "promote_model"
    else:
        print("[QUALITY] Decision: send_alert (does not meet promotion criteria)")
        return "send_alert"


# =============================================================================
# Inference
# =============================================================================

def run_inference(session: Session, input_table: str, output_table: str) -> int:
    """
    Run batch inference using the production model and Feature Store data.

    1. Builds a spine DataFrame from the input table (policies + customer keys)
    2. Retrieves enriched features from the Feature Store
    3. Calls mv.run() on the enriched data — the registered sklearn Pipeline
       handles all preprocessing (OrdinalEncoder + StandardScaler) internally
    4. Saves predictions to the output table

    Returns the number of predictions made.
    """
    from snowflake.ml.feature_store import FeatureStore, CreationMode
    from snowflake.snowpark import functions as F

    print(f"[INFERENCE] Starting inference: input={input_table}, output={output_table}")

    session.use_database(PIPELINE_DB)
    session.use_schema(DATA_SCHEMA)

    # Get production model
    print(f"[INFERENCE] Loading production model '{MODEL_NAME}' from registry...")
    registry = get_registry(session)
    mv = registry.get_model(MODEL_NAME).default
    print(f"[INFERENCE] Using production model: {mv.version_name}")

    # Get Feature Store and feature view
    print(f"[INFERENCE] Connecting to Feature Store: {FEATURE_STORE_DB}.{FEATURE_STORE_SCHEMA}")
    fs = FeatureStore(
        session=session,
        database=FEATURE_STORE_DB,
        name=FEATURE_STORE_SCHEMA,
        default_warehouse=WAREHOUSE,
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )
    fv = fs.get_feature_view(FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION)
    print(f"[INFERENCE] Feature view: {FEATURE_VIEW_NAME} v{FEATURE_VIEW_VERSION}")

    # Build spine DataFrame: policies joined with customer-level columns
    # needed by the model but not in the feature view
    print("[INFERENCE] Building spine DataFrame (POLICIES + CUSTOMERS join)...")
    current_year = datetime.now().year
    spine_df = (
        session.table(input_table)
        .join(
            session.table(f"{PIPELINE_DB}.{DATA_SCHEMA}.CUSTOMERS").select(
                "CUSTOMER_ID", "GENDER", "STATE",
            ),
            on="CUSTOMER_ID",
        )
        .with_column("CAR_AGE", F.lit(current_year) - F.col("CAR_YEAR"))
        .select(
            "CUSTOMER_ID", "POLICY_ID",
            "CAR_MAKE", "CAR_MODEL", "CAR_AGE", "KILOMETERS",
            "ENGINE_SIZE", "FUEL_TYPE", "TRANSMISSION", "COVERAGE_TYPE",
            "ESTIMATED_CAR_VALUE", "GENDER", "STATE",
        )
    )

    print(f"[INFERENCE] Spine DataFrame: {spine_df.count()} rows")

    # Retrieve enriched features from Feature Store
    print("[INFERENCE] Retrieving enriched features from Feature Store...")
    enriched_df = fs.retrieve_feature_values(
        spine_df=spine_df,
        features=[fv],
    )

    print(f"[INFERENCE] Enriched DataFrame columns ({len(enriched_df.columns)}): {enriched_df.columns}")

    # Run predictions — the model Pipeline handles encoding + scaling internally
    print("[INFERENCE] Running mv.run() predictions...")
    predictions = mv.run(enriched_df, function_name="predict")

    print(f"[INFERENCE] Saving predictions to {output_table}...")
    predictions.write.save_as_table(output_table, mode="overwrite")
    pred_count = session.table(output_table).count()
    print(f"[INFERENCE] Predictions saved: {pred_count} rows")
    return pred_count


# =============================================================================
# Cleanup
# =============================================================================

def cleanup(session: Session, expiry_days: int = 7) -> None:
    """Clean up old model versions and datasets beyond expiry."""
    print(f"[CLEANUP] Starting cleanup (expiry_days={expiry_days})")
    registry = get_registry(session)
    try:
        base_model = registry.get_model(MODEL_NAME)
        versions = base_model.versions()
        default_version = base_model.default.version_name
        print(f"[CLEANUP] Model '{MODEL_NAME}' has {len(versions)} version(s), default: {default_version}")

        cutoff = datetime.now() - timedelta(days=expiry_days)
        print(f"[CLEANUP] Cutoff date: {cutoff.isoformat()}")
        deleted_count = 0
        for v in versions:
            if v.version_name == default_version:
                print(f"[CLEANUP] Skipping default version: {v.version_name}")
                continue
            try:
                created = v.created_on
                if created and created < cutoff:
                    print(f"[CLEANUP] Deleting expired version: {v.version_name} (created: {created})")
                    base_model.delete_version(v.version_name)
                    deleted_count += 1
                else:
                    print(f"[CLEANUP] Keeping version: {v.version_name} (created: {created})")
            except Exception as e:
                print(f"[CLEANUP] Skipping version {v.version_name}: {e}")
        print(f"[CLEANUP] Deleted {deleted_count} expired version(s)")
    except Exception as e:
        print(f"[CLEANUP] Cleanup skipped: {e}")
