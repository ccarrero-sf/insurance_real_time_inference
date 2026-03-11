"""
Standalone ML Job script: Train Model.

Trains an XGBRegressor on SPCS compute pool, evaluates it,
saves artifacts to stage, and registers the model in the registry.
Designed to run via MLJobDefinition.register() with submit_directory pattern.

Uses TaskContext to read predecessor output and pass results downstream.
"""

import json
import sys
import types
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from snowflake.core.task.context import TaskContext
from snowflake.ml.dataset import load_dataset
from snowflake.ml.model.model_signature import infer_signature
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session

from helpers.feature_engineering import (
    CATEGORICAL_COLS,
    LABEL_COL,
    NUMERIC_COLS,
    encode_categoricals,
    scale_features,
)
from helpers.model_artifacts import (
    CarInsurancePricingModel,
    build_model_context,
    save_artifacts_to_stage,
)


def main() -> None:
    session = Session.builder.getOrCreate()

    pipeline_db = "CC_INSURANCE_PIPELINE"
    data_schema = "DATA"
    pipeline_schema = "PIPELINE_SUBMIT_FILE"
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
    _module_name = CarInsurancePricingModel.__module__
    if _module_name not in sys.modules:
        _mod = types.ModuleType(_module_name)
        _mod.CarInsurancePricingModel = CarInsurancePricingModel
        sys.modules[_module_name] = _mod

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


if __name__ == "__main__":
    main()
