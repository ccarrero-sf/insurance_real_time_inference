"""
Model Operations for Car Insurance ML Pipeline (submit_directory version).

In this version, training and inference run as standalone scripts via
MLJobDefinition.register(). See jobs/train_model.py and jobs/run_inference.py.

This module retains registry helpers, quality checks, and cleanup functions
that run on the warehouse via StoredProcedureCalls.
"""

import json
from datetime import datetime, timedelta
from typing import Optional

from snowflake.ml.model import ModelVersion
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session

from constants import (
    PIPELINE_DB,
    DATA_SCHEMA,
    PIPELINE_SCHEMA,
    MODEL_NAME,
    METRIC_NAME,
    METRIC_THRESHOLD,
)


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
