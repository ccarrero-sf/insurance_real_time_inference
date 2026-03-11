"""
Helper functions for model artifact management in the training pipeline.

Extracted from modeling.train_model_job() to keep the ML Job function concise.
Contains the CustomModel class and artifact serialisation helpers.
"""

import io
import os
import pickle
import tempfile

import cloudpickle as cp
import pandas as pd
from snowflake.ml.model import custom_model
from snowflake.snowpark import Session


# =========================================================================
# Custom Model class (embedded feature engineering in predict)
# =========================================================================

class CarInsurancePricingModel(custom_model.CustomModel):
    """XGBoost pricing model with built-in categorical encoding and scaling."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

        with open(context.path("xgb_model.ubj"), "rb") as f:
            self.model = pickle.load(f)
        with open(context.path("scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        with open(context.path("label_encoders.pkl"), "rb") as f:
            self.label_encoders = pickle.load(f)
        with open(context.path("feature_cols.pkl"), "rb") as f:
            self.feature_cols = pickle.load(f)

    @custom_model.inference_api
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()

        _categorical_cols = [
            "CAR_MAKE", "CAR_MODEL", "FUEL_TYPE", "TRANSMISSION",
            "COVERAGE_TYPE", "GENDER", "STATE",
        ]

        for col in _categorical_cols:
            if col in df.columns:
                le = self.label_encoders[col]
                df[f"{col}_ENCODED"] = df[col].apply(
                    lambda x, _le=le: _le.transform([str(x)])[0]
                    if str(x) in _le.classes_ else 0
                )

        available_cols = [c for c in self.feature_cols if c in df.columns]
        X = df[available_cols]
        X_scaled = self.scaler.transform(X)

        return pd.DataFrame(X_scaled, columns=available_cols)

    @custom_model.inference_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        import numpy as _np

        X_scaled = self.transform(input_df)

        predictions = self.model.predict(X_scaled.values)
        predictions = _np.maximum(predictions, 400)
        predictions = _np.minimum(predictions, 5000)

        return pd.DataFrame({"PREDICTED_PREMIUM": predictions})


# =========================================================================
# Artifact helpers
# =========================================================================

def save_artifacts_to_stage(
    session: Session,
    artifacts: dict,
    stage_path: str,
    version_name: str,
) -> str:
    """Pickle model artifacts dict and upload to a Snowflake stage."""
    model_pkl = cp.dumps(artifacts)
    artifact_path = f"{stage_path}/{version_name}/model_artifacts.pkl"
    session.file.put_stream(io.BytesIO(model_pkl), artifact_path, overwrite=True)
    print(f"Model artifacts saved to {artifact_path}")
    return artifact_path


def build_model_context(
    model,
    scaler,
    label_encoders: dict,
    feature_cols: list[str],
) -> tuple[CarInsurancePricingModel, custom_model.ModelContext]:
    """
    Save artifacts to temp files, build a ModelContext, and return an
    instantiated CarInsurancePricingModel ready for registry logging.
    """
    tmpdir = tempfile.mkdtemp()

    xgb_path = os.path.join(tmpdir, "xgb_model.ubj")
    with open(xgb_path, "wb") as f:
        pickle.dump(model, f)

    scaler_path = os.path.join(tmpdir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    le_path = os.path.join(tmpdir, "label_encoders.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(label_encoders, f)

    fc_path = os.path.join(tmpdir, "feature_cols.pkl")
    with open(fc_path, "wb") as f:
        pickle.dump(feature_cols, f)

    model_context = custom_model.ModelContext(
        artifacts={
            "xgb_model.ubj": xgb_path,
            "scaler.pkl": scaler_path,
            "label_encoders.pkl": le_path,
            "feature_cols.pkl": fc_path,
        }
    )

    pricing_model = CarInsurancePricingModel(model_context)
    print("CustomModel instantiated with embedded feature engineering")
    return pricing_model, model_context
