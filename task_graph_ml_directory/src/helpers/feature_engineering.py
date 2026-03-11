"""
Helper functions for feature engineering in the training pipeline.

Extracted from modeling.train_model_job() to keep the ML Job function concise.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


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


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list[str] = None,
) -> tuple[pd.DataFrame, dict, list[str]]:
    """
    Label-encode categorical columns.

    Returns:
        - df with new *_ENCODED columns added
        - dict of {col_name: fitted LabelEncoder}
        - list of encoded column names
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + "_ENCODED"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    encoded_cols = [c + "_ENCODED" for c in categorical_cols]
    return df, label_encoders, encoded_cols


def scale_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on the given feature columns.

    Returns:
        - scaled DataFrame with the same column names
        - the fitted StandardScaler
    """
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)
    return scaled, scaler
