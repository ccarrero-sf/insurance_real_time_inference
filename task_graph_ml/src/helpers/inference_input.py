"""
Helper functions for building inference input DataFrames.

Extracted from modeling.run_inference_job() to keep the ML Job function concise.
"""

from datetime import datetime

from snowflake.ml.feature_store import FeatureStore, CreationMode
from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark import functions as F


def build_inference_input(
    session: Session,
    pipeline_db: str,
    data_schema: str,
    feature_store_db: str,
    feature_store_schema: str,
    warehouse: str,
) -> DataFrame:
    """
    Build the enriched inference input by joining POLICIES + CUSTOMERS + Feature View.

    Returns a Snowpark DataFrame with the columns the model expects.
    """
    current_year = datetime.now().year

    policies_df = session.table(f"{pipeline_db}.{data_schema}.POLICIES")
    customers_df = session.table(f"{pipeline_db}.{data_schema}.CUSTOMERS").select(
        "CUSTOMER_ID", "AGE", "GENDER", "STATE",
        "YEARS_LICENSED", "CLAIMS_HISTORY", "CREDIT_SCORE",
    )

    # Join policies with customers and compute CAR_AGE
    input_df = (
        policies_df
        .join(customers_df, on="CUSTOMER_ID")
        .with_column("CAR_AGE", F.lit(current_year) - F.col("CAR_YEAR"))
    )

    # Join with feature view to get derived features
    fs = FeatureStore(
        session=session,
        database=feature_store_db,
        name=feature_store_schema,
        default_warehouse=warehouse,
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )
    fv = fs.get_feature_view("CUSTOMER_RISK_FEATURES", "v1")
    feature_df = fv.feature_df.select(
        "CUSTOMER_ID", "RISK_SCORE", "AVG_CLAIMS_PER_YEAR",
        "TOTAL_POLICIES", "AVG_CAR_AGE", "AVG_KILOMETERS",
        "TOTAL_CAR_VALUE", "AVG_DEDUCTIBLE",
    )
    input_df = input_df.join(feature_df, on="CUSTOMER_ID")

    # Select only the columns the model expects
    input_cols = [
        "CAR_AGE", "KILOMETERS", "ENGINE_SIZE", "ESTIMATED_CAR_VALUE",
        "AGE", "YEARS_LICENSED", "CLAIMS_HISTORY", "CREDIT_SCORE",
        "RISK_SCORE", "AVG_CLAIMS_PER_YEAR", "TOTAL_POLICIES", "AVG_CAR_AGE",
        "AVG_KILOMETERS", "TOTAL_CAR_VALUE", "AVG_DEDUCTIBLE",
        "CAR_MAKE", "CAR_MODEL", "FUEL_TYPE", "TRANSMISSION",
        "COVERAGE_TYPE", "GENDER", "STATE",
    ]
    return input_df.select(input_cols)
