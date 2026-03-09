"""
Feature Store Operations for Car Insurance ML Pipeline (ML Jobs version).

Sets up the Snowflake Feature Store with customer risk features,
and prepares training datasets with train/test splits.

This module runs on the warehouse (not compute pool) since Feature Store
operations don't benefit from SPCS compute.
"""

from datetime import datetime

from snowflake.ml.feature_store import FeatureStore, FeatureView, Entity, CreationMode
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F

from constants import (
    PIPELINE_DB,
    DATA_SCHEMA,
    FEATURE_STORE_DB,
    FEATURE_STORE_SCHEMA,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    DATASET_NAME,
    WAREHOUSE,
)


def get_feature_store(session: Session) -> FeatureStore:
    """Create or get the Feature Store instance."""
    fs = FeatureStore(
        session=session,
        database=FEATURE_STORE_DB,
        name=FEATURE_STORE_SCHEMA,
        default_warehouse=WAREHOUSE,
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )
    print(f"Feature Store: {FEATURE_STORE_DB}.{FEATURE_STORE_SCHEMA}")
    return fs


def setup_entity(fs: FeatureStore) -> Entity:
    """Register the Customer entity."""
    customer_entity = Entity(
        name="CUSTOMER",
        join_keys=["CUSTOMER_ID"],
        desc="Customer entity for car insurance",
    )
    fs.register_entity(customer_entity)
    print("Entity 'CUSTOMER' registered")
    return customer_entity


def create_feature_view(session: Session, fs: FeatureStore, entity: Entity) -> FeatureView:
    """Create and register the CUSTOMER_RISK_FEATURES feature view."""
    current_year = datetime.now().year

    customer_features_sql = f"""
    SELECT
        c.CUSTOMER_ID,
        c.AGE,
        c.YEARS_LICENSED,
        c.CLAIMS_HISTORY,
        c.CREDIT_SCORE,
        ROUND(
            (CASE WHEN c.AGE < 25 THEN 30 WHEN c.AGE > 65 THEN 20 ELSE 0 END) +
            (c.CLAIMS_HISTORY * 15) +
            (GREATEST(0, 40 - c.YEARS_LICENSED)) +
            (GREATEST(0, (700 - c.CREDIT_SCORE) / 10))
        , 2) AS RISK_SCORE,
        ROUND(c.CLAIMS_HISTORY / GREATEST(1, c.YEARS_LICENSED), 4) AS AVG_CLAIMS_PER_YEAR,
        CASE
            WHEN c.CREDIT_SCORE >= 750 THEN 'Excellent'
            WHEN c.CREDIT_SCORE >= 700 THEN 'Good'
            WHEN c.CREDIT_SCORE >= 650 THEN 'Fair'
            ELSE 'Poor'
        END AS CREDIT_TIER,
        COUNT(p.POLICY_ID) AS TOTAL_POLICIES,
        ROUND(AVG({current_year} - p.CAR_YEAR), 2) AS AVG_CAR_AGE,
        ROUND(AVG(p.KILOMETERS), 0) AS AVG_KILOMETERS,
        ROUND(SUM(p.ESTIMATED_CAR_VALUE), 2) AS TOTAL_CAR_VALUE,
        ROUND(AVG(p.DEDUCTIBLE), 0) AS AVG_DEDUCTIBLE,
        CURRENT_TIMESTAMP() AS UPDATED_AT
    FROM {PIPELINE_DB}.{DATA_SCHEMA}.CUSTOMERS c
    LEFT JOIN {PIPELINE_DB}.{DATA_SCHEMA}.POLICIES p ON c.CUSTOMER_ID = p.CUSTOMER_ID
    GROUP BY c.CUSTOMER_ID, c.AGE, c.YEARS_LICENSED, c.CLAIMS_HISTORY, c.CREDIT_SCORE
    """

    customer_features_df = session.sql(customer_features_sql)

    customer_fv = FeatureView(
        name=FEATURE_VIEW_NAME,
        entities=[entity],
        feature_df=customer_features_df,
        timestamp_col="UPDATED_AT",
        refresh_freq="5 minutes",
        refresh_mode="AUTO",
        desc="Customer risk and profile features for insurance pricing",
    )

    customer_fv = fs.register_feature_view(
        feature_view=customer_fv,
        version=FEATURE_VIEW_VERSION,
    )
    print(f"Feature View '{FEATURE_VIEW_NAME}' v{FEATURE_VIEW_VERSION} registered")
    return customer_fv


def setup_feature_store(session: Session) -> FeatureView:
    """
    Full Feature Store setup: create FS, register entity, create feature view.

    Returns the registered FeatureView.
    """
    session.use_database(PIPELINE_DB)
    session.use_schema(DATA_SCHEMA)

    fs = get_feature_store(session)
    entity = setup_entity(fs)
    fv = create_feature_view(session, fs, entity)
    return fv


def prepare_datasets(session: Session, feature_view: FeatureView = None) -> dict:
    """
    Prepare training dataset using Feature Store and split into train/test.

    Returns dict with dataset info for downstream tasks.
    """
    session.use_database(PIPELINE_DB)
    session.use_schema(DATA_SCHEMA)

    fs = get_feature_store(session)

    if feature_view is None:
        feature_view = fs.get_feature_view(FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION)

    current_year = datetime.now().year

    spine_df = (
        session.table(f"{PIPELINE_DB}.{DATA_SCHEMA}.POLICIES")
        .join(
            session.table(f"{PIPELINE_DB}.{DATA_SCHEMA}.CUSTOMERS").select("CUSTOMER_ID", "GENDER", "STATE"),
            on="CUSTOMER_ID",
        )
        .with_column("CAR_AGE", F.lit(current_year) - F.col("CAR_YEAR"))
        .select(
            "CUSTOMER_ID", "CAR_MAKE", "CAR_MODEL", "CAR_AGE", "KILOMETERS",
            "ENGINE_SIZE", "FUEL_TYPE", "TRANSMISSION", "COVERAGE_TYPE",
            "ESTIMATED_CAR_VALUE", "GENDER", "STATE", "ANNUAL_PREMIUM",
            F.col("UPDATED_AT").cast("timestamp").alias("TS"),
        )
    )

    print(f"Spine DataFrame: {spine_df.count()} rows")

    training_dataset = fs.generate_dataset(
        name=DATASET_NAME,
        spine_df=spine_df,
        features=[feature_view],
        spine_timestamp_col="TS",
        spine_label_cols=["ANNUAL_PREMIUM"],
        desc="Training dataset for car insurance premium prediction",
    )

    print(f"Dataset created: {training_dataset.fully_qualified_name}")

    dataset_df = training_dataset.read.to_snowpark_dataframe()
    row_count = dataset_df.count()
    print(f"Dataset rows: {row_count}")

    return {
        "dataset_name": training_dataset.fully_qualified_name,
        "dataset_version": training_dataset.selected_version.name,
        "row_count": row_count,
    }
