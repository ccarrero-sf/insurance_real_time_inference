import os

# =============================================================================
# Car Insurance ML Pipeline (ML Jobs version) - Configuration Constants
# =============================================================================
# All values can be overridden via environment variables.
# This version uses ML Jobs (@remote) as direct DAG task definitions
# instead of wrapping them in StoredProcedureCalls.
# =============================================================================

# -- Snowflake Role & Warehouse --
ROLE_NAME = os.getenv("SNOWFLAKE_ROLE", "SPCS_PSE_ROLE")
WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
COMPUTE_POOL = os.getenv("SNOWFLAKE_COMPUTE_POOL", "DEMO_POOL")

# -- Database & Schemas --
PIPELINE_DB = os.getenv("PIPELINE_DB", "CC_INSURANCE_PIPELINE")
PIPELINE_SCHEMA = os.getenv("PIPELINE_SCHEMA", "PIPELINE_MLJOBS")
DATA_SCHEMA = os.getenv("DATA_SCHEMA", "DATA")

# -- Source database (notebook's existing DB, for reference only) --
SOURCE_DB = os.getenv("SOURCE_DB", "CC_ML_INSURANCE")
SOURCE_SCHEMA = os.getenv("SOURCE_SCHEMA", "CAR_PRICING")

# -- Stages --
CODE_STAGE = f"@{PIPELINE_DB}.{PIPELINE_SCHEMA}.CODE_STAGE"
JOB_STAGE = f"@{PIPELINE_DB}.{PIPELINE_SCHEMA}.JOB_STAGE"
DAG_STAGE = f"@{PIPELINE_DB}.{PIPELINE_SCHEMA}.DAG_STAGE"
ARTIFACTS_STAGE = f"@{PIPELINE_DB}.{PIPELINE_SCHEMA}.ARTIFACTS_STAGE"

# -- Feature Store --
FEATURE_STORE_DB = PIPELINE_DB
FEATURE_STORE_SCHEMA = DATA_SCHEMA
FEATURE_STORE_NAME = f"{FEATURE_STORE_DB}.{FEATURE_STORE_SCHEMA}"

# -- Tables --
CUSTOMERS_TABLE = f"{PIPELINE_DB}.{DATA_SCHEMA}.CUSTOMERS"
POLICIES_TABLE = f"{PIPELINE_DB}.{DATA_SCHEMA}.POLICIES"

# -- Model & Dataset --
MODEL_NAME = os.getenv("MODEL_NAME", "CAR_INSURANCE_PRICING_MODEL")
DATASET_NAME = f"{PIPELINE_DB}.{DATA_SCHEMA}.CAR_INSURANCE_TRAINING_DATASET"
FEATURE_VIEW_NAME = "CUSTOMER_RISK_FEATURES"
FEATURE_VIEW_VERSION = "v1"

# -- ML Pipeline Config --
METRIC_NAME = os.getenv("METRIC_NAME", "test_r2")
METRIC_THRESHOLD = float(os.getenv("METRIC_THRESHOLD", "0.7"))

# -- Data Generation --
N_CUSTOMERS = int(os.getenv("N_CUSTOMERS", "5000"))
N_POLICIES = int(os.getenv("N_POLICIES", "8000"))
