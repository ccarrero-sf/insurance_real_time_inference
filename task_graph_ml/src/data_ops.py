"""
Data Operations for Car Insurance ML Pipeline (ML Jobs version).

Generates synthetic customer and policy data and uploads it to Snowflake.
The ingest_data_job function is decorated with @remote to run directly as an
ML Job DAG task on the compute pool.
"""

from snowflake.ml.jobs import remote
from snowflake.snowpark import Session

from constants import (
    PIPELINE_DB,
    PIPELINE_SCHEMA,
    COMPUTE_POOL,
    JOB_STAGE,
)


# =============================================================================
# ML Job version: runs data ingestion on SPCS compute pool as a DAG task.
#
# Decorated with @remote so it can be used directly as a DAG task definition
# (no StoredProcedureCall wrapper needed). Uses TaskContext to pass results
# to downstream tasks.
# =============================================================================

@remote(
    COMPUTE_POOL,
    stage_name=JOB_STAGE,
    database=PIPELINE_DB,
    schema=PIPELINE_SCHEMA,
    imports=[("helpers", "helpers")],
)
def ingest_data_job() -> None:
    """
    ML Job DAG task: generate synthetic data and upload to Snowflake.

    Runs on the compute pool. Uses TaskContext.set_return_value to pass
    results to downstream tasks in the DAG.
    """
    import json
    from helpers.data_generation import generate_customers, generate_policies
    from snowflake.core.task.context import TaskContext

    session = Session.builder.getOrCreate()

    pipeline_db = "CC_INSURANCE_PIPELINE"
    data_schema = "DATA"

    session.use_database(pipeline_db)
    session.use_schema(data_schema)

    n_customers = 5000
    n_policies = 8000

    # Generate synthetic data using helper functions
    customers_df = generate_customers(n_customers)
    policies_df = generate_policies(customers_df, n_policies)

    # Upload to Snowflake
    session.write_pandas(customers_df, "CUSTOMERS", auto_create_table=True, overwrite=True)
    session.write_pandas(policies_df, "POLICIES", auto_create_table=True, overwrite=True)

    cust_count = session.table("CUSTOMERS").count()
    pol_count = session.table("POLICIES").count()
    print(f"Uploaded: CUSTOMERS={cust_count} rows, POLICIES={pol_count} rows")

    result = {
        "customers_table": f"{pipeline_db}.{data_schema}.CUSTOMERS",
        "policies_table": f"{pipeline_db}.{data_schema}.POLICIES",
        "customers_count": cust_count,
        "policies_count": pol_count,
    }

    # Pass results to downstream DAG tasks via TaskContext
    ctx = TaskContext(session)
    ctx.set_return_value(json.dumps(result))
