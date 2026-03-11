"""
Standalone ML Job script: Ingest Data.

Generates synthetic customer and policy data and uploads to Snowflake.
Designed to run via MLJobDefinition.register() with submit_directory pattern.

Uses TaskContext.set_return_value to pass results to downstream DAG tasks.
"""

import json

from snowflake.snowpark import Session
from snowflake.core.task.context import TaskContext

from helpers.data_generation import generate_customers, generate_policies


def main() -> None:
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


if __name__ == "__main__":
    main()
