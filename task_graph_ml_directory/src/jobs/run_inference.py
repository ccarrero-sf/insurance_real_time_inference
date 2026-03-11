"""
Standalone ML Job script: Run Inference.

Runs batch inference on SPCS compute pool using the promoted model version.
Designed to run via MLJobDefinition.register() with submit_directory pattern.

Uses TaskContext to read predecessor output and pass results downstream.
"""

import json

from snowflake.core.task.context import TaskContext
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session

from helpers.inference_input import build_inference_input


def main() -> None:
    session = Session.builder.getOrCreate()

    pipeline_db = "CC_INSURANCE_PIPELINE"
    data_schema = "DATA"
    model_name = "CAR_INSURANCE_PRICING_MODEL"
    warehouse = "COMPUTE_WH"

    session.use_database(pipeline_db)
    session.use_schema(data_schema)

    # Read predecessor output
    ctx = TaskContext(session)
    promote_result = json.loads(ctx.get_predecessor_return_value("PROMOTE_MODEL"))
    print(f"Using promoted model: {promote_result['promoted_version']}")

    output_table = f"{pipeline_db}.{data_schema}.PREDICTIONS"

    registry = Registry(
        session=session,
        database_name=pipeline_db,
        schema_name=data_schema,
    )
    mv = registry.get_model(model_name).default

    # Build enriched input using helper
    input_df = build_inference_input(
        session=session,
        pipeline_db=pipeline_db,
        data_schema=data_schema,
        feature_store_db=pipeline_db,
        feature_store_schema=data_schema,
        warehouse=warehouse,
    )

    row_count = input_df.count()
    print(f"Running inference on {row_count} rows")

    predictions = mv.run(input_df, function_name="predict")
    predictions.write.save_as_table(output_table, mode="overwrite")

    pred_count = session.table(output_table).count()
    print(f"Predictions saved to {output_table}: {pred_count} rows")

    # Pass results to downstream tasks
    result = {
        "predictions_count": pred_count,
        "output_table": output_table,
        "model_version": promote_result["promoted_version"],
    }
    ctx.set_return_value(json.dumps(result))


if __name__ == "__main__":
    main()
