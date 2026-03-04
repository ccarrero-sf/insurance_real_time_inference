"""
Upload Pipeline Source Code to Snowflake Stage.

This script uploads all Python source files from task_graph/src/ to
@CC_INSURANCE_PIPELINE.PIPELINE_STG.CODE_STAGE so that Snowflake Tasks
can reference them via imports.

Must be run BEFORE deploying the DAG (deploy_dag.py).

Usage:
    python upload_code.py [--verify]

    --verify: After uploading, list stage contents to verify.
"""

import argparse
import os
import sys

from snowflake.snowpark import Session

# Files to upload to the code stage
# Source files live in task_graph/src/, one level up from scripts/
SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
SOURCE_FILES = [
    "constants.py",
    "data_ops.py",
    "feature_ops.py",
    "modeling.py",
    "pipeline_tasks.py",
]

# Stage path (without @)
STAGE_DB = "CC_INSURANCE_PIPELINE"
STAGE_SCHEMA = "PIPELINE_STG"
STAGE_NAME = "CODE_STAGE"
STAGE_PATH = f"@{STAGE_DB}.{STAGE_SCHEMA}.{STAGE_NAME}"


def upload_code(session: Session, verify: bool = False) -> None:
    """Upload all source files to CODE_STAGE."""
    session.use_database(STAGE_DB)
    session.use_schema(STAGE_SCHEMA)

    print(f"Uploading source files to {STAGE_PATH}")
    print(f"Source directory: {SOURCE_DIR}")
    print()

    uploaded = []
    for filename in SOURCE_FILES:
        filepath = os.path.join(SOURCE_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  WARNING: {filename} not found at {filepath}, skipping")
            continue

        result = session.file.put(
            local_file_name=filepath,
            stage_location=STAGE_PATH,
            auto_compress=False,
            overwrite=True,
        )
        status = result[0].status if result else "unknown"
        print(f"  {filename} -> {STAGE_PATH}/{filename} [{status}]")
        uploaded.append(filename)

    print(f"\nUploaded {len(uploaded)}/{len(SOURCE_FILES)} files")

    if verify:
        print(f"\nStage contents ({STAGE_PATH}):")
        files = session.sql(f"LIST {STAGE_PATH}").collect()
        for f in files:
            print(f"  {f['name']}  (size: {f['size']}, md5: {f['md5']})")

    print("\nUpload complete. You can now deploy the DAG with:")
    print("  python deploy_dag.py")


def get_session() -> Session:
    """Create a Snowpark session from the default connection."""
    return Session.builder.config("connection_name", "keypair").create()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload pipeline code to Snowflake stage")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="List stage contents after upload to verify",
    )
    args = parser.parse_args()

    session = get_session()
    print(f"Connected as: {session.get_current_role()}")

    try:
        upload_code(session, verify=args.verify)
    finally:
        session.close()
