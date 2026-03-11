"""
Upload Pipeline Source Code to Snowflake Stage.

This script uploads the entire src/ directory from task_graph_ml_directory/
to @CC_INSURANCE_PIPELINE.PIPELINE_SUBMIT_FILE.CODE_STAGE so that
StoredProcedure-based Snowflake Tasks can reference them via imports.

Note: ML Job scripts (jobs/) are NOT uploaded here. They are automatically
uploaded by MLJobDefinition.register() to JOB_STAGE during DAG deployment.

Must be run BEFORE deploying the DAG (deploy_dag.py).

Usage:
    python upload_code.py [--verify]

    --verify: After uploading, list stage contents to verify.
"""

import argparse
import os

from snowflake.snowpark import Session

# Source directory: task_graph_ml_directory/src/, one level up from scripts/
SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

# Directories to skip (handled by MLJobDefinition.register automatically)
SKIP_DIRS = {"jobs", "__pycache__"}

# Stage path
STAGE_DB = "CC_INSURANCE_PIPELINE"
STAGE_SCHEMA = "PIPELINE_SUBMIT_FILE"
STAGE_NAME = "CODE_STAGE"
STAGE_PATH = f"@{STAGE_DB}.{STAGE_SCHEMA}.{STAGE_NAME}"


def upload_code(session: Session, verify: bool = False) -> None:
    """Upload the entire src/ directory tree to CODE_STAGE, preserving structure."""
    session.use_database(STAGE_DB)
    session.use_schema(STAGE_SCHEMA)

    print(f"Uploading source directory to {STAGE_PATH}")
    print(f"Source directory: {SOURCE_DIR}")
    print()

    uploaded = 0
    for dirpath, dirnames, filenames in os.walk(SOURCE_DIR):
        # Skip directories that don't need to go to CODE_STAGE
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            if not filename.endswith(".py") and not filename.endswith(".sql"):
                continue

            filepath = os.path.join(dirpath, filename)
            # Compute the relative subdirectory from SOURCE_DIR
            rel_dir = os.path.relpath(dirpath, SOURCE_DIR)
            if rel_dir == ".":
                stage_target = STAGE_PATH
            else:
                stage_target = f"{STAGE_PATH}/{rel_dir}"

            result = session.file.put(
                local_file_name=filepath,
                stage_location=stage_target,
                auto_compress=False,
                overwrite=True,
            )
            status = result[0].status if result else "unknown"
            rel_path = os.path.relpath(filepath, SOURCE_DIR)
            print(f"  {rel_path} -> {stage_target}/{filename} [{status}]")
            uploaded += 1

    print(f"\nUploaded {uploaded} files")
    print("(ML Job scripts in jobs/ are uploaded automatically by MLJobDefinition.register())")

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
