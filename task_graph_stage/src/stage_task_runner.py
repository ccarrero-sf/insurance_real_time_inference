"""
Generic Stage Task Runner for Car Insurance ML Pipeline.

Dynamically imports and executes task functions from modules loaded via
@CODE_STAGE imports. This decouples the DAG definition from the task code:
updating .py files on the stage takes effect on the next DAG run without
redeploying the DAG.

Usage in DAG definition:
    Each DAGTask uses a thin wrapper that calls:
        run_task_from_stage(session, "pipeline_tasks", "task_ingest_data")
"""

import importlib

from snowflake.snowpark import Session


def run_task_from_stage(session: Session, module_name: str, func_name: str) -> str:
    """
    Import a module (available via stage imports) and call the named function.

    Args:
        session: Active Snowpark session provided by the task runtime.
        module_name: Python module to import (e.g. "pipeline_tasks").
                     Must be listed in the task's stage imports.
        func_name: Function name to call within the module (e.g. "task_ingest_data").
                   The function must accept (session: Session) and return str.

    Returns:
        The string return value from the task function.
    """
    module = importlib.import_module(module_name)
    # Reload to pick up any changes since the module was first imported
    # in this interpreter session (handles the case where Snowflake caches
    # the import across task runs within the same warehouse session).
    importlib.reload(module)

    task_func = getattr(module, func_name)
    return task_func(session)
