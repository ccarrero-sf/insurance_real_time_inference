-- =============================================================================
-- Car Insurance ML Pipeline (submit_directory version) - Infrastructure Setup
-- =============================================================================
-- This script creates all Snowflake objects needed for the ML pipeline.
-- Uses a SEPARATE database (CC_INSURANCE_PIPELINE) from the notebook's
-- CC_ML_INSURANCE to keep pipeline infrastructure isolated.
--
-- This version uses MLJobDefinition.register() with submit_directory pattern.
-- Schema: PIPELINE_SUBMIT_FILE (separate from the original PIPELINE schema)
--
-- Prerequisites:
--   - A compute pool must exist (or be created below)
--   - The executing role needs CREATE DATABASE privileges
--
-- Usage:
--   Run this script once to set up the environment.
--   Adjust ROLE_NAME, WAREHOUSE, and COMPUTE_POOL as needed.
-- =============================================================================

-- -------------------------
-- Configuration variables
-- -------------------------
SET ROLE_NAME = 'SPCS_PSE_ROLE';       -- Change to your role
SET WAREHOUSE = 'COMPUTE_WH';          -- Change to your warehouse
SET COMPUTE_POOL = 'DEMO_POOL';        -- Change to your compute pool
SET DB_NAME = 'CC_INSURANCE_PIPELINE';

-- -------------------------
-- Database and Schemas
-- -------------------------
CREATE DATABASE IF NOT EXISTS IDENTIFIER($DB_NAME);

-- PIPELINE_SUBMIT_FILE schema: tasks, stages, DAG artifacts
CREATE SCHEMA IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_SUBMIT_FILE');

-- DATA schema: Feature Store, Model Registry, raw data tables
CREATE SCHEMA IF NOT EXISTS IDENTIFIER($DB_NAME || '.DATA');

-- -------------------------
-- Stages
-- -------------------------
-- CODE_STAGE: holds Python source code uploaded from local
CREATE STAGE IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_SUBMIT_FILE.CODE_STAGE')
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- JOB_STAGE: ML Job payloads (used by MLJobDefinition.register)
CREATE STAGE IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_SUBMIT_FILE.JOB_STAGE')
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- DAG_STAGE: DAG-level artifacts and run metadata
CREATE STAGE IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_SUBMIT_FILE.DAG_STAGE')
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- ARTIFACTS_STAGE: model pickles, run artifacts, intermediate results
CREATE STAGE IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_SUBMIT_FILE.ARTIFACTS_STAGE')
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- -------------------------
-- Image Repository (for ML Jobs on SPCS)
-- -------------------------
CREATE IMAGE REPOSITORY IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_SUBMIT_FILE.ML_IMAGE_REPO');

-- -------------------------
-- Grants
-- -------------------------
-- Grant usage on database and schemas
GRANT USAGE ON DATABASE IDENTIFIER($DB_NAME) TO ROLE IDENTIFIER($ROLE_NAME);
GRANT ALL ON SCHEMA IDENTIFIER($DB_NAME || '.PIPELINE_SUBMIT_FILE') TO ROLE IDENTIFIER($ROLE_NAME);
GRANT ALL ON SCHEMA IDENTIFIER($DB_NAME || '.DATA') TO ROLE IDENTIFIER($ROLE_NAME);

-- Grant usage on stages
GRANT READ, WRITE ON STAGE IDENTIFIER($DB_NAME || '.PIPELINE_SUBMIT_FILE.CODE_STAGE') TO ROLE IDENTIFIER($ROLE_NAME);
GRANT READ, WRITE ON STAGE IDENTIFIER($DB_NAME || '.PIPELINE_SUBMIT_FILE.JOB_STAGE') TO ROLE IDENTIFIER($ROLE_NAME);
GRANT READ, WRITE ON STAGE IDENTIFIER($DB_NAME || '.PIPELINE_SUBMIT_FILE.DAG_STAGE') TO ROLE IDENTIFIER($ROLE_NAME);
GRANT READ, WRITE ON STAGE IDENTIFIER($DB_NAME || '.PIPELINE_SUBMIT_FILE.ARTIFACTS_STAGE') TO ROLE IDENTIFIER($ROLE_NAME);

-- Grant task execution (requires ACCOUNTADMIN or appropriate privileges)
-- GRANT EXECUTE TASK ON ACCOUNT TO ROLE IDENTIFIER($ROLE_NAME);

-- Grant compute pool usage
-- GRANT USAGE ON COMPUTE POOL IDENTIFIER($COMPUTE_POOL) TO ROLE IDENTIFIER($ROLE_NAME);

-- Grant warehouse usage
GRANT USAGE ON WAREHOUSE IDENTIFIER($WAREHOUSE) TO ROLE IDENTIFIER($ROLE_NAME);

-- -------------------------
-- Verify setup
-- -------------------------
SHOW SCHEMAS IN DATABASE IDENTIFIER($DB_NAME);
SHOW STAGES IN SCHEMA IDENTIFIER($DB_NAME || '.PIPELINE_SUBMIT_FILE');

SELECT 'Infrastructure setup complete for ' || $DB_NAME AS STATUS;
