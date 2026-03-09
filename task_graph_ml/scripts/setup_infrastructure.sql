-- =============================================================================
-- Car Insurance ML Pipeline (ML Jobs version) - Infrastructure Setup
-- =============================================================================
-- This script creates all Snowflake objects needed for the ML pipeline.
-- Uses a SEPARATE database (CC_INSURANCE_PIPELINE) from the notebook's
-- CC_ML_INSURANCE to keep pipeline infrastructure isolated.
--
-- This version adds DAG_STAGE for ML Job DAG task artifact storage.
-- Schema: PIPELINE_MLJOBS (separate from the original PIPELINE schema)
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

-- PIPELINE_MLJOBS schema: tasks, stages, DAG artifacts
CREATE SCHEMA IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_MLJOBS');

-- DATA schema: Feature Store, Model Registry, raw data tables
CREATE SCHEMA IF NOT EXISTS IDENTIFIER($DB_NAME || '.DATA');

-- -------------------------
-- Stages
-- -------------------------
-- CODE_STAGE: holds Python source code uploaded from local
CREATE STAGE IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_MLJOBS.CODE_STAGE')
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- JOB_STAGE: ML Job payloads (used by @remote decorator)
CREATE STAGE IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_MLJOBS.JOB_STAGE')
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- DAG_STAGE: DAG-level artifacts and run metadata
CREATE STAGE IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_MLJOBS.DAG_STAGE')
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- ARTIFACTS_STAGE: model pickles, run artifacts, intermediate results
CREATE STAGE IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_MLJOBS.ARTIFACTS_STAGE')
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- -------------------------
-- Image Repository (for ML Jobs on SPCS)
-- -------------------------
CREATE IMAGE REPOSITORY IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_MLJOBS.ML_IMAGE_REPO');

-- -------------------------
-- Grants
-- -------------------------
-- Grant usage on database and schemas
GRANT USAGE ON DATABASE IDENTIFIER($DB_NAME) TO ROLE IDENTIFIER($ROLE_NAME);
GRANT ALL ON SCHEMA IDENTIFIER($DB_NAME || '.PIPELINE_MLJOBS') TO ROLE IDENTIFIER($ROLE_NAME);
GRANT ALL ON SCHEMA IDENTIFIER($DB_NAME || '.DATA') TO ROLE IDENTIFIER($ROLE_NAME);

-- Grant usage on stages
GRANT READ, WRITE ON STAGE IDENTIFIER($DB_NAME || '.PIPELINE_MLJOBS.CODE_STAGE') TO ROLE IDENTIFIER($ROLE_NAME);
GRANT READ, WRITE ON STAGE IDENTIFIER($DB_NAME || '.PIPELINE_MLJOBS.JOB_STAGE') TO ROLE IDENTIFIER($ROLE_NAME);
GRANT READ, WRITE ON STAGE IDENTIFIER($DB_NAME || '.PIPELINE_MLJOBS.DAG_STAGE') TO ROLE IDENTIFIER($ROLE_NAME);
GRANT READ, WRITE ON STAGE IDENTIFIER($DB_NAME || '.PIPELINE_MLJOBS.ARTIFACTS_STAGE') TO ROLE IDENTIFIER($ROLE_NAME);

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
SHOW STAGES IN SCHEMA IDENTIFIER($DB_NAME || '.PIPELINE_MLJOBS');

SELECT 'Infrastructure setup complete for ' || $DB_NAME AS STATUS;
