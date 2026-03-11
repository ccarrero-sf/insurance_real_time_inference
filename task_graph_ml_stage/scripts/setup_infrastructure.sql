-- =============================================================================
-- Car Insurance ML Pipeline (submit_from_stage version) - Infrastructure Setup
-- =============================================================================
-- This script creates all Snowflake objects needed for the ML pipeline.
-- Uses a SEPARATE database (CC_INSURANCE_PIPELINE) from the notebook's
-- CC_ML_INSURANCE to keep pipeline infrastructure isolated.
--
-- This version uses a Snowflake Git Repository stage as the source for
-- ML Job code. MLJobDefinition.register() points to the Git repo stage
-- instead of uploading from a local directory.
--
-- Schema: PIPELINE_STAGE (separate from the directory version's PIPELINE_SUBMIT_FILE)
--
-- Prerequisites:
--   - A compute pool must exist (or be created below)
--   - The executing role needs CREATE DATABASE privileges
--   - A Git repository (e.g. GitHub) with the pipeline code committed
--
-- Usage:
--   Run this script once to set up the environment.
--   Adjust ROLE_NAME, WAREHOUSE, COMPUTE_POOL, and GIT variables as needed.
-- =============================================================================

-- -------------------------
-- Configuration variables
-- -------------------------
SET ROLE_NAME = 'SPCS_PSE_ROLE';       -- Change to your role
SET WAREHOUSE = 'COMPUTE_WH';          -- Change to your warehouse
SET COMPUTE_POOL = 'DEMO_POOL';        -- Change to your compute pool
SET DB_NAME = 'CC_INSURANCE_PIPELINE';

-- Git configuration - CHANGE THESE to match your repository
SET GIT_REPO_ORIGIN = 'https://github.com/YOUR_ORG/YOUR_REPO.git';  -- Change to your repo URL
SET GIT_API_INTEGRATION_NAME = 'ML_PIPELINE_GIT_API_INTEGRATION';
SET GIT_REPO_NAME = 'ML_PIPELINE_GIT_REPO';

-- -------------------------
-- Database and Schemas
-- -------------------------
CREATE DATABASE IF NOT EXISTS IDENTIFIER($DB_NAME);

-- PIPELINE_STAGE schema: tasks, stages, DAG artifacts, Git integration
CREATE SCHEMA IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_STAGE');

-- DATA schema: Feature Store, Model Registry, raw data tables
CREATE SCHEMA IF NOT EXISTS IDENTIFIER($DB_NAME || '.DATA');

-- -------------------------
-- Stages (internal, for payload/scratch and artifacts)
-- -------------------------
-- JOB_STAGE: ML Job payloads (scratch stage used by MLJobDefinition.register)
CREATE STAGE IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_STAGE.JOB_STAGE')
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- DAG_STAGE: DAG-level artifacts and run metadata
CREATE STAGE IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_STAGE.DAG_STAGE')
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- ARTIFACTS_STAGE: model pickles, run artifacts, intermediate results
CREATE STAGE IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_STAGE.ARTIFACTS_STAGE')
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- -------------------------
-- Image Repository (for ML Jobs on SPCS)
-- -------------------------
CREATE IMAGE REPOSITORY IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_STAGE.ML_IMAGE_REPO');

-- -------------------------
-- Git Integration (read-only, no credentials needed)
-- -------------------------
-- Step 1: Create an API Integration for Git HTTPS access
-- NOTE: Change API_ALLOWED_PREFIXES to match your Git hosting provider/org.
CREATE API INTEGRATION IF NOT EXISTS IDENTIFIER($GIT_API_INTEGRATION_NAME)
    API_PROVIDER = git_https_api
    API_ALLOWED_PREFIXES = ('https://github.com/ccarrero-sf')
    ENABLED = TRUE;

-- Step 2: Create the Git Repository stage (no credentials for public/read-only repos)
CREATE GIT REPOSITORY IF NOT EXISTS IDENTIFIER($DB_NAME || '.PIPELINE_STAGE.' || $GIT_REPO_NAME)
    API_INTEGRATION = IDENTIFIER($GIT_API_INTEGRATION_NAME)
    ORIGIN = $GIT_REPO_ORIGIN;

-- Step 3: Initial fetch to sync the repository
ALTER GIT REPOSITORY IDENTIFIER($DB_NAME || '.PIPELINE_STAGE.' || $GIT_REPO_NAME) FETCH;

-- -------------------------
-- Grants
-- -------------------------
-- Grant usage on database and schemas
GRANT USAGE ON DATABASE IDENTIFIER($DB_NAME) TO ROLE IDENTIFIER($ROLE_NAME);
GRANT ALL ON SCHEMA IDENTIFIER($DB_NAME || '.PIPELINE_STAGE') TO ROLE IDENTIFIER($ROLE_NAME);
GRANT ALL ON SCHEMA IDENTIFIER($DB_NAME || '.DATA') TO ROLE IDENTIFIER($ROLE_NAME);

-- Grant usage on stages
GRANT READ, WRITE ON STAGE IDENTIFIER($DB_NAME || '.PIPELINE_STAGE.JOB_STAGE') TO ROLE IDENTIFIER($ROLE_NAME);
GRANT READ, WRITE ON STAGE IDENTIFIER($DB_NAME || '.PIPELINE_STAGE.DAG_STAGE') TO ROLE IDENTIFIER($ROLE_NAME);
GRANT READ, WRITE ON STAGE IDENTIFIER($DB_NAME || '.PIPELINE_STAGE.ARTIFACTS_STAGE') TO ROLE IDENTIFIER($ROLE_NAME);

-- Grant usage on Git repository
GRANT READ ON GIT REPOSITORY IDENTIFIER($DB_NAME || '.PIPELINE_STAGE.' || $GIT_REPO_NAME) TO ROLE IDENTIFIER($ROLE_NAME);

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
SHOW STAGES IN SCHEMA IDENTIFIER($DB_NAME || '.PIPELINE_STAGE');
SHOW GIT REPOSITORIES IN SCHEMA IDENTIFIER($DB_NAME || '.PIPELINE_STAGE');

SELECT 'Infrastructure setup complete for ' || $DB_NAME AS STATUS;
