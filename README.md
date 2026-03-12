# Car Insurance Premium Prediction - Real-Time ML Pipeline

A complete end-to-end ML pipeline for predicting car insurance premiums using:
- **Snowflake Feature Store** with Online Feature Store for real-time serving
- **XGBoost** model with StandardScaler preprocessing
- **Snowflake Model Registry** for model management
- **SPCS (Snowpark Container Services)** for real-time inference

## Architecture

The project has three main paths: real-time inference, batch inference, and an automated pipeline.

```
┌──────────── REAL-TIME INFERENCE ─────────────┐
│                                               │
│  Streamlit Frontend                           │
│       │ POST /predict                         │
│       ▼                                       │
│  FastAPI Backend (SPCS)                       │
│    ├─ Online Feature Store lookup             │
│    └─ Model Inference (SPCS Service)          │
│                                               │
│  Gateway (stable URL routing)                 │
│                                               │
│  Perf Test App (concurrent load testing)      │
└───────────────────────────────────────────────┘

┌──────────── BATCH INFERENCE ─────────────────┐
│                                               │
│  Batch Inference Notebook                     │
│    ├─ Generate synthetic test data (200 rows) │
│    ├─ mv.run() via SPCS inference service     │
│    ├─ Query Inference Table (telemetry)       │
│    └─ Stream + Task transform pipeline        │
│       (polls every 5 min)                     │
└───────────────────────────────────────────────┘

┌──────────── AUTOMATED PIPELINE (DAG) ────────┐
│                                               │
│  INGEST_DATA (ML Job on compute pool)         │
│       │                                       │
│  TRAIN_MODEL (Feature Store + XGBoost)        │
│       │                                       │
│  CHECK_QUALITY (branch)                       │
│       ├─── pass ──▶ PROMOTE_MODEL             │
│       │                  │                    │
│       │             RUN_INFERENCE              │
│       │                                       │
│       └─── fail ──▶ SEND_ALERT                │
│                                               │
│  CLEANUP (finalizer - always runs)            │
│                                               │
│  Schedule: daily 9am UTC                      │
│  CI/CD: GitHub Actions deploys on push        │
│  Variants: task_graph / task_graph_ml_stage    │
└───────────────────────────────────────────────┘
```

## Prerequisites

1. **Snowflake Account** with:
   - SPCS enabled (Snowpark Container Services)
   - A role with sufficient privileges (e.g., `SPCS_PSE_ROLE`)
   - A warehouse for compute (e.g., `COMPUTE_WH`)

2. **Local Environment**:
   - Python 3.10 or 3.11
   - Docker (for building backend images)
   - Snowflake CLI (`snow`) installed and configured
   - Conda/Miniconda (recommended)

3. **Snowflake Connection**:
   - Configure a connection in `~/.snowflake/connections.toml` (see below)

## Configure Snowflake Connection

Before running any code, you need to configure a Snowflake connection. Create or edit `~/.snowflake/connections.toml`:

### Option A: Key Pair Authentication (Recommended)

First, generate a key pair if you don't have one:

```bash
# Generate private key
openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out ~/.snowflake/rsa_key.p8 -nocrypt

# Generate public key
openssl rsa -in ~/.snowflake/rsa_key.p8 -pubout -out ~/.snowflake/rsa_key.pub

# Display public key (copy this to Snowflake)
cat ~/.snowflake/rsa_key.pub
```

Then assign the public key to your Snowflake user:

```sql
-- Run in Snowflake (as ACCOUNTADMIN or SECURITYADMIN)
ALTER USER your_username SET RSA_PUBLIC_KEY='MIIBIjANBgkq...';  -- paste your public key here
```

Add to `~/.snowflake/connections.toml`:

```toml
[talent_keypair]
account = "your_account"           # e.g., "xy12345.us-west-2"
user = "your_username"
authenticator = "SNOWFLAKE_JWT"
private_key_path = "~/.snowflake/rsa_key.p8"
role = "SPCS_PSE_ROLE"             # or your role with SPCS privileges
warehouse = "COMPUTE_WH"
database = "CC_ML_INSURANCE"
schema = "CAR_PRICING"
```

### Option B: Password Authentication

Add to `~/.snowflake/connections.toml`:

```toml
[talent_keypair]
account = "your_account"           # e.g., "xy12345.us-west-2"
user = "your_username"
password = "your_password"
role = "SPCS_PSE_ROLE"
warehouse = "COMPUTE_WH"
database = "CC_ML_INSURANCE"
schema = "CAR_PRICING"
```

### Option C: SSO/Browser Authentication

```toml
[talent_keypair]
account = "your_account"
user = "your_username"
authenticator = "externalbrowser"
role = "SPCS_PSE_ROLE"
warehouse = "COMPUTE_WH"
database = "CC_ML_INSURANCE"
schema = "CAR_PRICING"
```

### Verify Connection

```bash
# Test the connection
snow connection test -c talent_keypair
```

You should see: `Connection test successful!`

## Project Structure

```
insurance_real_time_inference/
├── car_insurance_ml.ipynb                # Main notebook - sets up everything
├── car_insurance_batch_inference.ipynb    # Batch inference & transform pipeline
├── car_insurance_frontend.py             # Frontend Streamlit app
├── car_insurance_realtime_perf_test.py   # Performance testing app
├── requirements.txt                      # Python dependencies (local)
├── environment.yml                       # Conda environment
├── snowflake.yml                         # Snowflake project config
├── backend/
│   ├── app.py                            # FastAPI backend service
│   ├── Dockerfile                        # Docker build file
│   ├── requirements.txt                  # Backend dependencies
│   ├── deploy.sh                         # Deployment script
│   └── service-spec.yaml                 # SPCS service spec
├── task_graph/                           # Automated DAG (direct serialization)
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── setup_infrastructure.sql      # DB, schemas, stages, grants
│   │   ├── run_setup.py                  # Python runner for SQL setup
│   │   └── upload_code.py                # Uploads src/ to @CODE_STAGE
│   └── src/
│       ├── constants.py                  # Config (DB, schemas, thresholds)
│       ├── data_ops.py                   # Synthetic data generation + ML Job ingestion
│       ├── feature_ops.py                # Feature Store setup + datasets
│       ├── modeling.py                   # XGBoost training, registry, inference
│       ├── pipeline_tasks.py             # 7 DAG task entry points
│       └── deploy_dag.py                 # DAG deployment (daily 6am UTC)
├── task_graph_stage/                     # Automated DAG (stage-loader variant)
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── setup_infrastructure.sql
│   │   ├── run_setup.py
│   │   └── upload_code.py
│   └── src/
│       ├── constants.py
│       ├── data_ops.py
│       ├── feature_ops.py
│       ├── modeling.py
│       ├── pipeline_tasks.py
│       └── deploy_dag.py
└── .streamlit/
    ├── secrets.toml                      # Local secrets (PAT token)
    └── secrets.toml.example              # Template for secrets
```

## Step 1: Set Up Local Environment

### Option A: Using Conda (Recommended)

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate sf_env
```

### Option B: Using pip

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Run the Notebook (ML Pipeline Setup)

The notebook creates all required Snowflake objects:
- Database and schema (`CC_ML_INSURANCE.CAR_PRICING`)
- Synthetic data tables (`CUSTOMERS`, `POLICIES`)
- Feature Store with Online Feature Store enabled
- Trained XGBoost model registered in Model Registry
- Model inference service (SPCS)

### Run the Notebook

```bash
# Set your connection name
export SNOWFLAKE_CONNECTION_NAME=talent_keypair

# Open and run the notebook
jupyter notebook car_insurance_ml.ipynb
```

**Execute all cells in order.** The notebook will:

1. **Setup**: Connect to Snowflake, create database/schema
2. **Data Generation**: Create 5,000 customers and 8,000 policies
3. **Feature Store**: Create Entity, Feature View with online config
4. **Training**: Prepare dataset, train XGBoost model
5. **Model Registry**: Register model with metrics
6. **Inference Service**: Deploy model to SPCS

**Expected Duration**: ~15-20 minutes (model deployment can take time)

### Verify Notebook Completion

After running, you should see:
- `CC_ML_INSURANCE.CAR_PRICING` database/schema created
- Feature View `CUSTOMER_RISK_FEATURES` with online serving enabled
- Model `CAR_INSURANCE_PRICING_MODEL` registered
- Service `CAR_INSURANCE_INFERENCE_SVC` running

## Step 3: Deploy the Backend Service

The backend is a FastAPI service that:
- Receives prediction requests
- Fetches customer features from Online Feature Store
- Calls the model inference service
- Returns predictions with timing metrics

### Deploy Using Script

```bash
cd backend
chmod +x deploy.sh
./deploy.sh
```

The script will:
1. Create image repository if needed
2. Create/resume compute pool `BACKEND_CPU_POOL`
3. Build and push Docker image
4. Deploy `INSURANCE_BACKEND_SVC`
5. Display the service endpoint URL

**Expected Duration**: ~5-10 minutes

### Verify Backend Deployment

```bash
# Check service status
snow sql -c talent_keypair -q "CALL SYSTEM\$GET_SERVICE_STATUS('CC_ML_INSURANCE.CAR_PRICING.INSURANCE_BACKEND_SVC');"

# Get endpoint URL
snow sql -c talent_keypair -q "SHOW ENDPOINTS IN SERVICE CC_ML_INSURANCE.CAR_PRICING.INSURANCE_BACKEND_SVC;"
```

## Step 4: Configure Authentication

The frontend and performance test apps need a PAT (Programmatic Access Token) for authentication when running locally.

### Create PAT Token

1. Go to Snowsight
2. Click User Menu (bottom left) → **Preferences**
3. Go to **Programmatic Access Tokens**
4. Click **Add Token**
5. Copy the token value

### Configure Secrets

```bash
# Copy the example file
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# Edit with your PAT token
cat > .streamlit/secrets.toml << 'EOF'
[snowflake]
pat_token = "your-pat-token-here"
EOF
```

## Step 5: Run the Frontend App

The frontend allows users to:
- Select existing customers or create new quotes
- Enter driver and vehicle information
- Get instant premium predictions

```bash
# Set connection
export SNOWFLAKE_CONNECTION_NAME=talent_keypair

# Run frontend
streamlit run car_insurance_frontend.py
```

Open http://localhost:8501 in your browser.

### Using the Frontend

1. **Customer Selection**: Choose an existing customer or "New Customer"
2. **Driver Info**: Age, years licensed, gender, state
3. **Vehicle Info**: Make, model, age, kilometers, engine size
4. **Coverage**: Coverage type and estimated car value
5. Click **Calculate Premium** to get your quote

## Step 6: Run Performance Testing

The performance test app allows you to:
- Test the full pipeline at scale
- Measure latency breakdown (Feature Store, Model Inference)
- Monitor throughput and success rates
- View model inference telemetry

```bash
# Set connection
export SNOWFLAKE_CONNECTION_NAME=talent_keypair

# Run performance test
streamlit run car_insurance_realtime_perf_test.py
```

Open http://localhost:8501 in your browser.

### Performance Test Features

- **Single Request Test**: Test one request in sidebar
- **Continuous Load Test**: Start/stop continuous testing
- **Concurrency Control**: Adjust 1-20 concurrent workers
- **Real-time Metrics**: Latency, throughput, success rate
- **Component Breakdown**: Feature Store vs Model Inference time
- **Inference Telemetry**: Data from INFERENCE_TABLE

## Step 7: Batch Inference & Transform Pipeline

The batch inference notebook (`car_insurance_batch_inference.ipynb`) demonstrates:
- Generating synthetic test data (200 rows)
- Running batch predictions via `mv.run()` on the SPCS inference service
- Querying the Inference Table for logged predictions and telemetry
- Creating a stream + task pipeline that monitors new inference records and runs `transform` every 5 minutes

```bash
# Set connection
export SNOWFLAKE_CONNECTION_NAME=talent_keypair

# Open and run the notebook
jupyter notebook car_insurance_batch_inference.ipynb
```

**Execute all cells in order.** The notebook will:

1. **Generate Data**: Create 200 synthetic test records and upload to `TEST_BATCH_DATA`
2. **Batch Predict**: Call `mv.run(function_name="predict", service_name="CAR_INSURANCE_INFERENCE_SVC")` on the test data
3. **Query Inference Table**: Inspect logged predictions via `INFERENCE_TABLE()` (timestamps, query IDs, inputs, outputs)
4. **Transform Pipeline**: Create a stream on the inference table and a scheduled task that runs `transform` on new records every 5 minutes

## Step 8: Automated Pipeline (Task Graph)

The project includes three variants of an automated ML pipeline deployed as a Snowflake Task Graph (DAG). All implement the same workflow but differ in how task code is loaded and deployed.

### DAG Structure

```
Variants A & B:
INGEST_DATA >> TRAIN_MODEL >> CHECK_QUALITY >> [PROMOTE_MODEL, SEND_ALERT]
PROMOTE_MODEL >> RUN_INFERENCE
CLEANUP (finalizer - always runs)

Variant C (task_graph_ml_stage):
INGEST_DATA >> PREPARE_DATA >> TRAIN_MODEL >> CHECK_QUALITY >> [PROMOTE_MODEL, SEND_ALERT]
PROMOTE_MODEL >> RUN_INFERENCE
CLEANUP (finalizer - always runs)
```

| Task | Description |
|------|-------------|
| `INGEST_DATA` | Generate synthetic customer + policy data via ML Job on compute pool |
| `PREPARE_DATA` | *(Variant C only)* Feature engineering and data preparation via stored procedure |
| `TRAIN_MODEL` | Set up Feature Store, prepare datasets, train XGBoost model, register in Model Registry |
| `CHECK_QUALITY` | Branch: compare new model R2 against threshold; route to promote or alert |
| `PROMOTE_MODEL` | Set new model version as production default |
| `SEND_ALERT` | Log alert to `PIPELINE_ALERTS` table (model did not meet criteria) |
| `RUN_INFERENCE` | Run batch predictions using the promoted model |
| `CLEANUP` | Clean up old versions and temporary artifacts (always runs) |

The pipeline uses a **separate database** (`CC_INSURANCE_PIPELINE`) from the interactive notebook's `CC_ML_INSURANCE`, keeping pipeline infrastructure isolated.

### Variant A: `task_graph/` (Direct Serialization)

Task functions are serialized directly at deploy time. To update task logic, you must redeploy the DAG.

- Quality threshold: R2 >= 0.7
- Schema: `CC_INSURANCE_PIPELINE.PIPELINE`

```bash
cd task_graph

# 1. Set up infrastructure (run once)
# Execute scripts/setup_infrastructure.sql in Snowflake to create DB, schemas, stages, and grants

# 2. Upload source code to @CODE_STAGE
cd src
python deploy_dag.py          # Deploy DAG (suspended)
python deploy_dag.py --execute  # Deploy and immediately execute
```

### Variant B: `task_graph_stage/` (Stage Loader)

Each task uses a generic `stage_task_runner` that dynamically imports the target module from `@CODE_STAGE` at runtime. Updating `.py` files on the stage takes effect on the next DAG run **without redeploying** the DAG.

- Quality threshold: R2 >= 0.5 (with 0.05 tolerance)
- Schema: `CC_INSURANCE_PIPELINE.PIPELINE_STG`
- Preprocessing: uses sklearn Pipeline with OrdinalEncoder

```bash
cd task_graph_stage

# 1. Set up infrastructure (run once)
# Execute scripts/setup_infrastructure.sql in Snowflake

# 2. Upload source code to @CODE_STAGE
python scripts/upload_code.py

# 3. Deploy DAG
cd src
python deploy_dag.py          # Deploy DAG (suspended)
python deploy_dag.py --execute  # Deploy and immediately execute
```

### Variant C: `task_graph_ml_stage/` (CI/CD with GitHub Actions)

Extends Variant B with automated CI/CD deployment via GitHub Actions. Code is loaded from a **Git Repository stage** (`ML_PIPELINE_GIT_REPO`) instead of a manually-uploaded `@CODE_STAGE`. Pushing to `main` automatically deploys the DAG.

- Quality threshold: R2 >= 0.5 (with 0.05 tolerance)
- Schema: `CC_INSURANCE_PIPELINE.PIPELINE_STAGE`
- Code source: Git Repository stage (auto-synced on deploy)
- Authentication: Key-pair (no password)

#### Roles

| Role | Purpose | Assigned To |
|------|---------|-------------|
| `CICD_DEPLOY_RL` | Deploy/replace DAG, sync Git repo, manage stages | `CICD_DEPLOY_USER` (service account) |
| `PIPELINE_OPERATOR_RL` | Monitor, execute, suspend/resume tasks | Human users (e.g., `CCARRERO`) |

#### CI/CD Setup

1. **Generate a key pair** for the service account:

```bash
openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out cicd_rsa_key.p8 -nocrypt
openssl rsa -in cicd_rsa_key.p8 -pubout -out cicd_rsa_key.pub
```

2. **Run the setup script** (requires ACCOUNTADMIN):

```bash
cd task_graph_ml_stage

# Set required environment variables
export PUBLIC_KEY=$(cat cicd_rsa_key.pub | grep -v "BEGIN\|END" | tr -d '\n')
export OPERATOR_USER=CCARRERO  # User who will operate the pipeline

python scripts/setup_cicd_service_account.py
```

This creates:
- `CICD_DEPLOY_USER` service account with key-pair auth
- `CICD_DEPLOY_RL` role with deploy privileges
- `CICD_NETWORK_POLICY` allowing GitHub Actions IPs
- `PIPELINE_OPERATOR_RL` role granted to the specified user

3. **Configure GitHub Secrets** in your repository:

| Secret/Variable | Value |
|-----------------|-------|
| `SNOWFLAKE_ACCOUNT` | Your Snowflake account identifier (e.g., `phb14991`) |
| `SNOWFLAKE_USER` | `CICD_DEPLOY_USER` |
| `SNOWFLAKE_PRIVATE_KEY` | Contents of `cicd_rsa_key.p8` |

4. **Deploy**: Push changes to `task_graph_ml_stage/` on `main` branch, or trigger manually via `workflow_dispatch`.

#### Operating the Pipeline

Use `PIPELINE_OPERATOR_RL` to monitor and control tasks without deploy privileges:

```sql
USE ROLE PIPELINE_OPERATOR_RL;

-- Check DAG status
SHOW TASKS IN SCHEMA CC_INSURANCE_PIPELINE.PIPELINE_STAGE;

-- Execute the DAG manually
EXECUTE TASK CC_INSURANCE_PIPELINE.PIPELINE_STAGE.CAR_INSURANCE_ML_PIPELINE;

-- Suspend/resume the DAG
ALTER TASK CC_INSURANCE_PIPELINE.PIPELINE_STAGE.CAR_INSURANCE_ML_PIPELINE SUSPEND;
ALTER TASK CC_INSURANCE_PIPELINE.PIPELINE_STAGE.CAR_INSURANCE_ML_PIPELINE RESUME;

-- View task run history
SELECT *
FROM TABLE(CC_INSURANCE_PIPELINE.INFORMATION_SCHEMA.TASK_HISTORY())
ORDER BY SCHEDULED_TIME DESC
LIMIT 20;
```

### Monitoring the Pipeline

```sql
-- Check task history
SELECT * FROM TABLE(CC_INSURANCE_PIPELINE.INFORMATION_SCHEMA.TASK_HISTORY())
ORDER BY SCHEDULED_TIME DESC;

-- View pipeline alerts
SELECT * FROM CC_INSURANCE_PIPELINE.PIPELINE.PIPELINE_ALERTS
ORDER BY ALERT_TIME DESC;

-- Resume/execute the DAG manually
ALTER TASK CC_INSURANCE_PIPELINE.PIPELINE.CAR_INSURANCE_ML_PIPELINE RESUME;
EXECUTE TASK CC_INSURANCE_PIPELINE.PIPELINE.CAR_INSURANCE_ML_PIPELINE;
```

## Troubleshooting

### Notebook Issues


**Model registration takes too long**
- Model registration includes building container images. Wait up to 10 minutes.

### Backend Deployment Issues

**Compute pool not starting**
```bash
snow sql -c talent_keypair -q "ALTER COMPUTE POOL BACKEND_CPU_POOL RESUME;"
```

**Service not becoming READY**
```bash
# Check logs
snow sql -c talent_keypair -q "CALL SYSTEM\$GET_SERVICE_LOGS('CC_ML_INSURANCE.CAR_PRICING.INSURANCE_BACKEND_SVC', 0, 'insurance-backend');"
```

**Image push fails**
```bash
# Re-login to registry
snow spcs image-registry login -c talent_keypair
```

### Frontend/Performance Test Issues

**"Backend service not available"**
- Check that `INSURANCE_BACKEND_SVC` is running
- Verify endpoint is provisioned (not "Endpoints provisioning")

**"Could not load customers"**
- Check your connection name is correct
- Verify the notebook was run successfully

**Authentication errors**
- Verify PAT token is valid and not expired
- Check `.streamlit/secrets.toml` is configured correctly

## Cleanup

To remove all resources:

```sql
-- Drop services
DROP SERVICE IF EXISTS CC_ML_INSURANCE.CAR_PRICING.INSURANCE_BACKEND_SVC;
DROP SERVICE IF EXISTS CC_ML_INSURANCE.CAR_PRICING.CAR_INSURANCE_INFERENCE_SVC;

-- Drop compute pools
DROP COMPUTE POOL IF EXISTS BACKEND_CPU_POOL;

-- Drop interactive database (removes all objects)
DROP DATABASE IF EXISTS CC_ML_INSURANCE;

-- Drop pipeline database (removes DAGs, stages, feature store, etc.)
DROP DATABASE IF EXISTS CC_INSURANCE_PIPELINE;
```

## Key Snowflake Features Demonstrated

- **Feature Store**: Entity, Feature View with online serving
- **Online Feature Store**: Sub-second feature retrieval
- **Model Registry**: Version management, metrics tracking, production promotion
- **SPCS**: Container services for backend and model inference
- **Inference Table**: Auto-captured model telemetry and monitoring
- **ML Jobs**: `@remote` decorator for training/inference on SPCS compute pools
- **Task Graph (DAG)**: Automated pipeline with DAGTask, DAGTaskBranch, TaskContext, Cron scheduling
- **Gateway**: Stable URL routing to SPCS service endpoints
- **Streams & Tasks**: Change data capture pipeline for transform monitoring

## Configuration Reference

### Interactive Setup (Notebook)

| Parameter | Value |
|-----------|-------|
| Database | `CC_ML_INSURANCE` |
| Schema | `CAR_PRICING` |
| Feature View | `CUSTOMER_RISK_FEATURES` |
| Model Name | `CAR_INSURANCE_PRICING_MODEL` |
| Backend Service | `INSURANCE_BACKEND_SVC` |
| Model Service | `CAR_INSURANCE_INFERENCE_SVC` |
| Gateway | `CAR_INSURANCE_GATEWAY` |
| Compute Pool | `BACKEND_CPU_POOL` |
| Warehouse | `COMPUTE_WH` |

### Automated Pipeline (Task Graph)

| Parameter | `task_graph/` | `task_graph_stage/` | `task_graph_ml_stage/` |
|-----------|---------------|---------------------|------------------------|
| Database | `CC_INSURANCE_PIPELINE` | `CC_INSURANCE_PIPELINE` | `CC_INSURANCE_PIPELINE` |
| Schema | `PIPELINE` | `PIPELINE_STG` | `PIPELINE_STAGE` |
| Data Schema | `DATA` | `DATA` | `DATA` |
| Compute Pool | `DEMO_POOL` | `DEMO_POOL` | `DEMO_POOL` |
| DAG Name | `CAR_INSURANCE_ML_PIPELINE` | `CAR_INSURANCE_ML_PIPELINE` | `CAR_INSURANCE_ML_PIPELINE` |
| Schedule | Daily 6am UTC | Daily 6am UTC | Daily 9am UTC |
| Quality Threshold | R2 >= 0.7 | R2 >= 0.5 (± 0.05 tolerance) | R2 >= 0.5 (± 0.05 tolerance) |
| Code Loading | Direct serialization | Dynamic import from `@CODE_STAGE` | Git Repository stage (`ML_PIPELINE_GIT_REPO`) |
| Deployment | Manual | Manual | CI/CD (GitHub Actions) |
| Auth | Password/SSO | Password/SSO | Key-pair (service account) |
