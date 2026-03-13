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
│  Schedule: daily (6am or 9am UTC by variant)  │
│  CI/CD: GitHub Actions deploys on push (E)    │
│  Variants: A through E (see Step 8)           │
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
├── task_graph/                           # Variant A: SPs with @remote ML Jobs
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── setup_infrastructure.sql      # DB, schemas, stages, grants
│   │   ├── run_setup.py                  # Python runner for SQL setup
│   │   └── upload_code.py                # Uploads src/ to @CODE_STAGE
│   └── src/
│       ├── constants.py                  # Config (DB, schemas, thresholds)
│       ├── data_ops.py                   # Synthetic data generation + @remote ML Job
│       ├── feature_ops.py                # Feature Store setup + datasets
│       ├── modeling.py                   # XGBoost training, registry, @remote inference
│       ├── pipeline_tasks.py             # 7 DAG task entry points (all SPs)
│       └── deploy_dag.py                 # DAG deployment (daily 6am UTC)
├── task_graph_stage/                     # Variant B: Stage loader (no redeploy)
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
│       ├── stage_task_runner.py           # Dynamic module loader (importlib)
│       └── deploy_dag.py
├── task_graph_ml/                        # Variant C: Direct @remote ML Job functions
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── setup_infrastructure.sql
│   │   ├── run_setup.py
│   │   └── upload_code.py
│   └── src/
│       ├── constants.py
│       ├── data_ops.py                   # @remote ingest_data_job
│       ├── feature_ops.py
│       ├── modeling.py                   # @remote train_model_job, run_inference_job
│       ├── pipeline_tasks.py             # SP tasks (prepare, check, promote, alert, cleanup)
│       ├── deploy_dag.py
│       └── helpers/                      # Shared modules for @remote functions
│           ├── __init__.py
│           ├── data_generation.py
│           ├── feature_engineering.py
│           ├── inference_input.py
│           └── model_artifacts.py
├── task_graph_ml_directory/              # Variant D: MLJobDefinition with directory submission
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
│       ├── deploy_dag.py
│       ├── helpers/                      # Helpers for SP tasks (via CODE_STAGE)
│       │   └── (same as task_graph_ml)
│       └── jobs/                         # Standalone ML Job scripts
│           ├── ingest_data.py            # Entry point: data generation
│           ├── train_model.py            # Entry point: model training
│           ├── run_inference.py          # Entry point: batch inference
│           └── helpers/                  # Helpers for ML Jobs (bundled in job payload)
│               └── (same as src/helpers)
├── task_graph_ml_stage/                  # Variant E: Git integration + CI/CD
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── setup_infrastructure.sql      # Includes Git repo + API integration
│   │   ├── run_setup.py
│   │   └── setup_cicd_service_account.py # Creates CICD_DEPLOY_USER + roles
│   └── src/
│       ├── constants.py                  # Includes GIT_REPO_STAGE, GIT_SRC_PATH
│       ├── data_ops.py
│       ├── feature_ops.py
│       ├── modeling.py
│       ├── pipeline_tasks.py
│       ├── deploy_dag.py                 # Git refresh + sync + claim/deploy/release
│       ├── helpers/                      # Helpers for SP tasks
│       │   └── (same as task_graph_ml)
│       └── jobs/                         # Standalone ML Job scripts (read from Git stage)
│           ├── ingest_data.py
│           ├── train_model.py
│           ├── run_inference.py
│           └── helpers/
│               └── (same as src/helpers)
├── .github/
│   └── workflows/
│       └── deploy-dag.yml                # GitHub Actions: auto-deploy on push to main
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

The project includes five variants of an automated ML pipeline deployed as a Snowflake Task Graph (DAG). All implement the same core workflow — ingest data, train a model, evaluate quality, promote or alert, run inference, and clean up — but differ in how tasks are executed, how code is loaded, and how deployment is managed. Each variant builds on the previous one, progressively introducing more advanced patterns.

All variants use a **separate database** (`CC_INSURANCE_PIPELINE`) from the interactive notebook's `CC_ML_INSURANCE`, keeping pipeline infrastructure isolated.

### DAG Structures

**Variants A & B** (7 tasks):

```
INGEST_DATA >> TRAIN_MODEL >> CHECK_QUALITY >> [PROMOTE_MODEL, SEND_ALERT]
PROMOTE_MODEL >> RUN_INFERENCE
CLEANUP (finalizer - always runs)
```

**Variants C, D & E** (8 tasks — adds PREPARE_DATA):

```
INGEST_DATA >> PREPARE_DATA >> TRAIN_MODEL >> CHECK_QUALITY >> [PROMOTE_MODEL, SEND_ALERT]
PROMOTE_MODEL >> RUN_INFERENCE
CLEANUP (finalizer - always runs)
```

### Task Reference

| Task | Type | Description |
|------|------|-------------|
| `INGEST_DATA` | `DAGTask` | Generate synthetic customer + policy data (5,000 customers, 8,000 policies) |
| `PREPARE_DATA` | `DAGTask` | *(Variants C/D/E only)* Feature Store setup (entity, feature view) + training dataset generation |
| `TRAIN_MODEL` | `DAGTask` | Train XGBoost model, evaluate metrics, register in Model Registry |
| `CHECK_QUALITY` | `DAGTaskBranch` | Compare new model R2 against threshold; route to `PROMOTE_MODEL` or `SEND_ALERT` |
| `PROMOTE_MODEL` | `DAGTask` | Set new model version as production default (conditional) |
| `SEND_ALERT` | `DAGTask` | Log alert to `PIPELINE_ALERTS` table (conditional) |
| `RUN_INFERENCE` | `DAGTask` | Run batch predictions using the promoted model |
| `CLEANUP` | `DAGTask` (finalizer) | Delete expired model versions; always runs regardless of success/failure |

All tasks communicate via `TaskContext.set_return_value()` / `get_predecessor_return_value()` with JSON-serialized payloads.

---

### Variant A: `task_graph/` — Stored Procedures with `@remote` ML Jobs

All 7 tasks are defined as `StoredProcedureCall` objects. For the three compute-intensive tasks (`INGEST_DATA`, `TRAIN_MODEL`, `RUN_INFERENCE`), the stored procedure calls a wrapper function that submits an `@remote`-decorated function as an ML Job on the SPCS compute pool and blocks until completion via `.result()`. The `@remote` functions (in `data_ops.py` and `modeling.py`) are self-contained — they duplicate all imports, constants, and logic inside the function body so they serialize cleanly to the container runtime.

Source code is uploaded to `@CODE_STAGE` and imported by each stored procedure at runtime via the `imports=STAGE_IMPORTS` parameter. To update task logic, the **DAG must be redeployed**.

**Key characteristics:**
- All tasks use `StoredProcedureCall` (runs on warehouse)
- Heavy compute delegated to SPCS via `@remote(COMPUTE_POOL, stage_name=JOB_STAGE)`
- Three `@remote` functions: `ingest_data_remote`, `train_model_remote`, `run_inference_remote`
- Code loading: stage imports from `@CODE_STAGE`
- Quality threshold: R2 >= 0.7
- Schema: `CC_INSURANCE_PIPELINE.PIPELINE`
- Schedule: Daily 6am UTC

**How to deploy:**

```bash
cd task_graph

# 1. Set up infrastructure (run once)
python scripts/run_setup.py
# Or execute scripts/setup_infrastructure.sql directly in Snowsight

# 2. Deploy DAG (uploads code to @CODE_STAGE + creates tasks)
cd src
python deploy_dag.py            # Deploy in suspended state
python deploy_dag.py --execute  # Deploy and execute immediately
```

**To update code:** Edit source files, then re-run `python deploy_dag.py` to redeploy the DAG.

---

### Variant B: `task_graph_stage/` — Stage Loader (No Redeploy on Code Change)

Same DAG structure as Variant A, but introduces a **dynamic code loading** pattern via `stage_task_runner.py`. Instead of serializing the actual task functions into the stored procedures, each task uses a thin wrapper that calls `run_task_from_stage(session, module_name, func_name)`. This function uses `importlib.import_module()` followed by `importlib.reload()` to dynamically import the target module from `@CODE_STAGE` at runtime.

The stored procedures contain only the 2-line wrapper — all business logic lives in the `.py` files on `@CODE_STAGE`. Updating code on the stage takes effect on the **next DAG run without redeploying** the DAG.

**Key characteristics:**
- All tasks use `StoredProcedureCall` with a thin `_make_task_wrapper` closure
- Runtime code loaded from `@CODE_STAGE` via `importlib.import_module()` + `importlib.reload()`
- Heavy compute delegated to SPCS via `@remote` (same as Variant A)
- **No DAG redeployment needed** for code changes — only `upload_code.py` required
- DAG redeployment only needed for structural changes (dependencies, schedule, packages, new files)
- Quality threshold: R2 >= 0.7
- Schema: `CC_INSURANCE_PIPELINE.PIPELINE_CODE_STAGE`
- Schedule: Daily 6am UTC

**How to deploy:**

```bash
cd task_graph_stage

# 1. Set up infrastructure (run once)
python scripts/run_setup.py
# Or execute scripts/setup_infrastructure.sql directly in Snowsight

# 2. Upload source code to @CODE_STAGE
python scripts/upload_code.py

# 3. Deploy DAG
cd src
python deploy_dag.py            # Deploy in suspended state
python deploy_dag.py --execute  # Deploy and execute immediately
```

**To update code (no redeploy):** Edit source files, then run `python scripts/upload_code.py`. The next DAG execution will pick up the new code automatically.

---

### Variant C: `task_graph_ml/` — Direct `@remote` ML Job Functions (No SP Wrapper)

This variant changes how compute-intensive tasks are defined. Instead of wrapping `@remote` functions inside a `StoredProcedureCall`, the three heavy tasks (`INGEST_DATA`, `TRAIN_MODEL`, `RUN_INFERENCE`) pass the `@remote`-decorated function **directly** to `DAGTask(definition=...)`. This eliminates the stored procedure layer entirely for ML Jobs — the function runs directly on the SPCS compute pool.

Lightweight tasks (`PREPARE_DATA`, `CHECK_QUALITY`, `PROMOTE_MODEL`, `SEND_ALERT`, `CLEANUP`) remain as `StoredProcedureCall` on the warehouse.

This variant also introduces a `helpers/` directory with reusable modules (`data_generation`, `feature_engineering`, `inference_input`, `model_artifacts`) that are imported by the `@remote` functions via the `imports=[("helpers", "helpers")]` parameter in the `@remote` decorator. This demonstrates how to package dependencies for ML Jobs.

A `PREPARE_DATA` task is added between `INGEST_DATA` and `TRAIN_MODEL` to handle Feature Store setup and dataset generation as a separate warehouse-based step.

**Key characteristics:**
- ML Job tasks pass `@remote` functions directly as `DAGTask(definition=...)` — no `StoredProcedureCall`
- ML Jobs create their own session via `Session.builder.getOrCreate()` (runs on SPCS)
- `helpers/` directory imported via `@remote(imports=[("helpers", "helpers")])` decorator parameter
- 8-task DAG (adds `PREPARE_DATA`)
- Quality threshold: R2 >= 0.7
- Schema: `CC_INSURANCE_PIPELINE.PIPELINE_MLJOBS`
- Schedule: Daily 6am UTC

**How to deploy:**

```bash
cd task_graph_ml

# 1. Set up infrastructure (run once)
python scripts/run_setup.py
# Or execute scripts/setup_infrastructure.sql directly in Snowsight

# 2. Deploy DAG (uploads code + creates tasks)
cd src
python deploy_dag.py            # Deploy in suspended state
python deploy_dag.py --execute  # Deploy and execute immediately
```

**To update code:** Edit source files, then re-run `python deploy_dag.py` to redeploy the DAG.

---

### Variant D: `task_graph_ml_directory/` — MLJobDefinition with Directory Submission

This variant replaces the `@remote`-decorated functions with **`MLJobDefinition.register()`** using the `submit_directory` pattern. Each ML Job is a standalone Python script in the `jobs/` directory with its own `main()` function, `if __name__ == "__main__"` guard, and `Session.builder.getOrCreate()`. The entire `jobs/` directory (including `jobs/helpers/`) is packaged as the job payload.

`MLJobDefinition.register()` takes three parameters:
- `source`: the local `jobs/` directory (uploaded automatically)
- `entrypoint`: which script to run (e.g., `ingest_data.py`)
- `compute_pool`: the SPCS compute pool

The returned `MLJobDefinition` object is passed to `DAGTask(definition=...)`. Once registered, a job definition can be reused across multiple DAG runs without re-uploading the payload.

The `helpers/` directory is **duplicated** in two locations (`src/helpers/` for SP tasks via `CODE_STAGE` and `src/jobs/helpers/` for ML Jobs via the job payload) because the two execution environments resolve imports differently.

**Key characteristics:**
- ML Job tasks use `MLJobDefinition.register(source=jobs_dir, entrypoint=...)` 
- Each job is a standalone script with its own `main()` and session
- Entire `jobs/` directory bundled per job definition (including `helpers/`)
- Dual helpers: `src/helpers/` (SPs) and `src/jobs/helpers/` (ML Jobs) — identical content
- Two-stage code upload: `CODE_STAGE` (for SPs) and `JOB_STAGE` (for ML Jobs, handled by `register()`)
- 8-task DAG (same structure as Variant C)
- Quality threshold: R2 >= 0.7
- Schema: `CC_INSURANCE_PIPELINE.PIPELINE_SUBMIT_FILE`
- Schedule: Daily 6am UTC

**How to deploy:**

```bash
cd task_graph_ml_directory

# 1. Set up infrastructure (run once)
python scripts/run_setup.py
# Or execute scripts/setup_infrastructure.sql directly in Snowsight

# 2. Deploy DAG (uploads code to CODE_STAGE, registers MLJobDefinitions, creates tasks)
cd src
python deploy_dag.py            # Deploy in suspended state
python deploy_dag.py --execute  # Deploy and execute immediately
```

**To update code:** Edit source files (and/or job scripts), then re-run `python deploy_dag.py` to re-register job definitions and redeploy the DAG.

---

### Variant E: `task_graph_ml_stage/` — Git Integration with CI/CD (Most Complete)

The most complete variant. Same DAG structure as Variants C/D, but the code source is a **Snowflake Git Repository stage** (`ML_PIPELINE_GIT_REPO`) instead of local files. A GitHub Actions workflow auto-deploys the DAG on every push to `main` that modifies `task_graph_ml_stage/`.

At deploy time, the script:
1. **Refreshes the Git repo** (`ALTER GIT REPOSITORY ... FETCH`) to pull the latest commits
2. **Syncs source files** from the read-only Git stage to the writable `CODE_STAGE` via `COPY FILES` (needed because `sproc.register()` requires PUT access)
3. **Registers `MLJobDefinition`** objects from `GIT_JOBS_PATH` (the `jobs/` directory in the Git stage)
4. **Deploys the DAG** with a claim→deploy→release ownership pattern

**Job definitions capture a point-in-time snapshot** of the code when `MLJobDefinition.register()` runs. This means the DAG always runs with the code that was current at deploy time. Every code change requires a full redeployment — which is why the GitHub Actions workflow exists to automate this.

**Key characteristics:**
- Code sourced from Snowflake Git Repository stage (`ML_PIPELINE_GIT_REPO`)
- `MLJobDefinition.register(source=GIT_JOBS_PATH, ...)` reads from Git stage
- Job definitions are **frozen snapshots** — redeployment required for code changes
- GitHub Actions auto-deploys on push to `main` (path filter: `task_graph_ml_stage/**`)
- CI/CD uses `CICD_DEPLOY_RL` role with key-pair auth (service account, no password)
- Claim→deploy→release ownership transfer pattern (`CICD_DEPLOY_RL` → `SPCS_PSE_ROLE`)
- `PIPELINE_OPERATOR_RL` role for monitoring/executing without deploy privileges
- Quality threshold: R2 >= 0.7
- Schema: `CC_INSURANCE_PIPELINE.PIPELINE_STAGE`
- Schedule: Daily 9am UTC

#### Roles

| Role | Purpose | Assigned To |
|------|---------|-------------|
| `CICD_DEPLOY_RL` | Deploy/replace DAG, sync Git repo, manage stages, ownership transfer | `CICD_DEPLOY_USER` (service account) |
| `PIPELINE_OPERATOR_RL` | Monitor, execute, suspend/resume tasks (no deploy) | Human users (e.g., `CCARRERO`) |

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
- `CICD_DEPLOY_RL` role with deploy privileges (including `MANAGE GRANTS` for ownership transfer)
- `CICD_NETWORK_POLICY` allowing GitHub Actions IPs
- `PIPELINE_OPERATOR_RL` role granted to the specified user

3. **Configure GitHub Secrets** in your repository:

| Secret/Variable | Value |
|-----------------|-------|
| `SNOWFLAKE_ACCOUNT` | Your Snowflake account identifier (e.g., `phb14991`) |
| `SNOWFLAKE_USER` | `CICD_DEPLOY_USER` |
| `SNOWFLAKE_PRIVATE_KEY` | Contents of `cicd_rsa_key.p8` |

4. **Deploy**: Push changes to `task_graph_ml_stage/` on `main` branch, or trigger manually via `workflow_dispatch`.

**For local deployment** (without CI/CD):

```bash
cd task_graph_ml_stage/src
python deploy_dag.py            # Deploy in suspended state
python deploy_dag.py --execute  # Deploy and execute immediately
```

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

---

### Monitoring the Pipeline

These queries work across all variants (adjust the schema name for your variant):

```sql
-- Check task history (works for any variant)
SELECT * FROM TABLE(CC_INSURANCE_PIPELINE.INFORMATION_SCHEMA.TASK_HISTORY())
ORDER BY SCHEDULED_TIME DESC
LIMIT 20;

-- View pipeline alerts (adjust schema: PIPELINE, PIPELINE_CODE_STAGE, PIPELINE_MLJOBS,
-- PIPELINE_SUBMIT_FILE, or PIPELINE_STAGE)
SELECT * FROM CC_INSURANCE_PIPELINE.<SCHEMA>.PIPELINE_ALERTS
ORDER BY ALERT_TIME DESC;

-- Resume/execute the DAG manually (adjust schema)
ALTER TASK CC_INSURANCE_PIPELINE.<SCHEMA>.CAR_INSURANCE_ML_PIPELINE RESUME;
EXECUTE TASK CC_INSURANCE_PIPELINE.<SCHEMA>.CAR_INSURANCE_ML_PIPELINE;
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
- **MLJobDefinition**: Reusable job definitions with `submit_directory` and Git stage sources
- **Task Graph (DAG)**: Automated pipeline with DAGTask, DAGTaskBranch, TaskContext, Cron scheduling
- **Git Repository Stage**: Snowflake-native Git integration for code syncing
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

| Parameter | A: `task_graph/` | B: `task_graph_stage/` | C: `task_graph_ml/` | D: `task_graph_ml_directory/` | E: `task_graph_ml_stage/` |
|-----------|-------------------|------------------------|---------------------|-------------------------------|---------------------------|
| Database | `CC_INSURANCE_PIPELINE` | `CC_INSURANCE_PIPELINE` | `CC_INSURANCE_PIPELINE` | `CC_INSURANCE_PIPELINE` | `CC_INSURANCE_PIPELINE` |
| Schema | `PIPELINE` | `PIPELINE_CODE_STAGE` | `PIPELINE_MLJOBS` | `PIPELINE_SUBMIT_FILE` | `PIPELINE_STAGE` |
| Data Schema | `DATA` | `DATA` | `DATA` | `DATA` | `DATA` |
| Compute Pool | `DEMO_POOL` | `DEMO_POOL` | `DEMO_POOL` | `DEMO_POOL` | `DEMO_POOL` |
| DAG Name | `CAR_INSURANCE_ML_PIPELINE` | `CAR_INSURANCE_ML_PIPELINE` | `CAR_INSURANCE_ML_PIPELINE` | `CAR_INSURANCE_ML_PIPELINE` | `CAR_INSURANCE_ML_PIPELINE` |
| Tasks | 7 | 7 | 8 (adds PREPARE_DATA) | 8 (adds PREPARE_DATA) | 8 (adds PREPARE_DATA) |
| Schedule | Daily 6am UTC | Daily 6am UTC | Daily 6am UTC | Daily 6am UTC | Daily 9am UTC |
| Quality Threshold | R2 >= 0.7 | R2 >= 0.7 | R2 >= 0.7 | R2 >= 0.7 | R2 >= 0.7 |
| Task Execution | All `StoredProcedureCall` | All SP (via stage runner) | `@remote` direct + SP | `MLJobDefinition` + SP | `MLJobDefinition` (Git) + SP |
| Code Loading | Stage imports (`@CODE_STAGE`) | `importlib` from `@CODE_STAGE` | Stage imports + `@remote` | `register(source=local_dir)` | `register(source=GIT_STAGE)` |
| Redeploy on Code Change | Yes | **No** (upload only) | Yes | Yes | Yes (auto via CI/CD) |
| Deployment | Manual | Manual | Manual | Manual | CI/CD (GitHub Actions) |
| Auth | Password/SSO/Key-pair | Password/SSO/Key-pair | Password/SSO/Key-pair | Password/SSO/Key-pair | Key-pair (service account) |
