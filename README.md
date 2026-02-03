# Car Insurance Premium Prediction - Real-Time ML Pipeline

A complete end-to-end ML pipeline for predicting car insurance premiums using:
- **Snowflake Feature Store** with Online Feature Store for real-time serving
- **XGBoost** model with StandardScaler preprocessing
- **Snowflake Model Registry** for model management
- **SPCS (Snowpark Container Services)** for real-time inference

## Architecture

```
┌─────────────────────┐     ┌─────────────────────────┐     ┌──────────────────────┐
│  Frontend App       │────▶│  Backend SPCS Service   │────▶│  Model Inference     │
│  (Streamlit)        │     │  (FastAPI)              │     │  (SPCS Service)      │
└─────────────────────┘     └───────────┬─────────────┘     └──────────────────────┘
                                        │
                                        ▼
                            ┌─────────────────────────┐
                            │  Online Feature Store   │
                            │  (Customer Risk Data)   │
                            └─────────────────────────┘
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
insurance_real_time/
├── car_insurance_ml.ipynb          # Main notebook - sets up everything
├── car_insurance_frontend.py       # Frontend Streamlit app
├── car_insurance_realtime_perf_test.py  # Performance testing app
├── requirements.txt                # Python dependencies (local)
├── environment.yml                 # Conda environment
├── snowflake.yml                   # Snowflake project config
├── backend/
│   ├── app.py                      # FastAPI backend service
│   ├── Dockerfile                  # Docker build file
│   ├── requirements.txt            # Backend dependencies
│   ├── deploy.sh                   # Deployment script
│   └── service-spec.yaml           # SPCS service spec
└── .streamlit/
    ├── secrets.toml                # Local secrets (PAT token)
    └── secrets.toml.example        # Template for secrets
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

-- Drop database (removes all objects)
DROP DATABASE IF EXISTS CC_ML_INSURANCE;
```

## Key Snowflake Features Demonstrated

- **Feature Store**: Entity, Feature View with online serving
- **Online Feature Store**: Sub-second feature retrieval
- **Model Registry**: Version management, metrics tracking
- **SPCS**: Container services for backend and model inference
- **Inference Table**: Model telemetry and monitoring

## Configuration Reference

| Parameter | Value |
|-----------|-------|
| Database | `CC_ML_INSURANCE` |
| Schema | `CAR_PRICING` |
| Feature View | `CUSTOMER_RISK_FEATURES` |
| Model Name | `CAR_INSURANCE_PRICING_MODEL` |
| Backend Service | `INSURANCE_BACKEND_SVC` |
| Model Service | `CAR_INSURANCE_INFERENCE_SVC` |
| Compute Pool | `BACKEND_CPU_POOL` |
| Warehouse | `COMPUTE_WH` |
