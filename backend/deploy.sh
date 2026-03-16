#!/bin/bash
set -e

SNOWFLAKE_CONNECTION="keypair"
DATABASE="CC_ML_INSURANCE"
SCHEMA="CAR_PRICING"
IMAGE_REPO="yourownaccount.registry.snowflakecomputing.com/cc_ml_insurance/car_pricing/images"
IMAGE_NAME="insurance-backend"
IMAGE_TAG="latest"
SERVICE_NAME="INSURANCE_BACKEND_SVC"
COMPUTE_POOL="BACKEND_CPU_POOL"

echo "=========================================="
echo "Backend Deployment Script"
echo "=========================================="

cd "$(dirname "$0")"

echo ""
echo "[1/9] Ensuring image repository exists..."
snow sql -c ${SNOWFLAKE_CONNECTION} -q "CREATE IMAGE REPOSITORY IF NOT EXISTS ${DATABASE}.${SCHEMA}.IMAGES;"

echo ""
echo "[2/9] Creating compute pool ${COMPUTE_POOL}..."
snow sql -c ${SNOWFLAKE_CONNECTION} -q "
CREATE COMPUTE POOL IF NOT EXISTS ${COMPUTE_POOL}
  MIN_NODES = 1
  MAX_NODES = 2
  INSTANCE_FAMILY = CPU_X64_S
  AUTO_RESUME = TRUE
  AUTO_SUSPEND_SECS = 7200;
"

echo ""
echo "[3/9] Resuming compute pool (in case it was suspended)..."
snow sql -c ${SNOWFLAKE_CONNECTION} -q "ALTER COMPUTE POOL ${COMPUTE_POOL} RESUME;" || true

echo ""
echo "[4/9] Waiting for compute pool to be active..."
for i in {1..30}; do
    STATE=$(snow sql -c ${SNOWFLAKE_CONNECTION} -q "SELECT state FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))" --format json 2>/dev/null | grep -o '"state":"[^"]*"' | head -1 | cut -d'"' -f4 || echo "UNKNOWN")
    
    POOL_STATE=$(snow sql -c ${SNOWFLAKE_CONNECTION} -q "DESCRIBE COMPUTE POOL ${COMPUTE_POOL}" --format json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['state'] if d else 'UNKNOWN')" 2>/dev/null || echo "UNKNOWN")
    
    echo "  Compute pool state: ${POOL_STATE}"
    if [ "$POOL_STATE" = "ACTIVE" ] || [ "$POOL_STATE" = "IDLE" ]; then
        echo "  Compute pool is ready!"
        break
    fi
    sleep 10
done

echo ""
echo "[5/9] Logging into Snowflake image registry..."
snow spcs image-registry login -c ${SNOWFLAKE_CONNECTION}

echo ""
echo "[6/9] Building Docker image..."
docker build --platform linux/amd64 -t ${IMAGE_REPO}/${IMAGE_NAME}:${IMAGE_TAG} .

echo ""
echo "[7/9] Pushing image to Snowflake registry..."
docker push ${IMAGE_REPO}/${IMAGE_NAME}:${IMAGE_TAG}

echo ""
echo "[8/10] Dropping existing service if exists..."
snow sql -c ${SNOWFLAKE_CONNECTION} -q "DROP SERVICE IF EXISTS ${DATABASE}.${SCHEMA}.${SERVICE_NAME};"

echo ""
echo "[9/10] Getting model service DNS name..."
MODEL_SERVICE_NAME="CAR_INSURANCE_INFERENCE_SVC"
MODEL_DNS=$(snow sql -c ${SNOWFLAKE_CONNECTION} -q "SELECT dns_name FROM TABLE(INFORMATION_SCHEMA.SERVICE_INSTANCES('${DATABASE}.${SCHEMA}.${MODEL_SERVICE_NAME}')) LIMIT 1" --format json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['DNS_NAME'] if d else '')" 2>/dev/null || echo "")

if [ -z "$MODEL_DNS" ]; then
    MODEL_DNS=$(snow sql -c ${SNOWFLAKE_CONNECTION} -q "SHOW SERVICES LIKE '${MODEL_SERVICE_NAME}' IN SCHEMA ${DATABASE}.${SCHEMA}" --format json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['dns_name'] if d else '')" 2>/dev/null || echo "")
fi

if [ -z "$MODEL_DNS" ]; then
    echo "ERROR: Could not find model service DNS. Make sure CAR_INSURANCE_INFERENCE_SVC is running."
    exit 1
fi

MODEL_SERVICE_URL="http://${MODEL_DNS}:5000"
echo "  Model service URL: ${MODEL_SERVICE_URL}"

echo ""
echo "[10/10] Creating service..."
snow sql -c ${SNOWFLAKE_CONNECTION} -q "
CREATE SERVICE ${DATABASE}.${SCHEMA}.${SERVICE_NAME}
  IN COMPUTE POOL ${COMPUTE_POOL}
  FROM SPECIFICATION \$\$
spec:
  containers:
  - name: insurance-backend
    image: /${DATABASE}/${SCHEMA}/images/${IMAGE_NAME}:${IMAGE_TAG}
    env:
      SNOWFLAKE_WAREHOUSE: COMPUTE_WH
      MODEL_SERVICE_URL: ${MODEL_SERVICE_URL}
    resources:
      requests:
        memory: 1Gi
        cpu: 500m
      limits:
        memory: 2Gi
        cpu: 1000m
    readinessProbe:
      port: 8080
      path: /health
  endpoints:
  - name: api
    port: 8080
    public: true
\$\$
MIN_INSTANCES = 1
MAX_INSTANCES = 1;
"

echo ""
echo "[*] Granting endpoint access..."
snow sql -c ${SNOWFLAKE_CONNECTION} -q "GRANT SERVICE ROLE ${DATABASE}.${SCHEMA}.${SERVICE_NAME}!ALL_ENDPOINTS_USAGE TO ROLE SPCS_PSE_ROLE;"

echo ""
echo "=========================================="
echo "Waiting for service to be ready..."
echo "=========================================="
for i in {1..60}; do
    STATUS=$(snow sql -c ${SNOWFLAKE_CONNECTION} -q "CALL SYSTEM\$GET_SERVICE_STATUS('${DATABASE}.${SCHEMA}.${SERVICE_NAME}')" --format json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['SYSTEM\$GET_SERVICE_STATUS'])" 2>/dev/null || echo "[]")
    
    if echo "$STATUS" | grep -q '"status":"READY"'; then
        echo "Service is READY!"
        break
    else
        echo "  Waiting... (attempt $i/60)"
        sleep 5
    fi
done

echo ""
echo "=========================================="
echo "Getting service endpoint URL..."
echo "=========================================="
snow sql -c ${SNOWFLAKE_CONNECTION} -q "SHOW ENDPOINTS IN SERVICE ${DATABASE}.${SCHEMA}.${SERVICE_NAME};"

echo ""
echo "Deployment complete!"
