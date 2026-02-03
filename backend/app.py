"""
Car Insurance Pricing Backend Service

This FastAPI service:
1. Receives basic car/customer parameters via HTTP POST
2. Fetches additional customer features from Online Feature Store (if customer exists)
3. Calls the ML model inference endpoint via HTTP
4. Returns the prediction to the caller
"""
import os
import time
import logging
import requests
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATABASE = "CC_ML_INSURANCE"
SCHEMA = "CAR_PRICING"
FEATURE_VIEW_VERSION = "v1"
FEATURE_VIEW_NAME = "CUSTOMER_RISK_FEATURES"

MAX_RETRIES = 5
INITIAL_TIMEOUT = 60
RETRY_TIMEOUT = 120
RETRY_DELAY = 5

_session = None
_feature_store = None
_feature_view = None


def init_snowpark_session():
    """Initialize Snowpark session at startup."""
    global _session
    from snowflake.snowpark import Session
    
    logger.info("[STARTUP] Initializing Snowpark session...")
    
    if os.path.isfile("/snowflake/session/token"):
        connection_params = {
            "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
            "host": os.environ.get("SNOWFLAKE_HOST"),
            "authenticator": "oauth",
            "token": open("/snowflake/session/token").read(),
            "database": DATABASE,
            "schema": SCHEMA,
            "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        }
        logger.info("[STARTUP] Using SPCS OAuth token authentication")
    else:
        connection_params = {
            "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
            "host": os.environ.get("SNOWFLAKE_HOST"),
            "database": DATABASE,
            "schema": SCHEMA,
            "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        }
        logger.info("[STARTUP] Using default authentication")
    
    _session = Session.builder.configs(connection_params).create()
    logger.info("[STARTUP] Snowpark session created successfully")
    return _session


def init_feature_store():
    """Initialize Feature Store and Feature View at startup."""
    global _feature_store, _feature_view, _session
    from snowflake.ml.feature_store import FeatureStore
    
    logger.info("[STARTUP] Initializing Feature Store...")
    
    warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
    _feature_store = FeatureStore(
        session=_session,
        database=DATABASE,
        name=SCHEMA,
        default_warehouse=warehouse
    )
    logger.info("[STARTUP] Feature Store initialized")
    
    logger.info(f"[STARTUP] Loading Feature View {FEATURE_VIEW_NAME}/{FEATURE_VIEW_VERSION}...")
    try:
        _feature_view = _feature_store.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
        logger.info("[STARTUP] Feature View loaded successfully")
    except Exception as e:
        logger.error(f"[STARTUP] Failed to load Feature View {FEATURE_VIEW_NAME}/{FEATURE_VIEW_VERSION}: {e}")
        raise RuntimeError(f"Feature View {FEATURE_VIEW_NAME}/{FEATURE_VIEW_VERSION} not found.") from e
    
    return _feature_store, _feature_view


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all Snowflake resources at startup."""
    logger.info("[STARTUP] ========== Service Starting ==========")
    try:
        init_snowpark_session()
        init_feature_store()
        logger.info("[STARTUP] ========== Initialization Complete ==========")
    except Exception as e:
        logger.error(f"[STARTUP] Failed to initialize: {e}")
        raise
    yield
    logger.info("[SHUTDOWN] Service shutting down")


app = FastAPI(title="Car Insurance Pricing API", lifespan=lifespan)


class InsuranceRequest(BaseModel):
    customer_id: Optional[str] = None
    car_age: int
    kilometers: int
    engine_size: float
    estimated_car_value: float
    age: int
    years_licensed: int
    car_make: str
    car_model: str
    fuel_type: str
    transmission: str
    coverage_type: str
    gender: str
    state: str


class InsuranceResponse(BaseModel):
    predicted_premium: float
    customer_type: str
    features_used: dict
    timing_ms: dict


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.elapsed_ms = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        logger.info(f"[TIMING] {self.name}: {self.elapsed_ms:.2f}ms")


def calculate_risk_features_for_new_customer(age: int, years_licensed: int, credit_score: int = 700) -> dict:
    claims_history = 0
    risk_score = (
        (30 if age < 25 else (20 if age > 65 else 0)) +
        (claims_history * 15) +
        max(0, 40 - years_licensed) +
        max(0, (700 - credit_score) / 10)
    )
    return {
        'CLAIMS_HISTORY': claims_history,
        'CREDIT_SCORE': credit_score,
        'RISK_SCORE': round(risk_score, 2),
        'AVG_CLAIMS_PER_YEAR': 0.0,
        'AVG_DEDUCTIBLE': 500
    }


def get_customer_features_from_feature_store(customer_id: str) -> tuple[Optional[dict], dict]:
    """Retrieve customer features from Online Feature Store using Python API."""
    global _session, _feature_store, _feature_view
    timing = {}
    
    try:
        with Timer("create_spine_df") as t:
            spine_df = _session.create_dataframe([(customer_id,)], schema=["CUSTOMER_ID"])
        timing["create_spine_df_ms"] = t.elapsed_ms
        
        with Timer("retrieve_feature_values") as t:
            customer_features_df = _feature_store.retrieve_feature_values(
                spine_df=spine_df,
                features=[_feature_view]
            ).to_pandas()
        timing["retrieve_feature_values_ms"] = t.elapsed_ms
        
        if len(customer_features_df) > 0:
            row = customer_features_df.iloc[0].to_dict()
            if 'CUSTOMER_ID' in row:
                del row['CUSTOMER_ID']
            logger.info(f"[FEATURE_STORE] Retrieved features for customer {customer_id}: {list(row.keys())}")
            return row, timing
            
    except Exception as e:
        logger.error(f"[FEATURE_STORE] Error fetching customer features: {e}")
        timing["error"] = str(e)[:200]
    
    return None, timing


MODEL_SERVICE_INTERNAL_URL = os.environ.get("MODEL_SERVICE_URL", "http://car-insurance-inference-svc:5000")


def safe_int(value, default=0):
    """Convert value to int, handling NaN and None."""
    import math
    if value is None:
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value, default=0.0):
    """Convert value to float, handling NaN and None."""
    import math
    if value is None:
        return default
    try:
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def call_model_endpoint(features: dict) -> tuple[float, dict]:
    """Call the model inference service via internal SPCS DNS."""
    model_url = MODEL_SERVICE_INTERNAL_URL
    
    row_data = [
        safe_int(features.get('CAR_AGE'), 0),
        safe_int(features.get('KILOMETERS'), 0),
        safe_float(features.get('ENGINE_SIZE'), 2.0),
        safe_float(features.get('ESTIMATED_CAR_VALUE'), 25000),
        safe_int(features.get('AGE'), 30),
        safe_int(features.get('YEARS_LICENSED'), 10),
        safe_int(features.get('CLAIMS_HISTORY'), 0),
        safe_int(features.get('CREDIT_SCORE'), 700),
        safe_float(features.get('RISK_SCORE'), 0),
        safe_float(features.get('AVG_CLAIMS_PER_YEAR'), 0.0),
        safe_int(features.get('TOTAL_POLICIES'), 0),
        safe_float(features.get('AVG_CAR_AGE'), safe_float(features.get('CAR_AGE'), 0)),
        safe_int(features.get('AVG_KILOMETERS'), safe_int(features.get('KILOMETERS'), 0)),
        safe_float(features.get('TOTAL_CAR_VALUE'), safe_float(features.get('ESTIMATED_CAR_VALUE'), 25000)),
        safe_int(features.get('AVG_DEDUCTIBLE'), 500),
        str(features.get('CAR_MAKE') or 'Toyota'),
        str(features.get('CAR_MODEL') or 'Camry'),
        str(features.get('FUEL_TYPE') or 'Gasoline'),
        str(features.get('TRANSMISSION') or 'Automatic'),
        str(features.get('COVERAGE_TYPE') or 'Standard'),
        str(features.get('GENDER') or 'M'),
        str(features.get('STATE') or 'CA')
    ]
    
    payload = {"data": [[0] + row_data]}
    
    logger.info(f"[MODEL] Payload: {payload}")
    headers = {"Content-Type": "application/json"}
    
    timing = {"retries": 0, "total_wait_ms": 0}
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        timeout = INITIAL_TIMEOUT if attempt == 0 else RETRY_TIMEOUT
        timing["retries"] = attempt
        
        logger.info(f"[MODEL] Attempt {attempt + 1}/{MAX_RETRIES} - URL: {model_url}/predict, timeout: {timeout}s")
        
        try:
            with Timer(f"model_inference_attempt_{attempt + 1}") as t:
                response = requests.post(
                    f"{model_url}/predict",
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
            
            timing[f"attempt_{attempt + 1}_ms"] = t.elapsed_ms
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[MODEL] Success on attempt {attempt + 1}, response time: {t.elapsed_ms:.2f}ms")
                logger.info(f"[MODEL] Response: {result}")
                
                if "data" in result and len(result["data"]) > 0:
                    data_row = result["data"][0]
                    if isinstance(data_row, list) and len(data_row) > 1:
                        if isinstance(data_row[1], dict) and "PREDICTED_PREMIUM" in data_row[1]:
                            return data_row[1]["PREDICTED_PREMIUM"], timing
                        return float(data_row[1]), timing
                    if isinstance(data_row, dict) and "PREDICTED_PREMIUM" in data_row:
                        return data_row["PREDICTED_PREMIUM"], timing
                    if isinstance(data_row, list) and len(data_row) > 0:
                        return float(data_row[0]), timing
                
                raise HTTPException(status_code=500, detail=f"Invalid model response format: {result}")
            
            logger.warning(f"[MODEL] Attempt {attempt + 1} failed with status {response.status_code}: {response.text[:500]}")
            last_error = f"Status {response.status_code}: {response.text[:300]}"
            
        except requests.exceptions.Timeout:
            logger.warning(f"[MODEL] Attempt {attempt + 1} timed out after {timeout}s")
            last_error = f"Timeout after {timeout}s"
            timing[f"attempt_{attempt + 1}_ms"] = timeout * 1000
            
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"[MODEL] Attempt {attempt + 1} connection error: {str(e)[:200]}")
            last_error = f"Connection error: {str(e)[:100]}"
            
        except HTTPException:
            raise
            
        except Exception as e:
            logger.error(f"[MODEL] Attempt {attempt + 1} unexpected error: {str(e)}")
            last_error = str(e)[:200]
        
        if attempt < MAX_RETRIES - 1:
            wait_time = RETRY_DELAY * (attempt + 1)
            timing["total_wait_ms"] += wait_time * 1000
            logger.info(f"[MODEL] Waiting {wait_time}s before retry (service may be starting)...")
            time.sleep(wait_time)
    
    raise HTTPException(
        status_code=503,
        detail=f"Model service unavailable after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/")
def root():
    return {"service": "Car Insurance Pricing API", "status": "running"}


@app.post("/predict", response_model=InsuranceResponse)
def predict_premium(request: InsuranceRequest):
    request_start = time.perf_counter()
    timing = {}
    
    logger.info(f"[REQUEST] Starting prediction for customer_id={request.customer_id or 'NEW'}")
    
    with Timer("feature_preparation") as t:
        features = {
            'CAR_AGE': request.car_age,
            'KILOMETERS': request.kilometers,
            'ENGINE_SIZE': request.engine_size,
            'ESTIMATED_CAR_VALUE': request.estimated_car_value,
            'AGE': request.age,
            'YEARS_LICENSED': request.years_licensed,
            'CAR_MAKE': request.car_make,
            'CAR_MODEL': request.car_model,
            'FUEL_TYPE': request.fuel_type,
            'TRANSMISSION': request.transmission,
            'COVERAGE_TYPE': request.coverage_type,
            'GENDER': request.gender,
            'STATE': request.state
        }
    timing["feature_preparation_ms"] = t.elapsed_ms
    
    customer_type = "new"
    
    if request.customer_id:
        with Timer("feature_store_total") as t:
            customer_features, fs_timing = get_customer_features_from_feature_store(request.customer_id)
        timing["feature_store_total_ms"] = t.elapsed_ms
        timing.update({f"fs_{k}": v for k, v in fs_timing.items()})
        
        if customer_features:
            features.update(customer_features)
            customer_type = "existing"
            logger.info(f"[REQUEST] Using existing customer profile")
    
    if customer_type == "new":
        with Timer("risk_calculation") as t:
            risk_features = calculate_risk_features_for_new_customer(
                request.age, request.years_licensed
            )
            features.update(risk_features)
            features['TOTAL_POLICIES'] = 0
            features['AVG_CAR_AGE'] = request.car_age
            features['AVG_KILOMETERS'] = request.kilometers
            features['TOTAL_CAR_VALUE'] = request.estimated_car_value
        timing["risk_calculation_ms"] = t.elapsed_ms
        logger.info(f"[REQUEST] Using new customer defaults")
    
    with Timer("model_inference_total") as t:
        predicted_premium, model_timing = call_model_endpoint(features)
    timing["model_inference_total_ms"] = t.elapsed_ms
    timing.update({f"model_{k}": v for k, v in model_timing.items()})
    
    total_time = (time.perf_counter() - request_start) * 1000
    timing["total_request_ms"] = total_time
    
    logger.info(f"[REQUEST] Completed in {total_time:.2f}ms - Premium: ${predicted_premium:.2f}")
    logger.info(f"[TIMING SUMMARY] {timing}")
    
    return InsuranceResponse(
        predicted_premium=round(predicted_premium, 2),
        customer_type=customer_type,
        features_used=features,
        timing_ms=timing
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
