"""
Car Insurance Backend Performance Test

Performance testing for the backend SPCS service with real-time visualizations.
Tests the full flow: Backend -> Feature Store -> Model Inference

Supports:
- Streamlit in Snowflake (SiS): Uses OAuth token automatically
- Local development: Uses secrets.toml or manual input

Usage (local):
    streamlit run car_insurance_realtime_perf_test.py
"""

import os
import time
import random
import statistics
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATABASE = "CC_ML_INSURANCE"
SCHEMA = "CAR_PRICING"
BACKEND_SERVICE_NAME = "INSURANCE_BACKEND_SVC"
MODEL_NAME = "CAR_INSURANCE_PRICING_MODEL"

CAR_DATA = {
    'Toyota': ['Camry', 'Corolla', 'RAV4', 'Highlander', 'Prius'],
    'Honda': ['Civic', 'Accord', 'CR-V', 'Pilot', 'Odyssey'],
    'Ford': ['F-150', 'Mustang', 'Explorer', 'Escape', 'Bronco'],
    'BMW': ['3 Series', '5 Series', 'X3', 'X5', 'M3'],
    'Mercedes': ['C-Class', 'E-Class', 'GLC', 'GLE', 'S-Class'],
    'Chevrolet': ['Silverado', 'Malibu', 'Equinox', 'Tahoe', 'Corvette'],
    'Tesla': ['Model 3', 'Model Y', 'Model S', 'Model X'],
    'Nissan': ['Altima', 'Rogue', 'Sentra', 'Pathfinder', 'Maxima']
}
COVERAGE_TYPES = ['Basic', 'Standard', 'Premium', 'Comprehensive']
FUEL_TYPES = ['Gasoline', 'Diesel', 'Hybrid', 'Electric']
TRANSMISSIONS = ['Automatic', 'Manual', 'CVT']
STATES = ['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
ENGINE_SIZES = [1.5, 1.8, 2.0, 2.4, 2.5, 3.0, 3.5, 4.0]

if 'result_queue' not in st.session_state:
    st.session_state.result_queue = queue.Queue()
if 'test_running' not in st.session_state:
    st.session_state.test_running = False
if 'test_results' not in st.session_state:
    st.session_state.test_results = []
if 'test_thread' not in st.session_state:
    st.session_state.test_thread = None
if 'test_start_time' not in st.session_state:
    st.session_state.test_start_time = None
if 'current_concurrency' not in st.session_state:
    st.session_state.current_concurrency = 5
if 'total_requests_sent' not in st.session_state:
    st.session_state.total_requests_sent = 0
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()
if 'customers' not in st.session_state:
    st.session_state.customers = []


def is_running_in_sis():
    """Check if running in Streamlit in Snowflake."""
    return os.path.exists("/snowflake/session/token")


def get_snowflake_connection():
    """Get Snowflake connection - works in SiS or locally."""
    if is_running_in_sis():
        from snowflake.snowpark.context import get_active_session
        return get_active_session().connection
    else:
        import snowflake.connector
        connection_name = os.environ.get("SNOWFLAKE_CONNECTION_NAME", "talent_keypair")
        return snowflake.connector.connect(connection_name=connection_name)


def get_auth_token():
    """Get authentication token - OAuth in SiS, PAT locally."""
    if is_running_in_sis():
        with open("/snowflake/session/token", "r") as f:
            return f.read().strip()
    else:
        try:
            if "snowflake" in st.secrets and "pat_token" in st.secrets["snowflake"]:
                return st.secrets["snowflake"]["pat_token"]
        except Exception:
            pass
        return st.session_state.get("pat_token", "")


@st.cache_data(ttl=60)
def get_backend_url():
    """Discover backend URL from SHOW ENDPOINTS."""
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        cursor.execute(f"SHOW ENDPOINTS IN SERVICE {DATABASE}.{SCHEMA}.{BACKEND_SERVICE_NAME}")
        result = cursor.fetchone()
        if result:
            ingress_url = result[5]
            if ingress_url and not ingress_url.startswith("Endpoints provisioning"):
                return f"https://{ingress_url}"
    except Exception as e:
        pass
    return ""


@st.cache_data(ttl=300)
def get_customers():
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT CUSTOMER_ID FROM {DATABASE}.{SCHEMA}.CUSTOMERS
            ORDER BY CUSTOMER_ID LIMIT 100
        """)
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        st.error(f"Could not get customers: {e}")
        return []


def get_inference_table_stats(start_time):
    """Query INFERENCE_TABLE for model telemetry data since test start."""
    if not start_time:
        return None
    
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        
        query = f"""
        SELECT 
            TIMESTAMP,
            RESOURCE_ATTRIBUTES:"snow.query.id"::STRING as query_id,
            RESOURCE_ATTRIBUTES:"snow.service.name"::STRING as service_name,
            RECORD_ATTRIBUTES:"snow.model_serving.request.timestamp"::TIMESTAMP_NTZ as request_timestamp,
            RECORD_ATTRIBUTES:"snow.model_serving.response.timestamp"::TIMESTAMP_NTZ as response_timestamp,
            TIMESTAMPDIFF('millisecond', 
                RECORD_ATTRIBUTES:"snow.model_serving.request.timestamp"::TIMESTAMP_NTZ,
                RECORD_ATTRIBUTES:"snow.model_serving.response.timestamp"::TIMESTAMP_NTZ
            ) as inference_latency_ms
        FROM TABLE(INFERENCE_TABLE('{DATABASE}.{SCHEMA}.{MODEL_NAME}'))
        WHERE TIMESTAMP >= DATEADD('second', -{int(elapsed_seconds + 60)}, CURRENT_TIMESTAMP())
        ORDER BY TIMESTAMP DESC
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        
        if not rows:
            return None
        
        data = [dict(zip(columns, row)) for row in rows]
        latencies = [r['INFERENCE_LATENCY_MS'] for r in data if r['INFERENCE_LATENCY_MS'] is not None]
        
        if not latencies:
            return None
        
        return {
            "total_inferences": len(data),
            "avg_latency_ms": statistics.mean(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p50_latency_ms": calculate_percentile(latencies, 50),
            "p95_latency_ms": calculate_percentile(latencies, 95),
            "p99_latency_ms": calculate_percentile(latencies, 99),
            "unique_queries": len(set(r['QUERY_ID'] for r in data if r['QUERY_ID'])),
            "service_name": data[0].get('SERVICE_NAME', 'Unknown') if data else 'Unknown',
            "raw_data": data[:100]
        }
    except Exception as e:
        return {"error": str(e)}


def generate_test_request(customer_id=None):
    """Generate a test request matching the backend API format."""
    age = random.randint(18, 80)
    years_licensed = random.randint(1, min(age - 16, 60))
    car_age = random.randint(0, 15)
    kilometers = random.randint(0, 300000)
    
    car_make = random.choice(list(CAR_DATA.keys()))
    car_model = random.choice(CAR_DATA[car_make])
    fuel_type = 'Electric' if car_make == 'Tesla' else random.choice(FUEL_TYPES)
    
    base_prices = {'Toyota': 28000, 'Honda': 27000, 'Ford': 35000, 'BMW': 55000,
                   'Mercedes': 60000, 'Chevrolet': 32000, 'Tesla': 50000, 'Nissan': 26000}
    base_price = base_prices.get(car_make, 30000)
    estimated_car_value = base_price * (0.85 ** car_age) * max(0.5, 1 - (kilometers / 300000))
    
    return {
        "customer_id": customer_id,
        "car_age": car_age,
        "kilometers": kilometers,
        "engine_size": random.choice(ENGINE_SIZES),
        "estimated_car_value": round(estimated_car_value, 2),
        "age": age,
        "years_licensed": years_licensed,
        "car_make": car_make,
        "car_model": car_model,
        "fuel_type": fuel_type,
        "transmission": random.choice(TRANSMISSIONS),
        "coverage_type": random.choice(COVERAGE_TYPES),
        "gender": random.choice(['M', 'F']),
        "state": random.choice(STATES)
    }


def make_request(backend_url, headers, request_data, request_id):
    """Make a single request to the backend service."""
    start_time = time.perf_counter()
    try:
        response = requests.post(
            f"{backend_url}/predict",
            headers=headers,
            json=request_data,
            timeout=60
        )
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            timing = result.get("timing_ms", {})
            return {
                "request_id": request_id,
                "timestamp": datetime.now(),
                "status": "success",
                "latency_ms": latency_ms,
                "status_code": response.status_code,
                "predicted_premium": result.get("predicted_premium", 0),
                "customer_type": result.get("customer_type", "unknown"),
                "customer_id": request_data.get("customer_id"),
                "timing": timing,
                "feature_store_ms": timing.get("feature_store_total_ms", 0),
                "model_inference_ms": timing.get("model_inference_total_ms", 0),
                "input_sample": {
                    "car_make": request_data["car_make"],
                    "age": request_data["age"],
                    "car_age": request_data["car_age"]
                }
            }
        
        return {
            "request_id": request_id,
            "timestamp": datetime.now(),
            "status": "error",
            "latency_ms": latency_ms,
            "status_code": response.status_code,
            "error": response.text[:300]
        }
        
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "request_id": request_id,
            "timestamp": datetime.now(),
            "status": "error",
            "latency_ms": (end_time - start_time) * 1000,
            "status_code": 0,
            "error": str(e)[:300]
        }


def continuous_performance_test(backend_url, pat_token, result_queue, stop_event, concurrency_ref, customers, use_existing_customers):
    """Run continuous performance test in background thread."""
    headers = {
        "Authorization": f'Snowflake Token="{pat_token}"',
        "Content-Type": "application/json"
    }
    
    request_counter = 0
    
    while not stop_event.is_set():
        try:
            current_concurrency = concurrency_ref[0] if concurrency_ref else 5
            
            with ThreadPoolExecutor(max_workers=current_concurrency) as executor:
                futures = []
                for _ in range(current_concurrency):
                    if stop_event.is_set():
                        break
                    
                    customer_id = None
                    if use_existing_customers and customers:
                        customer_id = random.choice(customers)
                    
                    request_data = generate_test_request(customer_id)
                    future = executor.submit(
                        make_request, backend_url, headers, request_data, request_counter
                    )
                    futures.append(future)
                    request_counter += 1
                
                for future in as_completed(futures):
                    if stop_event.is_set():
                        break
                    try:
                        result = future.result(timeout=65)
                        result_queue.put(result)
                    except Exception as e:
                        result_queue.put({
                            "request_id": request_counter,
                            "timestamp": datetime.now(),
                            "status": "error",
                            "latency_ms": 0,
                            "status_code": 0,
                            "error": str(e)[:100]
                        })
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in test loop: {e}")
            time.sleep(1)


def process_queued_data():
    """Process data from queue and update session state."""
    new_results = []
    while not st.session_state.result_queue.empty():
        try:
            result = st.session_state.result_queue.get_nowait()
            new_results.append(result)
        except queue.Empty:
            break
    
    if new_results:
        st.session_state.test_results.extend(new_results)
        st.session_state.total_requests_sent += len(new_results)
        
        if len(st.session_state.test_results) > 1000:
            st.session_state.test_results = st.session_state.test_results[-1000:]


def calculate_percentile(data, percentile):
    if not data:
        return 0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


def get_recent_stats(window_seconds=30):
    """Get statistics for recent requests within the time window."""
    if not st.session_state.test_results:
        return None
    
    cutoff_time = datetime.now().timestamp() - window_seconds
    recent_results = [
        r for r in st.session_state.test_results 
        if r["timestamp"].timestamp() > cutoff_time
    ]
    
    if not recent_results:
        return None
    
    successful = [r for r in recent_results if r["status"] == "success"]
    failed = [r for r in recent_results if r["status"] == "error"]
    latencies = [r["latency_ms"] for r in successful]
    fs_latencies = [r.get("feature_store_ms", 0) for r in successful if r.get("feature_store_ms")]
    model_latencies = [r.get("model_inference_ms", 0) for r in successful if r.get("model_inference_ms")]
    
    return {
        "total": len(recent_results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(recent_results) * 100 if recent_results else 0,
        "throughput": len(recent_results) / window_seconds,
        "avg_latency": statistics.mean(latencies) if latencies else 0,
        "p95_latency": calculate_percentile(latencies, 95),
        "p99_latency": calculate_percentile(latencies, 99),
        "avg_feature_store_ms": statistics.mean(fs_latencies) if fs_latencies else 0,
        "avg_model_inference_ms": statistics.mean(model_latencies) if model_latencies else 0
    }


st.set_page_config(page_title="Backend Performance Test", page_icon="⚡", layout="wide")

st.title("⚡ Backend Service Performance Test")
st.markdown("Performance testing for the insurance backend service (Feature Store + Model Inference)")

with st.sidebar:
    st.subheader("🔐 Authentication")
    
    running_in_sis = is_running_in_sis()
    backend_url = get_backend_url()
    
    if running_in_sis:
        st.success("✅ Running in Snowflake (OAuth)")
    else:
        try:
            has_secrets = "snowflake" in st.secrets
        except Exception:
            has_secrets = False
        
        if has_secrets and "pat_token" in st.secrets["snowflake"]:
            st.success("✅ PAT Token from secrets")
        else:
            pat_token_input = st.text_input(
                "PAT Token", 
                type="password",
                help="Programmatic Access Token for authentication"
            )
            st.session_state.pat_token = pat_token_input
    
    if backend_url:
        st.success(f"✅ Backend: {backend_url[:40]}...")
    else:
        st.error("❌ Backend not available")
    
    st.divider()
    
    st.subheader("⚙️ Test Configuration")
    
    use_existing_customers = st.checkbox(
        "Use Existing Customers",
        value=True,
        help="Use real customer IDs to test Feature Store lookup"
    )
    
    if use_existing_customers and not st.session_state.customers:
        st.session_state.customers = get_customers()
        st.info(f"Loaded {len(st.session_state.customers)} customers")
    
    new_concurrency = st.slider(
        "Concurrent Workers",
        min_value=1,
        max_value=20,
        value=st.session_state.current_concurrency,
        step=1
    )
    
    if new_concurrency != st.session_state.current_concurrency:
        st.session_state.current_concurrency = new_concurrency
    
    st.divider()
    
    if st.button("🔍 Test Single Request"):
        auth_token = get_auth_token()
        if not auth_token:
            st.error("Please enter a PAT token first (or configure secrets.toml).")
        elif not backend_url:
            st.error("Backend URL not available.")
        else:
            with st.spinner("Testing..."):
                customer_id = random.choice(st.session_state.customers) if use_existing_customers and st.session_state.customers else None
                request_data = generate_test_request(customer_id)
                
                st.write("**Request:**")
                st.json(request_data)
                
                headers = {
                    "Authorization": f'Snowflake Token="{auth_token}"',
                    "Content-Type": "application/json"
                }
                
                start = time.perf_counter()
                response = requests.post(
                    f"{backend_url}/predict",
                    headers=headers,
                    json=request_data,
                    timeout=60
                )
                latency = (time.perf_counter() - start) * 1000
                
                st.write(f"**Status:** {response.status_code}")
                st.write(f"**Latency:** {latency:.1f}ms")
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Premium: ${result['predicted_premium']:,.2f}")
                    st.write(f"Customer Type: {result['customer_type']}")
                    
                    with st.expander("Timing Details"):
                        st.json(result.get('timing_ms', {}))
                else:
                    st.error(response.text)

col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    if st.button("🚀 Start Test", type="primary", disabled=st.session_state.test_running):
        auth_token = get_auth_token()
        backend_url = get_backend_url()
        if not auth_token:
            st.error("Please enter a PAT token first (or configure secrets.toml).")
        elif not backend_url:
            st.error("Backend URL not available.")
        else:
            st.session_state.test_running = True
            st.session_state.test_start_time = datetime.now()
            st.session_state.total_requests_sent = 0
            st.session_state.test_results = []
            st.session_state.stop_event.clear()
            
            while not st.session_state.result_queue.empty():
                st.session_state.result_queue.get()
            
            concurrency_ref = [st.session_state.current_concurrency]
            
            st.session_state.test_thread = threading.Thread(
                target=continuous_performance_test,
                args=(backend_url, auth_token, st.session_state.result_queue,
                      st.session_state.stop_event, concurrency_ref,
                      st.session_state.customers, use_existing_customers),
                daemon=True
            )
            
            st.session_state.test_thread.start()
            st.session_state.concurrency_ref = concurrency_ref
            st.success("✅ Test started!")
            st.rerun()

with col2:
    if st.button("⏹️ Stop Test", type="secondary", disabled=not st.session_state.test_running):
        st.session_state.test_running = False
        st.session_state.stop_event.set()
        st.info("⏹️ Test stopped!")

with col3:
    if st.button("🔄 Reset", type="secondary"):
        st.session_state.test_running = False
        st.session_state.stop_event.set()
        st.session_state.test_results = []
        st.session_state.total_requests_sent = 0
        st.session_state.test_start_time = None
        st.info("🔄 Reset complete!")

if st.session_state.test_running:
    st.success("🟢 **Test Running**")
else:
    st.info("🔴 **Test Stopped** - Click 'Start Test' to begin")

if st.session_state.test_running:
    process_queued_data()
    
    if hasattr(st.session_state, 'concurrency_ref') and st.session_state.concurrency_ref:
        st.session_state.concurrency_ref[0] = st.session_state.current_concurrency
    
    time.sleep(1)
    st.rerun()

if st.session_state.test_results:
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_time = (datetime.now() - st.session_state.test_start_time).total_seconds() if st.session_state.test_start_time else 0
        st.metric("Test Duration", f"{total_time:.1f}s")
        st.metric("Total Requests", st.session_state.total_requests_sent)
    
    with col2:
        successful = [r for r in st.session_state.test_results if r["status"] == "success"]
        failed = [r for r in st.session_state.test_results if r["status"] == "error"]
        st.metric("Successful", len(successful))
        st.metric("Failed", len(failed))
    
    with col3:
        success_rate = len(successful) / len(st.session_state.test_results) * 100 if st.session_state.test_results else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
        overall_throughput = len(st.session_state.test_results) / total_time if total_time > 0 else 0
        st.metric("Throughput", f"{overall_throughput:.1f} req/s")
    
    with col4:
        existing_customers = [r for r in successful if r.get("customer_type") == "existing"]
        new_customers = [r for r in successful if r.get("customer_type") == "new"]
        st.metric("Existing Customers", len(existing_customers))
        st.metric("New Customers", len(new_customers))
    
    st.subheader("📊 Recent Performance (Last 30 seconds)")
    
    recent_stats = get_recent_stats(30)
    if recent_stats:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Requests", recent_stats["total"])
        with col2:
            st.metric("Success Rate", f"{recent_stats['success_rate']:.1f}%")
        with col3:
            st.metric("Avg Latency", f"{recent_stats['avg_latency']:.0f}ms")
        with col4:
            st.metric("P95 Latency", f"{recent_stats['p95_latency']:.0f}ms")
        with col5:
            st.metric("Avg FS Time", f"{recent_stats['avg_feature_store_ms']:.0f}ms")
        with col6:
            st.metric("Avg Model Time", f"{recent_stats['avg_model_inference_ms']:.0f}ms")
    
    if successful:
        st.subheader("🎯 Recent Predictions")
        
        prediction_data = []
        for result in successful[-10:]:
            prediction_data.append({
                "ID": result["request_id"],
                "Customer": result.get("customer_id") or "New",
                "Type": result.get("customer_type", ""),
                "Premium": f"${result['predicted_premium']:,.2f}",
                "Total (ms)": f"{result['latency_ms']:.0f}",
                "FS (ms)": f"{result.get('feature_store_ms', 0):.0f}",
                "Model (ms)": f"{result.get('model_inference_ms', 0):.0f}",
                "Car": result.get('input_sample', {}).get('car_make', ''),
            })
        
        st.dataframe(pd.DataFrame(prediction_data), hide_index=True, use_container_width=True)
    
    if len(st.session_state.test_results) > 10:
        st.subheader("📈 Performance Visualizations")
        
        df_results = pd.DataFrame(st.session_state.test_results[-200:])
        df_results['timestamp_str'] = df_results['timestamp'].dt.strftime('%H:%M:%S')
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Latency', 'Component Latency Breakdown', 'Throughput', 'Latency Distribution')
        )
        
        successful_df = df_results[df_results['status'] == 'success']
        if not successful_df.empty:
            fig.add_trace(
                go.Scatter(x=successful_df['timestamp_str'], y=successful_df['latency_ms'],
                          mode='lines', name='Total Latency', line=dict(color='blue')),
                row=1, col=1
            )
            
            if 'feature_store_ms' in successful_df.columns:
                fig.add_trace(
                    go.Scatter(x=successful_df['timestamp_str'], y=successful_df['feature_store_ms'],
                              mode='lines', name='Feature Store', line=dict(color='green')),
                    row=1, col=2
                )
            if 'model_inference_ms' in successful_df.columns:
                fig.add_trace(
                    go.Scatter(x=successful_df['timestamp_str'], y=successful_df['model_inference_ms'],
                              mode='lines', name='Model Inference', line=dict(color='orange')),
                    row=1, col=2
                )
            
            fig.add_trace(
                go.Histogram(x=successful_df['latency_ms'], name='Latency', nbinsx=20,
                            marker_color='purple', opacity=0.7),
                row=2, col=2
            )
        
        if len(df_results) > 10:
            window_size = min(20, len(df_results) // 4)
            throughput_data = []
            timestamps = []
            
            for i in range(window_size, len(df_results)):
                window_start = df_results.iloc[i-window_size]['timestamp']
                window_end = df_results.iloc[i]['timestamp']
                window_duration = (window_end - window_start).total_seconds()
                
                if window_duration > 0:
                    throughput_data.append(window_size / window_duration)
                    timestamps.append(df_results.iloc[i]['timestamp_str'])
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=throughput_data, mode='lines',
                          name='Throughput', line=dict(color='red')),
                row=2, col=1
            )
        
        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    if failed:
        with st.expander("❌ Error Analysis"):
            error_counts = {}
            for r in failed:
                error = r.get("error", "Unknown")[:80]
                error_counts[error] = error_counts.get(error, 0) + 1
            
            error_df = pd.DataFrame([
                {"Error": k, "Count": v} 
                for k, v in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(error_df, hide_index=True, use_container_width=True)
    
    st.divider()
    st.subheader("🔬 Model Inference Telemetry (from INFERENCE_TABLE)")
    
    if st.button("🔄 Refresh Inference Stats"):
        st.cache_data.clear()
    
    inference_stats = get_inference_table_stats(st.session_state.test_start_time)
    
    if inference_stats:
        if "error" in inference_stats:
            st.error(f"Error querying INFERENCE_TABLE: {inference_stats['error']}")
        else:
            st.success(f"Service: **{inference_stats['service_name']}** | Total Inferences: **{inference_stats['total_inferences']}** | Unique Queries: **{inference_stats['unique_queries']}**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Inference", f"{inference_stats['avg_latency_ms']:.1f}ms")
            with col2:
                st.metric("P50 Inference", f"{inference_stats['p50_latency_ms']:.0f}ms")
            with col3:
                st.metric("P95 Inference", f"{inference_stats['p95_latency_ms']:.0f}ms")
            with col4:
                st.metric("P99 Inference", f"{inference_stats['p99_latency_ms']:.0f}ms")
            
            col5, col6 = st.columns(2)
            with col5:
                st.metric("Min Inference", f"{inference_stats['min_latency_ms']:.0f}ms")
            with col6:
                st.metric("Max Inference", f"{inference_stats['max_latency_ms']:.0f}ms")
            
            with st.expander("📋 Raw Inference Data (Last 100)"):
                raw_df = pd.DataFrame(inference_stats['raw_data'])
                if not raw_df.empty:
                    display_cols = ['TIMESTAMP', 'QUERY_ID', 'INFERENCE_LATENCY_MS']
                    available_cols = [c for c in display_cols if c in raw_df.columns]
                    st.dataframe(raw_df[available_cols], hide_index=True, use_container_width=True)
            
            if inference_stats['raw_data']:
                latencies = [r['INFERENCE_LATENCY_MS'] for r in inference_stats['raw_data'] if r['INFERENCE_LATENCY_MS']]
                if latencies:
                    fig_inf = go.Figure()
                    fig_inf.add_trace(go.Histogram(
                        x=latencies,
                        nbinsx=30,
                        name='Model Inference Latency',
                        marker_color='teal'
                    ))
                    fig_inf.update_layout(
                        title='Model Inference Latency Distribution (from INFERENCE_TABLE)',
                        xaxis_title='Latency (ms)',
                        yaxis_title='Count',
                        height=300
                    )
                    st.plotly_chart(fig_inf, use_container_width=True)
    else:
        st.info("No inference telemetry data available yet. Run a test first.")

st.divider()
st.caption("Backend Performance Test | Feature Store + Model Inference")
