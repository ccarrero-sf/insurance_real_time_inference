"""
Car Insurance Premium Calculator - Frontend

Streamlit app that collects car and driver information,
calls the backend SPCS service, and displays the predicted premium.

Supports:
- Streamlit in Snowflake (SiS): Uses OAuth token automatically
- Local development: Uses secrets.toml or manual input

Usage (local):
    streamlit run car_insurance_frontend.py
"""
import os
import streamlit as st
import requests

DATABASE = "CC_ML_INSURANCE"
SCHEMA = "CAR_PRICING"
BACKEND_SERVICE_NAME = "INSURANCE_BACKEND_SVC"

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
        st.warning(f"Could not discover backend URL: {e}")
    return ""


@st.cache_data(ttl=300)
def get_customers():
    """Fetch customers from Snowflake."""
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT CUSTOMER_ID, FIRST_NAME, LAST_NAME, AGE, GENDER, YEARS_LICENSED, STATE
        FROM {DATABASE}.{SCHEMA}.CUSTOMERS
        ORDER BY CUSTOMER_ID
    """)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    cursor.close()
    return [dict(zip(columns, row)) for row in rows]


st.set_page_config(
    page_title="Car Insurance Calculator",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Insurance Premium Calculator")
st.markdown("Get an instant quote for your car insurance premium")

with st.sidebar:
    st.header("Configuration")
    
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
            pat_token = st.text_input(
                "PAT Token",
                type="password",
                help="Programmatic Access Token for authentication"
            )
            st.session_state.pat_token = pat_token
    
    st.divider()
    if backend_url:
        st.success(f"✅ Backend: {backend_url[:40]}...")
    else:
        st.error("❌ Backend service not available")
    
    st.divider()
    st.caption("💡 Tip: Create `.streamlit/secrets.toml` with:")
    st.code("""[snowflake]
pat_token = "your-pat-token"
""", language="toml")

try:
    customers = get_customers()
    customer_options = ["-- New Customer --"] + [
        f"{c['CUSTOMER_ID']} - {c['FIRST_NAME']} {c['LAST_NAME']} ({c['AGE']}y, {c['STATE']})"
        for c in customers
    ]
    customers_loaded = True
except Exception as e:
    st.warning(f"Could not load customers from Snowflake: {e}")
    customer_options = ["-- New Customer --"]
    customers_loaded = False
    customers = []

st.subheader("Customer Selection")
selected_customer_option = st.selectbox(
    "Select Customer",
    customer_options,
    help="Select an existing customer or create a new one"
)

is_new_customer = selected_customer_option == "-- New Customer --"

if not is_new_customer and customers_loaded:
    selected_customer_id = selected_customer_option.split(" - ")[0]
    selected_customer = next((c for c in customers if c['CUSTOMER_ID'] == selected_customer_id), None)
else:
    selected_customer = None
    selected_customer_id = None

col1, col2 = st.columns(2)

with col1:
    st.subheader("Driver Information")
    
    if is_new_customer:
        customer_id = None
        age = st.slider("Driver Age", min_value=18, max_value=80, value=35)
        years_licensed = st.slider(
            "Years Licensed",
            min_value=1,
            max_value=min(age - 16, 60),
            value=min(10, age - 18)
        )
        gender = st.selectbox("Gender", ["M", "F"])
        state = st.selectbox("State", STATES)
    else:
        customer_id = selected_customer_id
        st.info(f"**Customer ID:** {selected_customer['CUSTOMER_ID']}")
        st.info(f"**Name:** {selected_customer['FIRST_NAME']} {selected_customer['LAST_NAME']}")
        
        age = selected_customer['AGE']
        years_licensed = selected_customer['YEARS_LICENSED']
        gender = selected_customer['GENDER']
        state = selected_customer['STATE']
        
        st.metric("Age", age)
        st.metric("Years Licensed", years_licensed)
        st.metric("Gender", gender)
        st.metric("State", state)

with col2:
    st.subheader("Vehicle Information")
    
    car_make = st.selectbox("Car Make", list(CAR_DATA.keys()))
    car_model = st.selectbox("Car Model", CAR_DATA[car_make])
    
    car_age = st.slider("Car Age (years)", min_value=0, max_value=15, value=3)
    kilometers = st.number_input(
        "Kilometers",
        min_value=0,
        max_value=300000,
        value=30000,
        step=5000
    )
    
    engine_size = st.selectbox("Engine Size (L)", ENGINE_SIZES, index=2)
    
    fuel_type = 'Electric' if car_make == 'Tesla' else st.selectbox("Fuel Type", FUEL_TYPES)
    transmission = st.selectbox("Transmission", TRANSMISSIONS)

st.subheader("Coverage Options")
col3, col4 = st.columns(2)

with col3:
    coverage_type = st.selectbox("Coverage Type", COVERAGE_TYPES, index=1)

with col4:
    base_prices = {
        'Toyota': 28000, 'Honda': 27000, 'Ford': 35000, 'BMW': 55000,
        'Mercedes': 60000, 'Chevrolet': 32000, 'Tesla': 50000, 'Nissan': 26000
    }
    base_price = base_prices.get(car_make, 30000)
    estimated_value = base_price * (0.85 ** car_age) * max(0.5, 1 - (kilometers / 300000))
    
    estimated_car_value = st.number_input(
        "Estimated Car Value ($)",
        min_value=1000,
        max_value=200000,
        value=int(estimated_value),
        step=1000
    )

st.divider()

if st.button("Calculate Premium", type="primary", use_container_width=True):
    backend_url = get_backend_url()
    auth_token = get_auth_token()
    
    if not backend_url:
        st.error("Please enter the Backend Service URL in the sidebar")
    elif not auth_token:
        st.error("Please enter your PAT Token in the sidebar (or configure secrets.toml)")
    else:
        url = backend_url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"
        url = url.rstrip("/")
        
        with st.spinner("Calculating your premium..."):
            try:
                payload = {
                    "customer_id": customer_id if customer_id else None,
                    "car_age": car_age,
                    "kilometers": kilometers,
                    "engine_size": engine_size,
                    "estimated_car_value": estimated_car_value,
                    "age": age,
                    "years_licensed": years_licensed,
                    "car_make": car_make,
                    "car_model": car_model,
                    "fuel_type": fuel_type,
                    "transmission": transmission,
                    "coverage_type": coverage_type,
                    "gender": gender,
                    "state": state
                }
                
                headers = {
                    "Authorization": f'Snowflake Token="{auth_token}"',
                    "Content-Type": "application/json"
                }
                
                response = requests.post(
                    f"{url}/predict",
                    headers=headers,
                    json=payload,
                    timeout=180
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("Quote Generated Successfully!")
                    
                    col_result1, col_result2 = st.columns([2, 1])
                    
                    with col_result1:
                        st.metric(
                            label="Annual Premium",
                            value=f"${result['predicted_premium']:,.2f}"
                        )
                        st.caption(f"Monthly: ${result['predicted_premium']/12:,.2f}")
                    
                    with col_result2:
                        customer_type = result.get('customer_type', 'new')
                        if customer_type == 'existing':
                            st.info("Existing Customer - Profile data applied")
                        else:
                            st.info("New Customer - Default risk profile")
                    
                    with st.expander("View Details"):
                        st.json(result.get('features_used', {}))
                    
                    with st.expander("Performance Metrics"):
                        timing = result.get('timing_ms', {})
                        if timing:
                            col_t1, col_t2, col_t3 = st.columns(3)
                            with col_t1:
                                st.metric("Total Time", f"{timing.get('total_request_ms', 0):.0f}ms")
                            with col_t2:
                                st.metric("Model Inference", f"{timing.get('model_inference_ms', 0):.0f}ms")
                            with col_t3:
                                retries = timing.get('model_retries', 0)
                                st.metric("Retries", retries)
                            
                            st.markdown("**Detailed Timing:**")
                            st.json(timing)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend service. Please check the URL.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.divider()
st.caption("Powered by Snowflake ML | Feature Store | SPCS")
