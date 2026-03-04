"""
Data Operations for Car Insurance ML Pipeline.

Generates synthetic customer and policy data and uploads it to Snowflake.
This module is designed to run both locally and as part of a Snowflake Task.
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

from snowflake.ml.jobs import remote
from snowflake.snowpark import Session

from constants import (
    PIPELINE_DB,
    DATA_SCHEMA,
    CUSTOMERS_TABLE,
    POLICIES_TABLE,
    N_CUSTOMERS,
    N_POLICIES,
    COMPUTE_POOL,
    JOB_STAGE,
)


CAR_MAKES = {
    "Toyota": ["Camry", "Corolla", "RAV4", "Highlander", "Prius"],
    "Honda": ["Civic", "Accord", "CR-V", "Pilot", "Odyssey"],
    "Ford": ["F-150", "Mustang", "Explorer", "Escape", "Bronco"],
    "BMW": ["3 Series", "5 Series", "X3", "X5", "M3"],
    "Mercedes": ["C-Class", "E-Class", "GLC", "GLE", "S-Class"],
    "Chevrolet": ["Silverado", "Malibu", "Equinox", "Tahoe", "Corvette"],
    "Tesla": ["Model 3", "Model Y", "Model S", "Model X"],
    "Nissan": ["Altima", "Rogue", "Sentra", "Pathfinder", "Maxima"],
}

FUEL_TYPES = ["Gasoline", "Diesel", "Hybrid", "Electric"]
TRANSMISSIONS = ["Automatic", "Manual", "CVT"]
COVERAGE_TYPES = ["Basic", "Standard", "Premium", "Comprehensive"]

BASE_PRICES = {
    "Toyota": 28000,
    "Honda": 27000,
    "Ford": 35000,
    "BMW": 55000,
    "Mercedes": 60000,
    "Chevrolet": 32000,
    "Tesla": 50000,
    "Nissan": 26000,
}


def generate_customers(n_customers: int = N_CUSTOMERS, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic customer data."""
    np.random.seed(seed)
    random.seed(seed)

    first_names = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer",
        "Michael", "Linda", "William", "Elizabeth", "David", "Susan",
        "Richard", "Jessica", "Joseph", "Sarah",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
        "Miller", "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson",
        "Taylor", "Thomas", "Moore", "Jackson",
    ]
    states = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"]

    customers_data = []
    for i in range(1, n_customers + 1):
        age = max(18, min(80, int(np.random.normal(42, 15))))
        years_licensed = max(1, min(age - 16, int(np.random.exponential(15))))
        claims_history = min(int(np.random.exponential(0.8)), 10)
        credit_score = max(300, min(850, int(np.random.normal(700, 80))))

        customers_data.append({
            "CUSTOMER_ID": f"CUST_{str(i).zfill(6)}",
            "FIRST_NAME": random.choice(first_names),
            "LAST_NAME": random.choice(last_names),
            "AGE": age,
            "GENDER": random.choice(["M", "F"]),
            "YEARS_LICENSED": years_licensed,
            "CLAIMS_HISTORY": claims_history,
            "CREDIT_SCORE": credit_score,
            "STATE": random.choice(states),
        })

    return pd.DataFrame(customers_data)


def generate_policies(
    customers_df: pd.DataFrame,
    n_policies: int = N_POLICIES,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic policy data linked to customers."""
    np.random.seed(seed)
    random.seed(seed)

    current_year = datetime.now().year
    customer_ids = customers_df["CUSTOMER_ID"].tolist()

    policies_data = []
    for i in range(n_policies):
        cust_id = random.choice(customer_ids)
        customer = customers_df[customers_df["CUSTOMER_ID"] == cust_id].iloc[0]

        car_make = random.choice(list(CAR_MAKES.keys()))
        car_model = random.choice(CAR_MAKES[car_make])
        car_year = random.randint(2010, current_year)
        car_age = current_year - car_year

        if car_make == "Tesla":
            fuel_type = "Electric"
        elif car_make in ["BMW", "Mercedes"]:
            fuel_type = random.choices(FUEL_TYPES, weights=[0.6, 0.15, 0.2, 0.05])[0]
        else:
            fuel_type = random.choices(FUEL_TYPES, weights=[0.7, 0.1, 0.15, 0.05])[0]

        transmission = random.choices(TRANSMISSIONS, weights=[0.7, 0.15, 0.15])[0]
        kilometers = int(max(1000, car_age * np.random.normal(15000, 5000) + np.random.normal(0, 5000)))
        engine_size = random.choice([1.5, 1.8, 2.0, 2.4, 2.5, 3.0, 3.5, 4.0, 5.0])

        base_price = BASE_PRICES[car_make]
        depreciation = 0.85 ** car_age
        km_factor = max(0.5, 1 - (kilometers / 300000))
        car_value = base_price * depreciation * km_factor

        coverage_type = random.choice(COVERAGE_TYPES)
        deductible = random.choice([250, 500, 1000, 1500, 2000])

        # Premium calculation
        base_premium = 800
        age_factor = 1.5 if customer["AGE"] < 25 else (1.2 if customer["AGE"] > 65 else 1.0)
        experience_factor = max(0.8, 1.3 - (customer["YEARS_LICENSED"] * 0.02))
        claims_factor = 1 + (customer["CLAIMS_HISTORY"] * 0.15)
        credit_factor = max(0.8, 1.3 - ((customer["CREDIT_SCORE"] - 600) / 500))
        car_value_factor = 0.03 * (car_value / 10000)
        make_factor = {
            "BMW": 1.3, "Mercedes": 1.35, "Tesla": 1.25, "Chevrolet": 1.05,
            "Ford": 1.1, "Toyota": 0.95, "Honda": 0.95, "Nissan": 1.0,
        }.get(car_make, 1.0)
        coverage_factor = {"Basic": 0.7, "Standard": 1.0, "Premium": 1.3, "Comprehensive": 1.6}[coverage_type]
        deductible_factor = {250: 1.2, 500: 1.1, 1000: 1.0, 1500: 0.9, 2000: 0.85}[deductible]

        annual_premium = base_premium * age_factor * experience_factor * claims_factor * credit_factor
        annual_premium *= (1 + car_value_factor) * make_factor * coverage_factor * deductible_factor
        annual_premium += np.random.normal(0, 50)
        annual_premium = max(400, min(5000, annual_premium))

        start_date = datetime.now() - timedelta(days=random.randint(0, 365))

        policies_data.append({
            "POLICY_ID": f"POL_{str(i + 1).zfill(7)}",
            "CUSTOMER_ID": cust_id,
            "CAR_MAKE": car_make,
            "CAR_MODEL": car_model,
            "CAR_YEAR": car_year,
            "COLOR": random.choice(["Black", "White", "Silver", "Gray", "Blue", "Red", "Green", "Brown"]),
            "KILOMETERS": kilometers,
            "ENGINE_SIZE": engine_size,
            "FUEL_TYPE": fuel_type,
            "TRANSMISSION": transmission,
            "COVERAGE_TYPE": coverage_type,
            "DEDUCTIBLE": deductible,
            "ESTIMATED_CAR_VALUE": round(car_value, 2),
            "ANNUAL_PREMIUM": round(annual_premium, 2),
            "POLICY_START_DATE": start_date.strftime("%Y-%m-%d"),
            "UPDATED_AT": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    return pd.DataFrame(policies_data)


def ingest_data(session: Session) -> dict:
    """
    Generate synthetic data and upload to Snowflake.

    Returns a dict with table names and row counts.
    """
    session.use_database(PIPELINE_DB)
    session.use_schema(DATA_SCHEMA)

    print("Generating synthetic customer data...")
    customers_df = generate_customers()
    print(f"Generated {len(customers_df)} customers")

    print("Generating synthetic policy data...")
    policies_df = generate_policies(customers_df)
    print(f"Generated {len(policies_df)} policies")
    print(f"Premium range: ${policies_df['ANNUAL_PREMIUM'].min():.2f} - ${policies_df['ANNUAL_PREMIUM'].max():.2f}")

    print("Uploading to Snowflake...")
    session.write_pandas(customers_df, "CUSTOMERS", auto_create_table=True, overwrite=True)
    session.write_pandas(policies_df, "POLICIES", auto_create_table=True, overwrite=True)

    cust_count = session.table("CUSTOMERS").count()
    pol_count = session.table("POLICIES").count()
    print(f"Uploaded: CUSTOMERS={cust_count} rows, POLICIES={pol_count} rows")

    return {
        "customers_table": CUSTOMERS_TABLE,
        "policies_table": POLICIES_TABLE,
        "customers_count": cust_count,
        "policies_count": pol_count,
    }


# =============================================================================
# ML Job version: runs data ingestion on SPCS compute pool
# =============================================================================

@remote(COMPUTE_POOL, stage_name=JOB_STAGE)
def ingest_data_remote(session: Session, n_customers: int, n_policies: int) -> dict:
    """
    Generate synthetic data and upload to Snowflake, running on the compute pool.

    Self-contained: all imports and logic inside the function body so it
    serializes cleanly to the Container Runtime.
    """
    import numpy as np
    import pandas as pd
    import random
    from datetime import datetime, timedelta

    pipeline_db = "CC_INSURANCE_PIPELINE"
    data_schema = "DATA"

    session.use_database(pipeline_db)
    session.use_schema(data_schema)

    car_makes = {
        "Toyota": ["Camry", "Corolla", "RAV4", "Highlander", "Prius"],
        "Honda": ["Civic", "Accord", "CR-V", "Pilot", "Odyssey"],
        "Ford": ["F-150", "Mustang", "Explorer", "Escape", "Bronco"],
        "BMW": ["3 Series", "5 Series", "X3", "X5", "M3"],
        "Mercedes": ["C-Class", "E-Class", "GLC", "GLE", "S-Class"],
        "Chevrolet": ["Silverado", "Malibu", "Equinox", "Tahoe", "Corvette"],
        "Tesla": ["Model 3", "Model Y", "Model S", "Model X"],
        "Nissan": ["Altima", "Rogue", "Sentra", "Pathfinder", "Maxima"],
    }
    fuel_types = ["Gasoline", "Diesel", "Hybrid", "Electric"]
    transmissions = ["Automatic", "Manual", "CVT"]
    coverage_types = ["Basic", "Standard", "Premium", "Comprehensive"]
    base_prices = {
        "Toyota": 28000, "Honda": 27000, "Ford": 35000, "BMW": 55000,
        "Mercedes": 60000, "Chevrolet": 32000, "Tesla": 50000, "Nissan": 26000,
    }

    # Generate customers
    np.random.seed(42)
    random.seed(42)
    first_names = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer",
        "Michael", "Linda", "William", "Elizabeth", "David", "Susan",
        "Richard", "Jessica", "Joseph", "Sarah",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
        "Miller", "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson",
        "Taylor", "Thomas", "Moore", "Jackson",
    ]
    states = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"]

    customers_data = []
    for i in range(1, n_customers + 1):
        age = max(18, min(80, int(np.random.normal(42, 15))))
        years_licensed = max(1, min(age - 16, int(np.random.exponential(15))))
        claims_history = min(int(np.random.exponential(0.8)), 10)
        credit_score = max(300, min(850, int(np.random.normal(700, 80))))
        customers_data.append({
            "CUSTOMER_ID": f"CUST_{str(i).zfill(6)}",
            "FIRST_NAME": random.choice(first_names),
            "LAST_NAME": random.choice(last_names),
            "AGE": age, "GENDER": random.choice(["M", "F"]),
            "YEARS_LICENSED": years_licensed,
            "CLAIMS_HISTORY": claims_history,
            "CREDIT_SCORE": credit_score,
            "STATE": random.choice(states),
        })
    customers_df = pd.DataFrame(customers_data)
    print(f"Generated {len(customers_df)} customers")

    # Generate policies
    np.random.seed(42)
    random.seed(42)
    current_year = datetime.now().year
    customer_ids = customers_df["CUSTOMER_ID"].tolist()
    policies_data = []
    for i in range(n_policies):
        cust_id = random.choice(customer_ids)
        customer = customers_df[customers_df["CUSTOMER_ID"] == cust_id].iloc[0]
        car_make = random.choice(list(car_makes.keys()))
        car_model = random.choice(car_makes[car_make])
        car_year = random.randint(2010, current_year)
        car_age = current_year - car_year

        if car_make == "Tesla":
            fuel_type = "Electric"
        elif car_make in ["BMW", "Mercedes"]:
            fuel_type = random.choices(fuel_types, weights=[0.6, 0.15, 0.2, 0.05])[0]
        else:
            fuel_type = random.choices(fuel_types, weights=[0.7, 0.1, 0.15, 0.05])[0]

        transmission = random.choices(transmissions, weights=[0.7, 0.15, 0.15])[0]
        kilometers = int(max(1000, car_age * np.random.normal(15000, 5000) + np.random.normal(0, 5000)))
        engine_size = random.choice([1.5, 1.8, 2.0, 2.4, 2.5, 3.0, 3.5, 4.0, 5.0])
        base_price = base_prices[car_make]
        depreciation = 0.85 ** car_age
        km_factor = max(0.5, 1 - (kilometers / 300000))
        car_value = base_price * depreciation * km_factor
        coverage_type = random.choice(coverage_types)
        deductible = random.choice([250, 500, 1000, 1500, 2000])

        base_premium = 800
        age_factor = 1.5 if customer["AGE"] < 25 else (1.2 if customer["AGE"] > 65 else 1.0)
        experience_factor = max(0.8, 1.3 - (customer["YEARS_LICENSED"] * 0.02))
        claims_factor = 1 + (customer["CLAIMS_HISTORY"] * 0.15)
        credit_factor = max(0.8, 1.3 - ((customer["CREDIT_SCORE"] - 600) / 500))
        car_value_factor = 0.03 * (car_value / 10000)
        make_factor = {
            "BMW": 1.3, "Mercedes": 1.35, "Tesla": 1.25, "Chevrolet": 1.05,
            "Ford": 1.1, "Toyota": 0.95, "Honda": 0.95, "Nissan": 1.0,
        }.get(car_make, 1.0)
        coverage_factor = {"Basic": 0.7, "Standard": 1.0, "Premium": 1.3, "Comprehensive": 1.6}[coverage_type]
        deductible_factor = {250: 1.2, 500: 1.1, 1000: 1.0, 1500: 0.9, 2000: 0.85}[deductible]

        annual_premium = base_premium * age_factor * experience_factor * claims_factor * credit_factor
        annual_premium *= (1 + car_value_factor) * make_factor * coverage_factor * deductible_factor
        annual_premium += np.random.normal(0, 50)
        annual_premium = max(400, min(5000, annual_premium))
        start_date = datetime.now() - timedelta(days=random.randint(0, 365))

        policies_data.append({
            "POLICY_ID": f"POL_{str(i + 1).zfill(7)}",
            "CUSTOMER_ID": cust_id, "CAR_MAKE": car_make, "CAR_MODEL": car_model,
            "CAR_YEAR": car_year,
            "COLOR": random.choice(["Black", "White", "Silver", "Gray", "Blue", "Red", "Green", "Brown"]),
            "KILOMETERS": kilometers, "ENGINE_SIZE": engine_size,
            "FUEL_TYPE": fuel_type, "TRANSMISSION": transmission,
            "COVERAGE_TYPE": coverage_type, "DEDUCTIBLE": deductible,
            "ESTIMATED_CAR_VALUE": round(car_value, 2),
            "ANNUAL_PREMIUM": round(annual_premium, 2),
            "POLICY_START_DATE": start_date.strftime("%Y-%m-%d"),
            "UPDATED_AT": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    policies_df = pd.DataFrame(policies_data)
    print(f"Generated {len(policies_df)} policies")

    # Upload to Snowflake
    session.write_pandas(customers_df, "CUSTOMERS", auto_create_table=True, overwrite=True)
    session.write_pandas(policies_df, "POLICIES", auto_create_table=True, overwrite=True)

    cust_count = session.table("CUSTOMERS").count()
    pol_count = session.table("POLICIES").count()
    print(f"Uploaded: CUSTOMERS={cust_count} rows, POLICIES={pol_count} rows")

    return {
        "customers_table": f"{pipeline_db}.{data_schema}.CUSTOMERS",
        "policies_table": f"{pipeline_db}.{data_schema}.POLICIES",
        "customers_count": cust_count,
        "policies_count": pol_count,
    }


def submit_ingest_job(session: Session) -> dict:
    """
    Submit data ingestion as an ML Job on the compute pool and wait for result.

    Returns dict with table names and row counts.
    """
    print(f"Submitting ingest job to compute pool {COMPUTE_POOL}...")
    job = ingest_data_remote(session, N_CUSTOMERS, N_POLICIES)
    print(f"ML Job submitted: {job.id}")

    result = job.result()
    print(f"Ingest complete: {result['customers_count']} customers, {result['policies_count']} policies")
    return result
