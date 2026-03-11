"""
Helper functions for generating synthetic car insurance data.

Extracted from data_ops.ingest_data_job() to keep the ML Job function concise.
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# -- Reference data --

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

BASE_PRICES = {
    "Toyota": 28000, "Honda": 27000, "Ford": 35000, "BMW": 55000,
    "Mercedes": 60000, "Chevrolet": 32000, "Tesla": 50000, "Nissan": 26000,
}

FUEL_TYPES = ["Gasoline", "Diesel", "Hybrid", "Electric"]
TRANSMISSIONS = ["Automatic", "Manual", "CVT"]
COVERAGE_TYPES = ["Basic", "Standard", "Premium", "Comprehensive"]

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer",
    "Michael", "Linda", "William", "Elizabeth", "David", "Susan",
    "Richard", "Jessica", "Joseph", "Sarah",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson",
    "Taylor", "Thomas", "Moore", "Jackson",
]
STATES = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"]


def generate_customers(n_customers: int) -> pd.DataFrame:
    """Generate synthetic customer data."""
    np.random.seed(42)
    random.seed(42)

    customers_data = []
    for i in range(1, n_customers + 1):
        age = max(18, min(80, int(np.random.normal(42, 15))))
        years_licensed = max(1, min(age - 16, int(np.random.exponential(15))))
        claims_history = min(int(np.random.exponential(0.8)), 10)
        credit_score = max(300, min(850, int(np.random.normal(700, 80))))
        customers_data.append({
            "CUSTOMER_ID": f"CUST_{str(i).zfill(6)}",
            "FIRST_NAME": random.choice(FIRST_NAMES),
            "LAST_NAME": random.choice(LAST_NAMES),
            "AGE": age, "GENDER": random.choice(["M", "F"]),
            "YEARS_LICENSED": years_licensed,
            "CLAIMS_HISTORY": claims_history,
            "CREDIT_SCORE": credit_score,
            "STATE": random.choice(STATES),
        })

    df = pd.DataFrame(customers_data)
    print(f"Generated {len(df)} customers")
    return df


def generate_policies(customers_df: pd.DataFrame, n_policies: int) -> pd.DataFrame:
    """Generate synthetic policy data linked to existing customers."""
    np.random.seed(42)
    random.seed(42)

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

        fuel_type = _pick_fuel_type(car_make)
        transmission = random.choices(TRANSMISSIONS, weights=[0.7, 0.15, 0.15])[0]
        kilometers = int(max(1000, car_age * np.random.normal(15000, 5000) + np.random.normal(0, 5000)))
        engine_size = random.choice([1.5, 1.8, 2.0, 2.4, 2.5, 3.0, 3.5, 4.0, 5.0])

        car_value = _estimate_car_value(car_make, car_age, kilometers)
        coverage_type = random.choice(COVERAGE_TYPES)
        deductible = random.choice([250, 500, 1000, 1500, 2000])
        annual_premium = _calculate_premium(customer, car_make, car_value, coverage_type, deductible)
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

    df = pd.DataFrame(policies_data)
    print(f"Generated {len(df)} policies")
    return df


# -- Private helpers --

def _pick_fuel_type(car_make: str) -> str:
    if car_make == "Tesla":
        return "Electric"
    elif car_make in ["BMW", "Mercedes"]:
        return random.choices(FUEL_TYPES, weights=[0.6, 0.15, 0.2, 0.05])[0]
    else:
        return random.choices(FUEL_TYPES, weights=[0.7, 0.1, 0.15, 0.05])[0]


def _estimate_car_value(car_make: str, car_age: int, kilometers: int) -> float:
    base_price = BASE_PRICES[car_make]
    depreciation = 0.85 ** car_age
    km_factor = max(0.5, 1 - (kilometers / 300000))
    return base_price * depreciation * km_factor


def _calculate_premium(customer, car_make: str, car_value: float,
                       coverage_type: str, deductible: int) -> float:
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
    return max(400, min(5000, annual_premium))
