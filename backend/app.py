"""
================================================================================
✈️ AeroFare — Hybrid Flight Price Intelligence API
1st Semester Computer Science & Engineering Project
Team K — Scholars Institute of Technology and Management
================================================================================
This FastAPI service provides two main functions:
1. Machine Learning Price Estimator: Predicts historical fair flight fares using
   a trained scikit-learn Random Forest model.
2. Hybrid Live Telemetry & Backup Engine: Compares live market quotes (Google Flights)
   against the ML baseline, detects price surges or sold-out dates, and provides
   intelligent backup options (alternative dates, cheaper carriers, and train options).
================================================================================
"""

import os
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Import live market tracker functions
from live_tracker import get_live_flight_offers, log_live_price_for_retraining

# Initialize FastAPI application
app = FastAPI(
    title="AeroFare Hybrid ML + Live Price API",
    description="Flight fare prediction, live market comparison, and smart backup engine.",
    version="2.0.0"
)

# Enable CORS so frontend (HTML/JS) can call this API from any local port or browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# 1. LOAD MACHINE LEARNING MODEL & FEATURE COLUMNS
# ------------------------------------------------------------------------------
MODEL_PATH = "aerofare_model.joblib"
COLUMNS_PATH = "model_columns.joblib"

print("✈️ [AeroFare] Loading trained Random Forest model and column definitions...")
model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)
print(f"✅ [AeroFare] Model loaded successfully! (Trained on {len(model_columns)} encoded features)")

# All standard airlines in the Indian domestic dataset
ALL_AIRLINES = ["Air_India", "AirAsia", "GO_FIRST", "IndiGo", "SpiceJet", "Vistara"]


# ------------------------------------------------------------------------------
# 2. REQUEST DATA SCHEMA
# ------------------------------------------------------------------------------
class FlightSearch(BaseModel):
    airline: str = "IndiGo"
    source_city: str = "Delhi"
    destination_city: str = "Mumbai"
    departure_time: str = "Morning"
    arrival_time: str = "Evening"
    flight_class: str = Field(default="Economy", alias="class")
    stops_num: int = 0
    duration_mins: int = 140
    travel_date: Optional[str] = None
    days_left: Optional[int] = None

    class Config:
        populate_by_name = True


# ------------------------------------------------------------------------------
# 3. HELPER FUNCTIONS FOR FEATURE PREPARATION & PREDICTION
# ------------------------------------------------------------------------------
def clean_airline_name(raw_name: str) -> str:
    """
    Normalizes airline input strings to match the exact training one-hot column names.
    Note: In model_columns, 'airline_Indigo' uses title-case 'Indigo'.
    """
    clean = raw_name.replace(" ", "").replace("-", "").replace("_", "").lower()
    if "indigo" in clean:
        return "Indigo"
    if "airindia" in clean:
        return "Air_India"
    if "gofirst" in clean:
        return "GO_FIRST"
    if "spicejet" in clean:
        return "SpiceJet"
    if "vistara" in clean:
        return "Vistara"
    if "airasia" in clean:
        return "AirAsia"
    return raw_name


def predict_ml_fares_batch(rows: List[dict]) -> List[float]:
    """
    Fast, vectorized prediction: encodes multiple flight scenarios in a single batch.
    Takes ~50 milliseconds for 20+ rows instead of multiple seconds.
    """
    if not rows:
        return []
    
    cleaned_rows = []
    for r in rows:
        row_copy = r.copy()
        row_copy["airline"] = clean_airline_name(row_copy.get("airline", ""))
        cleaned_rows.append(row_copy)

    df = pd.DataFrame(cleaned_rows)
    df_encoded = pd.get_dummies(df).reindex(columns=model_columns, fill_value=0)
    predictions = model.predict(df_encoded)
    return [round(float(p), 2) for p in predictions]


def predict_ml_fare(row_dict: dict) -> float:
    """Helper to predict a single flight scenario."""
    return predict_ml_fares_batch([row_dict])[0]


def compute_days_left(search: FlightSearch) -> int:
    """Calculates lead-time (days remaining) from the selected travel date."""
    if search.travel_date:
        try:
            travel_dt = datetime.strptime(search.travel_date, "%Y-%m-%d").date()
            return max(1, (travel_dt - date.today()).days)
        except Exception:
            pass
    if search.days_left is not None:
        return max(1, search.days_left)
    return 10


def build_analytics_arrays(base_input: dict):
    """
    Builds analytics data arrays for Chart.js:
    1. Cross-airline price comparison (all 6 domestic carriers).
    2. 45-day booking window trend (advance booking curves).
    """
    # 1. Airline Comparison
    airline_rows = []
    for airline_name in ALL_AIRLINES:
        temp = base_input.copy()
        temp["airline"] = airline_name
        airline_rows.append(temp)
    
    airline_prices = predict_ml_fares_batch(airline_rows)
    airline_comparison = [
        {"airline": ALL_AIRLINES[i].replace("_", " "), "price": airline_prices[i]}
        for i in range(len(ALL_AIRLINES))
    ]
    airline_comparison.sort(key=lambda x: x["price"])

    # 2. Advance Booking Window Trend (Days: 1, 3, 7, 14, 21, 30, 45)
    trend_days = [1, 3, 7, 14, 21, 30, 45]
    trend_rows = []
    for d in trend_days:
        temp = base_input.copy()
        temp["days_left"] = d
        trend_rows.append(temp)
    
    trend_prices = predict_ml_fares_batch(trend_rows)
    booking_trend = [
        {"days_left": trend_days[i], "price": trend_prices[i]}
        for i in range(len(trend_days))
    ]

    return airline_comparison, booking_trend


# ------------------------------------------------------------------------------
# 4. 🚆 INDIAN RAILWAYS BACKUP DATABASE
# Realistic popular train connections between the 6 domestic metro cities
# ------------------------------------------------------------------------------
TRAIN_BACKUP_DATABASE = {
    ("Delhi", "Mumbai"): {"train_name": "Mumbai Rajdhani Express (12952)", "travel_time": "15h 30m", "ac_fare": 2450, "class_name": "3-Tier AC"},
    ("Mumbai", "Delhi"): {"train_name": "August Kranti Rajdhani (12953)", "travel_time": "15h 45m", "ac_fare": 2450, "class_name": "3-Tier AC"},
    ("Delhi", "Bangalore"): {"train_name": "Karnataka Express (12628)", "travel_time": "33h 40m", "ac_fare": 2850, "class_name": "3-Tier AC"},
    ("Bangalore", "Delhi"): {"train_name": "Bengaluru Rajdhani (22691)", "travel_time": "33h 15m", "ac_fare": 2850, "class_name": "3-Tier AC"},
    ("Delhi", "Kolkata"): {"train_name": "Howrah Rajdhani Express (12302)", "travel_time": "17h 15m", "ac_fare": 2380, "class_name": "3-Tier AC"},
    ("Kolkata", "Delhi"): {"train_name": "Poorva Express (12303)", "travel_time": "18h 00m", "ac_fare": 2200, "class_name": "3-Tier AC"},
    ("Delhi", "Hyderabad"): {"train_name": "Telangana Express (12724)", "travel_time": "23h 50m", "ac_fare": 2150, "class_name": "3-Tier AC"},
    ("Hyderabad", "Delhi"): {"train_name": "AP Express (20805)", "travel_time": "24h 10m", "ac_fare": 2150, "class_name": "3-Tier AC"},
    ("Delhi", "Chennai"): {"train_name": "Tamil Nadu Express (12622)", "travel_time": "32h 10m", "ac_fare": 2650, "class_name": "3-Tier AC"},
    ("Chennai", "Delhi"): {"train_name": "Grand Trunk Express (12615)", "travel_time": "33h 20m", "ac_fare": 2650, "class_name": "3-Tier AC"},
    ("Mumbai", "Bangalore"): {"train_name": "Udyan Express (11301)", "travel_time": "21h 30m", "ac_fare": 1680, "class_name": "3-Tier AC"},
    ("Bangalore", "Mumbai"): {"train_name": "CSMT Bengaluru Express (11014)", "travel_time": "21h 15m", "ac_fare": 1680, "class_name": "3-Tier AC"},
    ("Mumbai", "Kolkata"): {"train_name": "Howrah Duronto Express (12261)", "travel_time": "26h 40m", "ac_fare": 2450, "class_name": "3-Tier AC"},
    ("Kolkata", "Mumbai"): {"train_name": "Gitanjali Express (12860)", "travel_time": "29h 30m", "ac_fare": 2300, "class_name": "3-Tier AC"},
    ("Mumbai", "Hyderabad"): {"train_name": "Vande Bharat Express (20706)", "travel_time": "12h 45m", "ac_fare": 1450, "class_name": "AC Chair Car"},
    ("Hyderabad", "Mumbai"): {"train_name": "Hussainsagar Express (12702)", "travel_time": "13h 20m", "ac_fare": 1380, "class_name": "3-Tier AC"},
    ("Mumbai", "Chennai"): {"train_name": "Chennai Mail (11027)", "travel_time": "22h 15m", "ac_fare": 1950, "class_name": "3-Tier AC"},
    ("Chennai", "Mumbai"): {"train_name": "Mumbai Express (22158)", "travel_time": "22h 30m", "ac_fare": 1950, "class_name": "3-Tier AC"},
    ("Bangalore", "Chennai"): {"train_name": "Vande Bharat Express (20608)", "travel_time": "4h 25m", "ac_fare": 980, "class_name": "AC Chair Car"},
    ("Chennai", "Bangalore"): {"train_name": "Shatabdi Express (12027)", "travel_time": "4h 45m", "ac_fare": 1050, "class_name": "AC Chair Car"},
    ("Bangalore", "Hyderabad"): {"train_name": "Vande Bharat Express (20704)", "travel_time": "8h 30m", "ac_fare": 1540, "class_name": "AC Chair Car"},
    ("Hyderabad", "Bangalore"): {"train_name": "Kacheguda Express (12785)", "travel_time": "10h 30m", "ac_fare": 1350, "class_name": "3-Tier AC"},
    ("Bangalore", "Kolkata"): {"train_name": "Howrah SF Express (12864)", "travel_time": "31h 00m", "ac_fare": 2480, "class_name": "3-Tier AC"},
    ("Kolkata", "Bangalore"): {"train_name": "Anga Express (12254)", "travel_time": "31h 30m", "ac_fare": 2480, "class_name": "3-Tier AC"},
    ("Kolkata", "Chennai"): {"train_name": "Coromandel Express (12841)", "travel_time": "25h 30m", "ac_fare": 2180, "class_name": "3-Tier AC"},
    ("Chennai", "Kolkata"): {"train_name": "Howrah Mail (12840)", "travel_time": "26h 00m", "ac_fare": 2180, "class_name": "3-Tier AC"},
    ("Kolkata", "Hyderabad"): {"train_name": "Falaknuma Express (12703)", "travel_time": "25h 15m", "ac_fare": 2050, "class_name": "3-Tier AC"},
    ("Hyderabad", "Kolkata"): {"train_name": "East Coast Express (18046)", "travel_time": "26h 30m", "ac_fare": 1980, "class_name": "3-Tier AC"},
    ("Chennai", "Hyderabad"): {"train_name": "Vande Bharat Express (20677)", "travel_time": "8h 30m", "ac_fare": 1420, "class_name": "AC Chair Car"},
    ("Hyderabad", "Chennai"): {"train_name": "Charminar Express (12760)", "travel_time": "13h 05m", "ac_fare": 1280, "class_name": "3-Tier AC"},
}


def get_train_backup(source: str, dest: str) -> dict:
    """Returns train backup route for the given city pair."""
    pair = (source, dest)
    if pair in TRAIN_BACKUP_DATABASE:
        return TRAIN_BACKUP_DATABASE[pair]
    # Default fallback
    return {
        "train_name": f"Indian Railways Inter-City Superfast ({source} to {dest})",
        "travel_time": "18h 00m",
        "ac_fare": 1950,
        "class_name": "3-Tier AC"
    }


# ------------------------------------------------------------------------------
# 5. SMART BACKUP OPTIONS GENERATOR (CORE REQUIREMENT)
# ------------------------------------------------------------------------------
def generate_backup_options(
    base_input: dict,
    travel_date_str: Optional[str],
    current_ml_price: float,
    current_live_price: Optional[float],
    selected_airline: str,
    all_live_offers: list,
    airline_comparison: list
) -> dict:
    """
    Constructs a comprehensive backup plan if:
    a) No flights are operating on the selected date.
    b) Live price is higher than the ML analysed historical price (Surge).
    """
    source_city = base_input["source_city"]
    dest_city = base_input["destination_city"]

    # 1. Alternative Travel Dates Window (-2, -1, +1, +2, +3 days)
    alternative_dates = []
    best_date_option = None

    if travel_date_str:
        try:
            target_dt = datetime.strptime(travel_date_str, "%Y-%m-%d").date()
            today = date.today()
            offsets = [-2, -1, 1, 2, 3]

            alt_rows = []
            alt_meta = []
            for offset in offsets:
                cand_dt = target_dt + timedelta(days=offset)
                if cand_dt > today:
                    days_ahead = (cand_dt - today).days
                    temp_row = base_input.copy()
                    temp_row["days_left"] = days_ahead
                    alt_rows.append(temp_row)
                    alt_meta.append({
                        "date": cand_dt.strftime("%Y-%m-%d"),
                        "day_name": cand_dt.strftime("%a, %d %b"),
                        "days_left": days_ahead,
                        "offset": offset
                    })

            if alt_rows:
                alt_prices = predict_ml_fares_batch(alt_rows)
                for meta, pr in zip(alt_meta, alt_prices):
                    diff = round(pr - current_ml_price, 2)
                    savings = round(-diff, 2) if diff < 0 else 0
                    entry = {
                        "date": meta["date"],
                        "day_name": meta["day_name"],
                        "days_left": meta["days_left"],
                        "estimated_price": pr,
                        "diff_amount": diff,
                        "savings": savings,
                        "is_cheaper": diff < -100
                    }
                    alternative_dates.append(entry)

                # Identify date with highest savings
                cheapest_date_entry = min(alternative_dates, key=lambda x: x["estimated_price"])
                if cheapest_date_entry["is_cheaper"]:
                    best_date_option = cheapest_date_entry
        except Exception as e:
            print("Error computing alternative dates:", e)

    # 2. Cheaper Domestic Airline Options
    cheaper_airlines = []
    effective_current_price = current_live_price if current_live_price else current_ml_price

    # Check live offers first
    if all_live_offers:
        seen_airlines = set()
        clean_selected = clean_airline_name(selected_airline).lower()
        for fl in all_live_offers:
            air = fl.get("airline", "Unknown")
            air_clean = clean_airline_name(air).lower()
            if air_clean != clean_selected and air not in seen_airlines:
                live_p = fl.get("live_price", 0)
                if live_p < effective_current_price:
                    savings = round(effective_current_price - live_p, 2)
                    cheaper_airlines.append({
                        "airline": air,
                        "price": live_p,
                        "savings": savings,
                        "source": "Live Market Quote",
                        "stops_num": fl.get("stops_num", 0)
                    })
                    seen_airlines.add(air)
                    if len(cheaper_airlines) >= 3:
                        break

    # If no live cheaper airlines found, fallback to ML airline comparison
    if not cheaper_airlines and airline_comparison:
        for item in airline_comparison:
            air = item["airline"]
            if clean_airline_name(air).lower() != clean_airline_name(selected_airline).lower():
                pr = item["price"]
                if pr < effective_current_price:
                    cheaper_airlines.append({
                        "airline": air,
                        "price": pr,
                        "savings": round(effective_current_price - pr, 2),
                        "source": "Historical ML Analysed",
                        "stops_num": base_input.get("stops_num", 0)
                    })
                    if len(cheaper_airlines) >= 3:
                        break

    # 3. Indian Railways Multi-Modal Backup Option
    train_info = get_train_backup(source_city, dest_city)
    rail_fare = train_info["ac_fare"]
    rail_savings = max(0, round(effective_current_price - rail_fare, 2))
    train_backup = {
        "train_name": train_info["train_name"],
        "travel_time": train_info["travel_time"],
        "class_name": train_info["class_name"],
        "ac_fare": rail_fare,
        "savings_vs_flight": rail_savings,
        "note": "Fixed tariff • No dynamic surge pricing • Luggage allowance included"
    }

    # 4. Actionable Travel Recommendations
    recommendations = []
    if current_live_price is None:
        recommendations.append(f"No direct live flight found for {selected_airline.replace('_', ' ')} on selected date.")
        if cheaper_airlines:
            top_alt = cheaper_airlines[0]
            recommendations.append(f"Switch to {top_alt['airline']} (from ₹{top_alt['price']:,.0f}) to travel on your planned day.")
        if best_date_option:
            recommendations.append(f"Shift your departure to {best_date_option['day_name']} to save up to ₹{best_date_option['savings']:,.0f}.")
        recommendations.append(f"Take {train_info['train_name']} ({train_info['travel_time']}) in {train_info['class_name']} for ₹{rail_fare:,.0f}.")
    else:
        diff_amount = round(current_live_price - current_ml_price, 2)
        diff_percent = round((diff_amount / max(1.0, current_ml_price)) * 100, 1)

        if diff_percent >= 10:
            recommendations.append(f"Live fare is ₹{diff_amount:,.0f} ({diff_percent}%) above the fair historical analysed price.")
            if best_date_option:
                recommendations.append(f"Flexible date tip: Fly on {best_date_option['day_name']} to save ₹{best_date_option['savings']:,.0f}!")
            if cheaper_airlines:
                top_alt = cheaper_airlines[0]
                recommendations.append(f"Switch carrier to {top_alt['airline']} to save ₹{top_alt['savings']:,.0f} on the same day.")
            if rail_savings > 2000:
                recommendations.append(f"Budget saver: {train_info['train_name']} costs only ₹{rail_fare:,.0f} (Save ₹{rail_savings:,.0f} vs flight).")
        else:
            recommendations.append(f"Current live fare matches or beats historical fair price (₹{current_ml_price:,.0f}). Safe to book now!")

    # Deal Status Classification
    if current_live_price is None:
        deal_status = "UNAVAILABLE"
        deal_label = "Date / Carrier Unavailable"
    else:
        diff_p = (current_live_price - current_ml_price) / max(1.0, current_ml_price) * 100
        if diff_p <= -8:
            deal_status = "GREAT_DEAL"
            deal_label = f"Great Deal! ({abs(diff_p):.1f}% below fair price)"
        elif diff_p <= 8:
            deal_status = "FAIR_PRICE"
            deal_label = f"Fair Price (Within ±{abs(diff_p):.1f}% of fair price)"
        elif diff_p <= 25:
            deal_status = "SURGE_ALERT"
            deal_label = f"Price Surge (+{diff_p:.1f}% above fair price)"
        else:
            deal_status = "CRITICAL_SURGE"
            deal_label = f"Heavy Price Surge (+{diff_p:.1f}% above fair price)"

    return {
        "deal_status": deal_status,
        "deal_label": deal_label,
        "alternative_dates": alternative_dates,
        "best_date_option": best_date_option,
        "cheaper_airlines": cheaper_airlines,
        "train_backup": train_backup,
        "smart_recommendations": recommendations
    }


# ------------------------------------------------------------------------------
# 6. API ENDPOINTS
# ------------------------------------------------------------------------------
@app.get("/")
def health_check():
    """Health check endpoint to verify backend service status."""
    return {
        "status": "AeroFare ML & Telemetry API Online",
        "version": "2.0.0",
        "team": "Team K (Scholars Institute of Technology and Management)",
        "endpoints": ["/predict", "/analyze"]
    }


@app.post("/predict")
@app.post("/predict/")
def predict_only(search: FlightSearch):
    """
    Standard Mode: ML Fare Estimator
    Predicts the historical fair ticket price using our trained Random Forest model.
    Also provides alternative dates, carrier comparisons, and train backup.
    """
    days_left = compute_days_left(search)
    display_airline = search.airline.replace("_", " ")

    base_input = {
        "airline": search.airline,
        "source_city": search.source_city,
        "destination_city": search.destination_city,
        "departure_time": search.departure_time,
        "arrival_time": search.arrival_time,
        "class": search.flight_class,
        "stops_num": search.stops_num,
        "duration_mins": search.duration_mins,
        "days_left": days_left,
    }

    ml_predicted_price = predict_ml_fare(base_input)
    airline_comparison, booking_trend = build_analytics_arrays(base_input)

    backup_options = generate_backup_options(
        base_input=base_input,
        travel_date_str=search.travel_date,
        current_ml_price=ml_predicted_price,
        current_live_price=None,
        selected_airline=search.airline,
        all_live_offers=[],
        airline_comparison=airline_comparison
    )

    return {
        "days_left": days_left,
        "selected_airline": display_airline,
        "predicted_price": ml_predicted_price,
        "ml_predicted_price": ml_predicted_price,
        "live_lowest_price": None,
        "market_cheapest_offer": None,
        "price_difference_inr": 0.0,
        "price_difference_percent": 0.0,
        "ai_verdict": f"Historical fair fare for {display_airline} is ₹{ml_predicted_price:,.0f}.",
        "top_5_live_flights": [],
        "airline_comparison": airline_comparison,
        "booking_trend": booking_trend,
        "backup_options": backup_options
    }


@app.post("/analyze")
@app.post("/analyze/")
def analyze_live_vs_ml(search: FlightSearch):
    """
    Hybrid Mode: ML Fare + Real-Time Market Telemetry
    Compares live Google Flights market prices against historical ML baseline,
    detects price surges, and outputs smart backup options when unavailable or expensive.
    """
    days_left = compute_days_left(search)
    display_airline = search.airline.replace("_", " ")
    target_clean = clean_airline_name(search.airline).lower()

    base_input = {
        "airline": search.airline,
        "source_city": search.source_city,
        "destination_city": search.destination_city,
        "departure_time": search.departure_time,
        "arrival_time": search.arrival_time,
        "class": search.flight_class,
        "stops_num": search.stops_num,
        "duration_mins": search.duration_mins,
        "days_left": days_left,
    }

    # 1. Historical ML Analysed Price
    ml_predicted_price = predict_ml_fare(base_input)
    airline_comparison, booking_trend = build_analytics_arrays(base_input)

    # 2. Fetch live flight offers for route & date
    all_live_offers = []
    if search.travel_date:
        all_live_offers = get_live_flight_offers(
            search.source_city,
            search.destination_city,
            search.travel_date,
            search.flight_class,
            stops_num=search.stops_num
        )

    # Filter specifically for selected airline
    selected_airline_offers = [
        offer for offer in all_live_offers
        if target_clean in clean_airline_name(offer.get("airline", "")).lower()
        or clean_airline_name(offer.get("airline", "")).lower() in target_clean
    ]

    market_cheapest = all_live_offers[0] if all_live_offers else None

    # Case A: Selected airline has NO live flights on this date
    if not selected_airline_offers:
        backup_options = generate_backup_options(
            base_input=base_input,
            travel_date_str=search.travel_date,
            current_ml_price=ml_predicted_price,
            current_live_price=None,
            selected_airline=search.airline,
            all_live_offers=all_live_offers,
            airline_comparison=airline_comparison
        )

        alt_note = (
            f" Cheapest available carrier on this date is {market_cheapest['airline']} at ₹{market_cheapest['live_price']:,.0f}."
            if market_cheapest else " Check our Smart Backup Options below."
        )

        return {
            "days_left": days_left,
            "selected_airline": display_airline,
            "predicted_price": ml_predicted_price,
            "ml_predicted_price": ml_predicted_price,
            "live_lowest_price": None,
            "market_cheapest_offer": market_cheapest,
            "price_difference_inr": None,
            "price_difference_percent": None,
            "ai_verdict": f"No live {display_airline} flights available on this date.{alt_note}",
            "top_5_live_flights": all_live_offers[:5],
            "airline_comparison": airline_comparison,
            "booking_trend": booking_trend,
            "backup_options": backup_options
        }

    # Case B: Compare selected airline's live fare against historical ML fare
    lowest_selected_live = selected_airline_offers[0]["live_price"]
    log_live_price_for_retraining(base_input, lowest_selected_live)

    diff_amount = round(lowest_selected_live - ml_predicted_price, 2)
    diff_percent = round((diff_amount / max(1.0, ml_predicted_price)) * 100, 1)

    if diff_percent <= -8:
        verdict = f"GREAT DEAL on {display_airline}! Live fare is {abs(diff_percent)}% (₹{abs(diff_amount):,.0f}) cheaper than historical fair price. Book now!"
    elif diff_percent >= 10:
        verdict = f"SURGE ALERT on {display_airline}! Live fare is {diff_percent}% (₹{diff_amount:,.0f}) higher than fair price. Explore our Backup Options below!"
    else:
        verdict = f"FAIR PRICE on {display_airline}! Live fare closely matches historical fair price ({diff_percent:+}%)."

    # Order flight offers: selected airline at top, followed by competitors
    other_offers = [o for o in all_live_offers if o not in selected_airline_offers]
    ordered_offers = (selected_airline_offers + other_offers)[:6]

    backup_options = generate_backup_options(
        base_input=base_input,
        travel_date_str=search.travel_date,
        current_ml_price=ml_predicted_price,
        current_live_price=lowest_selected_live,
        selected_airline=search.airline,
        all_live_offers=all_live_offers,
        airline_comparison=airline_comparison
    )

    return {
        "days_left": days_left,
        "selected_airline": display_airline,
        "predicted_price": ml_predicted_price,
        "ml_predicted_price": ml_predicted_price,
        "live_lowest_price": lowest_selected_live,
        "market_cheapest_offer": market_cheapest,
        "price_difference_inr": diff_amount,
        "price_difference_percent": diff_percent,
        "ai_verdict": verdict,
        "top_5_live_flights": ordered_offers,
        "airline_comparison": airline_comparison,
        "booking_trend": booking_trend,
        "backup_options": backup_options
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
