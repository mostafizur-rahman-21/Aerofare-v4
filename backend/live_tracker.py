"""
================================================================================
✈️ AeroFare — Live Market Telemetry Module
1st Semester Computer Science & Engineering Project
Team K — Scholars Institute of Technology and Management
================================================================================
This module fetches live flight fares from Google Flights (via SerpApi)
and logs captured real-world market prices to a local CSV file for future
model retraining experiments.
================================================================================
"""

import os
import csv
import requests
from datetime import datetime

# API Key for SerpApi Google Flights engine (supports environment variable override)
SERPAPI_KEY = os.getenv(
    "SERPAPI_KEY",
    "0f16ee52dc6b646b76b0ce5a88094dd6bf1bc97bef6ce40649ac69adad3f0b80"
)

# Store live price logs outside the project folder so local Live Server never reloads unexpectedly
DATA_DIR = os.path.expanduser("~/aerofare_data")
os.makedirs(DATA_DIR, exist_ok=True)
LOG_FILE = os.path.join(DATA_DIR, "live_prices_history.csv")

# IATA airport codes for the 6 domestic metro cities
CITY_TO_IATA = {
    "Delhi": "DEL",
    "Mumbai": "BOM",
    "Bangalore": "BLR",
    "Kolkata": "CCU",
    "Hyderabad": "HYD",
    "Chennai": "MAA"
}


def get_live_flight_offers(
    source_city: str,
    destination_city: str,
    travel_date: str,
    flight_class: str = "Economy",
    stops_num: int = None
) -> list:
    """
    Fetches real-time flight offers from Google Flights using SerpApi.
    Returns a sorted list of flight dictionaries: [{airline, live_price, duration_mins, stops_num}, ...]
    """
    dep_code = CITY_TO_IATA.get(source_city)
    arr_code = CITY_TO_IATA.get(destination_city)

    # Validate city codes and API key presence
    if not dep_code or not arr_code or not SERPAPI_KEY or SERPAPI_KEY.startswith("PASTE_"):
        return []

    # SerpApi Google Flights travel class mapping: 1 = Economy, 3 = Business
    travel_class_code = 3 if flight_class == "Business" else 1

    params = {
        "engine": "google_flights",
        "departure_id": dep_code,
        "arrival_id": arr_code,
        "outbound_date": travel_date,
        "currency": "INR",
        "hl": "en",
        "type": "2",  # One-way flight
        "travel_class": travel_class_code,
        "api_key": SERPAPI_KEY
    }

    # Filter non-stop flights if stops_num == 0
    if stops_num == 0:
        params["stops"] = "1"

    try:
        response = requests.get("https://serpapi.com/search.json", params=params, timeout=12)
        data = response.json()

        # If SerpApi returned an error (e.g. quota exhausted or invalid query)
        if "error" in data:
            print("⚠️ [AeroFare Live Tracker] SerpApi notice:", data.get("error"))
            return []

        # Combine best flights and other flights sections
        flights = data.get("best_flights", []) + data.get("other_flights", [])
        live_results = []

        for item in flights:
            price = item.get("price")
            legs = item.get("flights", [])
            first_leg = legs[0] if legs else {}
            airline_name = first_leg.get("airline", "Unknown")
            duration = item.get("total_duration", 140)
            stops = min(max(0, len(legs) - 1), 2)

            # Match stops criteria if specified
            if price and (stops_num is None or stops == stops_num):
                live_results.append({
                    "airline": airline_name,
                    "live_price": float(price),
                    "duration_mins": int(duration),
                    "stops_num": stops
                })

        # Sort flights by lowest price first
        live_results.sort(key=lambda x: x["live_price"])
        return live_results

    except Exception as e:
        print("⚠️ [AeroFare Live Tracker] Live API fetch error:", e)
        return []


def log_live_price_for_retraining(row_dict: dict, live_price: float):
    """
    Appends fetched live flight prices to ~/aerofare_data/live_prices_history.csv.
    Enables future fine-tuning or retraining with real-world observed market data.
    """
    try:
        file_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "airline", "source_city", "destination_city",
                    "departure_time", "arrival_time", "class", "stops_num",
                    "duration_mins", "days_left", "price"
                ])
            writer.writerow([
                datetime.now().isoformat(),
                row_dict.get("airline", ""),
                row_dict.get("source_city", ""),
                row_dict.get("destination_city", ""),
                row_dict.get("departure_time", ""),
                row_dict.get("arrival_time", ""),
                row_dict.get("class", ""),
                row_dict.get("stops_num", 0),
                row_dict.get("duration_mins", 140),
                row_dict.get("days_left", 10),
                live_price
            ])
    except Exception as e:
        print("⚠️ [AeroFare Live Tracker] Could not log live price:", e)
