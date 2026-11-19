# ✈️ AeroFare — AI Flight Fare Intelligence & Telemetry

An end-to-end Machine Learning, Real-Time Market Telemetry, and Travel Resilience web application developed for 1st-semester Computer Science & Engineering students. It estimates Indian domestic flight ticket fares from historical booking data, compares them against live Google Flights market quotes, detects price surges, and provides an automated 3-tier backup engine when flights are unavailable or overpriced.

> **Academic Prototype Notice:** AeroFare's ML baseline model is trained on 300,000+ historical Indian domestic flight records (`Clean_Dataset.csv`). When live market telemetry is queried, the system evaluates real-time market quotes against the historical ML fair baseline and suggests flexible dates, cheaper alternative carriers, and Indian Railways express backups.

---

## 👥 Team K — Scholars Institute of Technology and Management

* **Project Window:** 16–20 November 2026 *(Development: 16–18 Nov | Presentations: 19–20 Nov)*
* **Mostafizur Rahman** *(Team Leader)* — Data Preprocessing, Random Forest ML Model, FastAPI Backend, Smart Backup Engine, UI/UX Architecture & Deployment
* **Mahammad Ashif** — Frontend UI/UX Architecture & Chart.js Visualizations
* **Mountain Brahma** — Frontend Styling, Responsive Layout & Interactive Controls
* **Ruhan e Farhad** — End-to-End Sample Testing & Quality Assurance

---

## ✨ Key Features

### 1. Core Machine Learning & REST API Engine
* **Historical ML Fare Estimator (`POST /predict`):** Predicts fair flight ticket fares in Indian Rupees (`₹`) using a trained `scikit-learn` **Random Forest Regressor** (`aerofare_model.joblib`) across 29 one-hot encoded features (`model_columns.joblib`), achieving a holdout test Mean Absolute Error (MAE) of **~₹1,084.95**.
* **Automated Lead-Time Calculation:** Dynamically calculates `days_left` from the user's selected `travel_date` (`YYYY-MM-DD`).
* **Vectorized Batch Predictions:** Fast matrix-based predictions for multi-date and multi-airline comparisons (~50ms response time).

### 2. 🛡️ Smart Backup & Travel Resilience Engine (Core Requirement)
Activated automatically when:
* **No flights are operating on the selected date** (sold-out or carrier not operating).
* **Live market fare is significantly higher than the analysed fair baseline** (Price Surge Alert).

It provides 3 interactive backup alternatives:
1. **📅 Flexible Travel Dates (5-Day Strip):** Evaluates ±3 adjacent days using ML to find cheaper travel days (e.g. *"Fly on Tue, 24 Nov to save ₹1,350"*). 1-click **"Select Date"** updates the search immediately.
2. **✈️ Alternative Domestic Airlines:** Scans other carriers (IndiGo, Air India, Vistara, SpiceJet, AirAsia) operating on the same route with lower fares. 1-click **"Switch Carrier"** re-runs the analysis.
3. **🚆 Indian Railways Multi-Modal Backup:** Integrates confirmed high-speed express train alternatives (e.g., *Mumbai Rajdhani Express* / *Vande Bharat Express*) with realistic durations (12–15h), fixed 3-Tier AC tariffs (~₹2,450), zero dynamic surge pricing, and net savings calculations.

### 3. Interactive Frontend & Visual Analytics
* **Visual Deal & Price Surge Assessment Gauge:** Instant horizontal status meter classifying fares into:
  * 🟢 **Great Deal** (Live fare < Fair price by 5%+)
  * 🔵 **Fair Price** (Live fare matches historical baseline within ±8%)
  * 🟠 **Surge Alert** (Live fare is 10% to 25% above baseline)
  * 🔴 **Critical Surge** (Live fare is >25% above baseline)
* **Interactive Analytics (Chart.js):**
  * **Live Market vs. Historical ML** benchmark bar chart.
  * **All-Airline Comparison** bar chart across 6 domestic carriers (`AirAsia`, `SpiceJet`, `IndiGo`, `GO_FIRST`, `Air_India`, `Vistara`).
  * **45-Day Booking Window Trend** interactive line/bar chart displaying optimal advance booking windows.
* **🎓 In-App Student Viva & Project Guide Modal:** Click **"Viva Guide"** in the navigation header to view problem statement, ML architecture details, model formulas, and common viva questions with answers.
* **Academic Presentation Fallback Mode:** Seamless client-side simulation fallback if the FastAPI server is not started, ensuring live college presentations never crash or fail.

---

## 🛠️ Tech Stack

* **Machine Learning:** Python 3, `scikit-learn` (`RandomForestRegressor`), `pandas`, `numpy`, `joblib`
* **Backend & Telemetry:** FastAPI, Uvicorn, Pydantic v2, `requests` (`app.py`, `live_tracker.py`)
* **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript (ES6+), Chart.js
* **Dataset:** [Kaggle — Flight Price Prediction Dataset](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) (`Clean_Dataset.csv`, 300,000+ rows)

---

## 📂 Repository Structure

```text
AeroFare/
├── backend/
│   ├── app.py                  # FastAPI server (/predict & /analyze) with Smart Backup Engine
│   ├── live_tracker.py         # Google Flights telemetry & price logging module
│   ├── aerofare_model.joblib   # Trained Random Forest Regressor model
│   ├── model_columns.joblib    # 29 one-hot encoded training feature columns
│   └── requirements.txt        # Clean Python dependencies
├── ml/
│   └── aerofare_model.ipynb    # Data exploration, preprocessing, model training & MAE evaluation
├── frontend/
│   └── index.html              # Interactive web dashboard with backup engine & viva guide
├── docs/
│   └── PRD_Group_K_AeroFare.docx # Project Requirements Document
├── .gitignore
└── README.md
```

---

## 🚀 How to Run the Project (Beginner-Friendly)

### Step 1: Start the Backend REST API
Open a terminal in the `backend/` directory:
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
*The FastAPI server will start at `http://127.0.0.1:8000`. You can test it in your browser at `http://127.0.0.1:8000/docs` to see interactive Swagger API documentation.*

### Step 2: Open the Frontend Dashboard
Simply open `frontend/index.html` in your web browser:
* Double-click `frontend/index.html`, OR
* Use VS Code **"Open with Live Server"**.

*The dashboard automatically connects to `http://127.0.0.1:8000`. If the backend is offline, it gracefully runs in built-in **Academic Presentation Demo Mode**.*
