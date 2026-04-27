
"""
HOW TO RUN:
    Step 1: python train_model.py (trains and saves the model)
    Step 2: python app.py (starts the web application)
    Step 3: Open http://127.0.0.1:5000 in your browser
"""

from flask import Flask, render_template, request, redirect, url_for, session
import pickle       #load saved ML model
import numpy as np  #Data handling
import os  #file handling

app = Flask(__name__)  #create flash application
app.secret_key = "heartcare_secret_2025" # needed for session storage

# ─────────────────────────────────────────────
# LOAD MODEL & SCALER (once at startup)
# ─────────────────────────────────────────────
MODEL_PATH = "model/model.pkl"
SCALER_PATH = "model/scaler.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Please run: python train_model.py first."
    )

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

# ─────────────────────────────────────────────
# LABEL MAPPINGS (user-readable → model numeric)
# ─────────────────────────────────────────────
GENDER_MAP = {"Male": 1, "Female": 0}

CHEST_PAIN_MAP = {
    "No Pain / Rarely" : 0,
    "Mild" : 1,
    "Moderate" : 2,
    "Severe / Frequent" : 3,
}

ECG_MAP = {
    "Normal" : 0,
    "ST-T Wave Abnormality" : 1,
    "Left Ventricular Hypertrophy" : 2,
}

ST_SLOPE_MAP = {
    "Upsloping (Lower Risk)" : 0,
    "Flat (Moderate Risk)" : 1,
    "Downsloping (High Risk)" : 2,
}

YES_NO_MAP = {"No": 0, "Yes": 1}


# ─────────────────────────────────────────────
# RISK ADJUSTMENT ENGINE
# Combines ML probability with clinical rules
# This is standard post-processing in medical AI
# ─────────────────────────────────────────────
def calculate_risk(base_prob, age, resting_bp, cholesterol,
                   chest_pain_val, exercise_angina_val, oldpeak):
    """
    Adds domain-based risk adjustments to the ML base probability.
    Each high-risk clinical indicator adds a small weight.
    Final probability is capped at 0.95 (never claim certainty).
    """
    risk_score = 0

    if age >= 60: risk_score += 1
    if resting_bp >= 150: risk_score += 1
    if cholesterol >= 240: risk_score += 1
    if chest_pain_val >= 2: risk_score += 1
    if exercise_angina_val == 1: risk_score += 1
    if oldpeak >= 2.0: risk_score += 1

    adjustment = risk_score * 0.05 # each factor adds 5%
    final_prob = min(base_prob + adjustment, 0.95)

    # Map to risk category
    if final_prob < 0.30:
        category = "Low Risk"
        message = "Your current indicators suggest a low risk. Keep maintaining a healthy lifestyle, regular exercise, and balanced diet."
        level_num = 1
    elif final_prob < 0.55:
        category = "Medium Risk"
        message = "Your indicators suggest moderate risk. Please monitor your health regularly, reduce high-calorie intake, and consult a physician for routine checkups."
        level_num = 2
    else:
        category = "High Risk"
        message = "Your indicators suggest elevated risk. Please consult a qualified doctor immediately for proper clinical evaluation."
        level_num = 3

    return {
        "category" : category,
        "message" : message,
        "probability": round(final_prob * 100, 1),
        "level_num" : level_num,
        "base_prob" : round(base_prob * 100, 1),
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Landing page — user selects mode"""
    return render_template("index.html")


@app.route("/normal")
def normal():
    """Normal User mode — simplified inputs"""
    return render_template(
        "normal.html",
        chest_pain_options = list(CHEST_PAIN_MAP.keys()),
    )


@app.route("/clinical")
def clinical():
    """Clinical User mode — full medical inputs"""
    return render_template(
        "clinical.html",
        chest_pain_options = list(CHEST_PAIN_MAP.keys()),
        ecg_options = list(ECG_MAP.keys()),
        st_slope_options = list(ST_SLOPE_MAP.keys()),
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives form data from either mode,
    runs the ML model, applies risk engine,
    then shows the result page.
    """
    try:
        mode = request.form.get("mode", "normal")

        # ── Common inputs (both modes) ──
        age = float(request.form["age"])
        gender_label = request.form["gender"]
        chest_pain_label = request.form["chest_pain"]
        resting_bp = float(request.form["resting_bp"])
        exercise_label = request.form["exercise_angina"]

        sex = GENDER_MAP[gender_label]
        chest_pain_val = CHEST_PAIN_MAP[chest_pain_label]
        exercise_val = YES_NO_MAP[exercise_label]

        # ── Clinical inputs (clinical mode only; defaults for normal mode) ──
        if mode == "clinical":
            cholesterol = float(request.form["cholesterol"])
            fbs_label = request.form["fasting_bs"]
            ecg_label = request.form["resting_ecg"]
            max_hr = float(request.form["max_hr"])
            oldpeak = float(request.form["oldpeak"])
            st_slope_label = request.form["st_slope"]

            fasting_bs_val = YES_NO_MAP[fbs_label]
            ecg_val = ECG_MAP[ecg_label]
            st_slope_val = ST_SLOPE_MAP[st_slope_label]
        else:
            # Sensible population-average defaults for normal mode
            cholesterol = 200.0
            fasting_bs_val = 0
            ecg_val = 0
            max_hr = 150.0
            oldpeak = 1.0
            st_slope_val = 1

        # ── Build feature vector (must match training column order) ──
        features = np.array([[
            age, sex, chest_pain_val, resting_bp, cholesterol,
            fasting_bs_val, ecg_val, max_hr, exercise_val,
            oldpeak, st_slope_val
        ]])

        # ── Scale and predict ──
        features_scaled = scaler.transform(features)
        base_prob = model.predict_proba(features_scaled)[0][1]

        # ── Apply risk adjustment engine ──
        result = calculate_risk(
            base_prob, age, resting_bp, cholesterol,
            chest_pain_val, exercise_val, oldpeak
        )

        result["mode"] = "Clinical User" if mode == "clinical" else "Normal User"

        # Store in session for result page
        session["result"] = result

        return redirect(url_for("result"))

    except Exception as e:
        return render_template("error.html", error=str(e))


@app.route("/result")
def result():
    """Displays prediction result on a fresh page"""
    result_data = session.get("result")
    if not result_data:
        return redirect(url_for("index"))
    return render_template("result.html", result=result_data)


@app.route("/about")
def about():
    """About / Team page"""
    return render_template("about.html")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Starting Heart Disease Risk Predictor...")
    print(" Open your browser at: http://127.0.0.1:5000\n")
    app.run(debug=True)
