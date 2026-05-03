from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "heartcare_secret_2025"

# ─────────────────────────────────────────────
# SAFE BASE DIRECTORY (important for deployment)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH  = os.path.join(BASE_DIR, "model", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Please run: python train_model.py first."
    )

model  = pickle.load(open(MODEL_PATH,  "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

# ─────────────────────────────────────────────
# LABEL MAPPINGS
# ─────────────────────────────────────────────
GENDER_MAP = {"Male": 1, "Female": 0}

CHEST_PAIN_MAP = {
    "No Pain / Rarely"    : 0,
    "Mild"                : 1,
    "Moderate"            : 2,
    "Severe / Frequent"   : 3,
}

ECG_MAP = {
    "Normal"                          : 0,
    "ST-T Wave Abnormality"           : 1,
    "Left Ventricular Hypertrophy"    : 2,
}

ST_SLOPE_MAP = {
    "Upsloping (Lower Risk)"  : 0,
    "Flat (Moderate Risk)"    : 1,
    "Downsloping (High Risk)" : 2,
}

YES_NO_MAP = {"No": 0, "Yes": 1}


# ─────────────────────────────────────────────
# RISK ENGINE (unchanged)
# ─────────────────────────────────────────────
def calculate_risk(base_prob, age, resting_bp, cholesterol,
                   chest_pain_val, exercise_angina_val, oldpeak):

    risk_score = 0

    if age >= 60:                   risk_score += 1
    if resting_bp >= 150:           risk_score += 1
    if cholesterol >= 240:          risk_score += 1
    if chest_pain_val >= 2:         risk_score += 1
    if exercise_angina_val == 1:    risk_score += 1
    if oldpeak >= 2.0:              risk_score += 1

    adjustment  = risk_score * 0.05
    final_prob  = min(base_prob + adjustment, 0.95)

    if final_prob < 0.30:
        category  = "Low Risk"
        message   = "Your current indicators suggest a low risk. Keep maintaining a healthy lifestyle."
        level_num = 1
    elif final_prob < 0.55:
        category  = "Medium Risk"
        message   = "Moderate risk. Monitor health and consult a doctor."
        level_num = 2
    else:
        category  = "High Risk"
        message   = "High risk. Please consult a doctor immediately."
        level_num = 3

    return {
        "category"   : category,
        "message"    : message,
        "probability": round(final_prob * 100, 1),
        "level_num"  : level_num,
        "base_prob"  : round(base_prob * 100, 1),
    }


# ─────────────────────────────────────────────
# ROUTES (UNCHANGED)
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/normal")
def normal():
    return render_template(
        "normal.html",
        chest_pain_options=list(CHEST_PAIN_MAP.keys()),
    )


@app.route("/clinical")
def clinical():
    return render_template(
        "clinical.html",
        chest_pain_options=list(CHEST_PAIN_MAP.keys()),
        ecg_options=list(ECG_MAP.keys()),
        st_slope_options=list(ST_SLOPE_MAP.keys()),
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        mode = request.form.get("mode", "normal")

        age = float(request.form["age"])
        gender_label = request.form["gender"]
        chest_pain_label = request.form["chest_pain"]
        resting_bp = float(request.form["resting_bp"])
        exercise_label = request.form["exercise_angina"]

        sex = GENDER_MAP[gender_label]
        chest_pain_val = CHEST_PAIN_MAP[chest_pain_label]
        exercise_val = YES_NO_MAP[exercise_label]

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
            cholesterol = 200.0
            fasting_bs_val = 0
            ecg_val = 0
            max_hr = 150.0
            oldpeak = 1.0
            st_slope_val = 1

        features = np.array([[
            age, sex, chest_pain_val, resting_bp, cholesterol,
            fasting_bs_val, ecg_val, max_hr, exercise_val,
            oldpeak, st_slope_val
        ]])

        features_scaled = scaler.transform(features)
        base_prob = model.predict_proba(features_scaled)[0][1]

        result = calculate_risk(
            base_prob, age, resting_bp, cholesterol,
            chest_pain_val, exercise_val, oldpeak
        )

        result["mode"] = "Clinical User" if mode == "clinical" else "Normal User"

        session["result"] = result

        return redirect(url_for("result"))

    except Exception as e:
        return render_template("error.html", error=str(e))


@app.route("/result")
def result():
    result_data = session.get("result")
    if not result_data:
        return redirect(url_for("index"))
    return render_template("result.html", result=result_data)


@app.route("/about")
def about():
    return render_template("about.html")


# ─────────────────────────────────────────────
# PRODUCTION RUN CONFIG (FIXED)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)