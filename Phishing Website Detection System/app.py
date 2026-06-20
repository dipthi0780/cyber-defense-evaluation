from flask import Flask, render_template, request
from flask_cors import cross_origin
import pandas as pd
import pickle

from FeatureExtraction import featureExtraction

app = Flask(__name__, template_folder="templates")

# ===============================
# LOAD TRAINED MODEL
# ===============================

model = pickle.load(open("RandomForest_model.pkl", "rb"))

# ===============================
# HOME PAGE
# ===============================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ===============================
# DETECT URL PAGE
# ===============================

@app.route("/detecturl")
def detecturl():
    return render_template("DetectURL.html")

# ===============================
# ABOUT PAGE
# ===============================

@app.route('/about')
def about():
    return render_template('about.html')

# ===============================
# IMPLEMENTATION PAGE
# ===============================

@app.route('/behindthescenes')
def behindthescenes():
    return render_template("behindthescenes.html")

# ===============================
# PREDICTION ROUTE
# ===============================

@app.route("/prediction", methods=["POST"])
def prediction():

    url = request.form.get("url", "").strip()

    print("Received URL:", url)

    if url == "":
        return render_template(
            "DetectURL.html",
            prediction_text="❗ Please enter a URL",
            confidence=""
        )

    # Add http if missing
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    # ===============================
    # FEATURE EXTRACTION
    # ===============================

    features = featureExtraction(url)
    print("Extracted Features:", features)

    df = pd.DataFrame([features])

    # ===============================
    # MODEL PREDICTION
    # ===============================

    probs = model.predict_proba(df)[0]

    legit_prob = probs[0]
    phish_prob = probs[1]

    print("Legitimate Probability:", legit_prob)
    print("Phishing Probability:", phish_prob)

    # ===============================
    # CLASSIFICATION + CONFIDENCE
    # ===============================

    if phish_prob > legit_prob:

        confidence = phish_prob * 100

        if phish_prob >= 0.80:
            result = "🚨 High Risk Phishing Website"

        elif phish_prob >= 0.50:
            result = "⚠️ Suspicious Website"

        else:
            result = "⚠️ Possible Phishing Website"

    else:

        confidence = legit_prob * 100
        result = "✅ Legitimate Website"

    confidence = round(confidence, 2)

    # ===============================
    # RETURN RESULT
    # ===============================

    return render_template(
        "DetectURL.html",
        prediction_text=result,
        confidence=str(confidence) + "%"
    )

# ===============================
# RUN APP
# ===============================

if __name__ == "__main__":
    app.run(debug=True)