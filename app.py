from flask import Flask, render_template, request
import pickle
import re

# Initialize Flask app
app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function (MUST be same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        text = request.form["news"]

        # Clean input text
        cleaned_text = clean_text(text)

        # Vectorize
        vect_text = vectorizer.transform([cleaned_text])

        # Prediction
        prediction = model.predict(vect_text)[0]
        confidence = max(model.predict_proba(vect_text)[0]) * 100

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
