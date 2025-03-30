from flask import Flask, request, jsonify, render_template_string
import pickle
import warnings
import os
from score import score

# Ignore warnings
warnings.filterwarnings("ignore")

# Create Flask app
app = Flask(__name__)

# Load the trained pipeline (includes TF-IDF vectorizer and model)
pipeline_path = r"D:\CMI\SEM4\AML\Assignment_3\Model\best_pipeline.pkl"
with open(pipeline_path, "rb") as file:
    pipeline = pickle.load(file)

# HTML template for the home route
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            margin: 0;
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            text-align: center;
            width: 350px;
        }
        h1 {
            margin-bottom: 15px;
        }
        input[type="text"] {
            width: 90%;
            padding: 10px;
            margin: 15px 0;
            border: none;
            border-radius: 5px;
        }
        input[type="submit"] {
            width: 100%;
            padding: 10px;
            border: none;
            background: #ff9800;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background: #e68900;
        }
        .result {
            margin-top: 20px;
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Spam Classifier</h1>
        <form action="/score" method="post">
            <input type="text" id="text" name="text" placeholder="Enter your text here..." required>
            <input type="submit" value="Classify">
        </form>
        {% if prediction is not none %}
            <div class="result">
                <h2>Prediction:</h2>
                <p><strong>{{ prediction }}</strong></p>
                <h2>Propensity Score:</h2>
                <p>{{ propensity }}</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

# Home route
@app.route("/", methods=["GET"])
def home():
    """
    Renders a simple HTML form for user input.
    """
    return render_template_string(html_template)

# Score endpoint
@app.route("/score", methods=["POST"])
def score_endpoint():
    """
    Handles POST requests to classify text.
    Returns:
        JSON: A response with prediction and propensity score.
    """
    text = request.form.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    prediction, propensity = score(text, pipeline, 0.50)
    response = {"prediction": prediction, "propensity": propensity}
    return jsonify(response)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
