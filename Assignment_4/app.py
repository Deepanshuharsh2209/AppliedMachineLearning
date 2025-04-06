# import libraries
from flask import Flask, request, jsonify, render_template_string
import pickle
from score import score
import warnings
import os

# ignore warnings
warnings.filterwarnings("ignore")

# create flask app
app = Flask(__name__)

# load model
filename = os.path.join(os.path.dirname(__file__),"best_pipeline.pkl")
model = pickle.load(open(filename, "rb"))

# html template for the home route
html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Classification - Spam Detector</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            font-family: 'Inter', sans-serif;
        }

        body {
            margin: 0;
            padding: 0;
            background: #eef2f7;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .container {
            background: #ffffff;
            padding: 40px;
            width: 100%;
            max-width: 500px;
            border-radius: 16px;
            box-shadow: 0 20px 30px rgba(0,0,0,0.1);
        }

        h2 {
            margin-bottom: 20px;
            text-align: center;
            color: #333;
            font-weight: 600;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
        }

        input[type="text"] {
            width: 100%;
            padding: 12px;
            margin-bottom: 20px;
            border: 1px solid #ccc;
            border-radius: 8px;
            font-size: 16px;
        }

        input[type="submit"] {
            width: 100%;
            background-color: #1e88e5;
            color: #fff;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        input[type="submit"]:hover {
            background-color: #1565c0;
        }

        .result {
            margin-top: 30px;
            padding: 20px;
            background-color: #f6fafd;
            border-left: 5px solid #1e88e5;
            border-radius: 8px;
        }

        .result h3 {
            margin-top: 0;
            color: #1e88e5;
        }

        .result p {
            margin: 5px 0;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Spam Detection Tool</h2>
        <form action="/score" method="post">
            <label for="text">Paste a message below:</label>
            <input type="text" id="text" name="text" placeholder="e.g., Win a free iPhone now!">
            <input type="submit" value="Analyze Text">
        </form>

        {% if prediction %}
        <div class="result">
            <h3>Classification Result</h3>
            <p><strong>Prediction:</strong> {{ prediction }}</p>
            <p><strong>Propensity Score:</strong> {{ probability }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""
# home route
@app.route('/', methods=['GET'])
def home():
    """
    Handle GET requests to the home route.

    The home route renders the HTML template for the web form.
    """
    return render_template_string(html)

# score endpoint
@app.route('/score', methods=['POST'])
def score_endpoint():
    """
    Handle POST requests to the score endpoint.

    The score endpoint processes the submitted text and scores it using the model.
    The prediction and propensity are returned as a JSON response.

    Returns:
        JSON: A JSON response containing the prediction and propensity.
    """
    
    text = request.form['text']
    prediction, probability = score(text, model, 0.50)
    response = {'prediction': prediction, 'propensity': probability}
    return jsonify(response)
    
# run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
