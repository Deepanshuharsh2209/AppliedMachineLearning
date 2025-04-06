import os
import time
import pickle
import requests
import subprocess

# Sample texts for classification
TEST_TEXTS = [
    "Reminder: Your dental appointment is scheduled for next Monday at 3 PM. Don’t forget!",
    "Breaking news: Scientists discover a new species of dinosaur in Argentina.",
    "Limited-time offer! Buy one get one free on all items until midnight. Shop now!",
    "Hi Alex, can you review the updated design by Thursday? Let me know your feedback.",
    "Urgent: Your bank account has been compromised. Click here immediately to secure it.",
    "Team, quarterly earnings call is at 11 AM tomorrow. Be ready with your decks.",
    "Congratulations! You’ve been chosen for an exclusive membership reward. Claim here.",
    "FYI: The maintenance work on the server will start tonight at 10 PM.",
]

def test_docker():
    """
    Launches the Docker container running the Flask spam classifier,
    sends test POST requests, and validates responses.
    """
    container_name = "spam_checker"
    image_name = "spam-detector-app"

    # Clean up any old instances
    subprocess.run(["docker", "stop", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "rm", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Build Docker image
    subprocess.run(["docker", "build", "-t", image_name, "."], check=True)

    # Run container
    subprocess.run([
        "docker", "run", "-d", "-p", "5000:5000", "--name", container_name, image_name
    ], check=True)

    # Allow time for Flask server to start
    time.sleep(6)

    # Test homepage
    homepage = requests.get("http://127.0.0.1:5000/")
    assert homepage.status_code == 200
    assert "Spam Classifier" in homepage.text

    # Test /score endpoint
    for text in TEST_TEXTS:
        response = requests.post("http://127.0.0.1:5000/score", data={"text": text})
        assert response.status_code == 200
        result = response.json()
        assert "prediction" in result and "propensity" in result
        assert isinstance(result["prediction"], bool)
        assert isinstance(result["propensity"], float)
        assert 0.0 <= result["propensity"] <= 1.0
        assert result["prediction"] in [0, 1]

    # Cleanup
    subprocess.run(["docker", "stop", container_name], check=True)
    subprocess.run(["docker", "rm", container_name], check=True)
