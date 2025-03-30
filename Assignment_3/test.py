import unittest
import pickle
from score import score
import subprocess
import time
import requests

# Function to load the pre-trained model
def load_test_model(filepath):
    """Loads a pre-trained test model from a pickle file."""
    with open(filepath, 'rb') as file:
        return pickle.load(file)

MESSAGES = [
    "Your package has been delivered. Track your order here.",
    "Congratulations! You’ve won a brand-new iPhone. Click to claim.",
    "Reminder: Your doctor's appointment is tomorrow at 10 AM.",
    "Final notice: Your car warranty is expiring soon. Renew now.",
    "Can you review the document I sent and provide feedback?",
    "Earn $500 daily from home! No skills required, apply now.",
    "Team outing this weekend! Let us know if you can join.",
    "Immediate action required: Your PayPal account is on hold.",
    "Free access to our premium webinar for a limited time.",
    "Meeting rescheduled to 3 PM. Please update your calendar.",
    "Your loan request has been approved. Submit your details here.",
    "Hurry! Only 5 spots left for our exclusive training program.",
    "Lunch break extended by 30 minutes today. Enjoy your meal!",
    "You have been selected for a VIP investment opportunity.",
    "The office will be closed on Monday for maintenance.",
    "Special offer: Buy 1, get 1 free on all purchases today.",
    "Your bank account was accessed from an unknown device.",
    "Reminder: Submit your project report by Friday evening.",
    "Exclusive deal: Get 70% off our best-selling products!",
    "Hello! Just checking in, how was your weekend?",
]

class TestScoreFunction(unittest.TestCase):
    """Test case for the score function."""
    
    @classmethod
    def setUpClass(cls):
        """Loads the trained model once before all test cases."""
        cls.model_path = r"D:\CMI\SEM4\AML\Assignment_3\Model\best_pipeline.pkl"
        cls.model = load_test_model(cls.model_path)

    def test_smoke(self):
        """Smoke test: Ensures function runs without crashing."""
        for message in MESSAGES:
            prediction, propensity = score(message, self.model, 0.5)
            self.assertIsNotNone(prediction)
            self.assertIsNotNone(propensity)

    def test_output_format(self):
        """Test if output types are correct."""
        for message in MESSAGES:
            prediction, propensity = score(message, self.model, 0.5)
            self.assertIsInstance(prediction, bool)
            self.assertIsInstance(propensity, float)

    def test_value_constraints(self):
        """Check if values are within expected ranges."""
        for message in MESSAGES:
            prediction, propensity = score(message, self.model, 0.5)
            self.assertIn(prediction, [True, False])
            self.assertGreaterEqual(propensity, 0.0)
            self.assertLessEqual(propensity, 1.0)

    def test_threshold_0_always_1(self):
        """Check if setting threshold = 0 always returns prediction = 1."""
        for message in MESSAGES:
            prediction, _ = score(message, self.model, 0.0)
            self.assertTrue(prediction)

    def test_threshold_1_always_0(self):
        """Check if setting threshold = 1 always returns prediction = 0."""
        for message in MESSAGES:
            prediction, _ = score(message, self.model, 1.0)
            self.assertFalse(prediction)

    def test_obvious_spam(self):
        """Check if obvious spam messages return prediction = 1."""
        spam_messages = [
            "You won a free vacation! Click here to claim now.",
            "Get rich fast! Earn $500 per day from home.",
            "Your account has been locked. Update your info immediately.",
        ]
        for message in spam_messages:
            prediction, _ = score(message, self.model, 0.5)
            self.assertTrue(prediction)

    def test_obvious_non_spam(self):
        """Check if obvious non-spam messages return prediction = 0."""
        non_spam_messages = [
            "Hey, are we still meeting at 5 PM today?",
            "Reminder: The report is due on Friday.",
            "Can you please review my code and provide feedback?",
        ]
        for message in non_spam_messages:
            prediction, _ = score(message, self.model, 0.5)
            self.assertFalse(prediction)


def test_flask():
    """
    Integration test for the Flask application.
    """

    # Start the Flask app in a separate process
    process = subprocess.Popen(["python", "app.py"])
    time.sleep(2)  # Allow the server to start

    try:
        # Test the home page
        response = requests.get("http://127.0.0.1:5000/")
        assert response.status_code == 200
        assert "Spam Classifier" in response.text  # Match with actual HTML title

        # Test the /score endpoint with multiple examples
        test_cases = [
            "This is a discount offer, claim now!",  # Likely spam
            "Meeting at 10 AM in the office.",  # Likely not spam
            "Congratulations! You've won a free gift card!",  # Spam
            "Let's grab lunch today?"  # Not spam
        ]

        for text in test_cases:
            response = requests.post("http://127.0.0.1:5000/score", data={"text": text})
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data and "propensity" in data
            assert isinstance(data["prediction"], bool)
            assert isinstance(data["propensity"], float)
            assert 0 <= data["propensity"] <= 1

        print("All Flask integration tests passed")

    finally:
        # Ensure the Flask app is stopped
        process.terminate()
