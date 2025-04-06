import string
import pickle
import numpy as np
from sklearn.base import BaseEstimator
from nltk.tokenize import word_tokenize
from contractions import fix
import nltk

nltk.download('punkt')
def preprocess_text(text: str) -> str:
    """
    Preprocess the input text.
    Steps: Lowercasing, expanding contractions, removing punctuation, tokenizing, and filtering short words.
    """
    text = text.lower()  # Lowercase
    text = fix(text)  # Expand contractions
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenization
    words = [word for word in words if len(word) > 1]  # Remove short words
    return " ".join(words) if words else "emptytext"

def score(text: str, model: BaseEstimator, threshold: float = 0.5) -> tuple[bool, float]:
    """
    Scores a trained pipeline on a given text.
    
    Args:
        text (str): The input text.
        model (BaseEstimator): The trained pipeline (TF-IDF + Classifier).
        threshold (float): Decision threshold for classification.
    
    Returns:
        tuple[bool, float]: (Prediction, Propensity Score)
    """
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    if not isinstance(model, BaseEstimator):
        raise ValueError("Model must be an instance of sklearn BaseEstimator")
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1")

    # Preprocess the text
    processed_text = preprocess_text(text)

    # Predict using the pipeline (TF-IDF + Classifier)
    if hasattr(model, "predict_proba"):
        propensity = model.predict_proba([processed_text])[0][1]  # Probability of positive class
    else:
        propensity = model.decision_function([processed_text])
        propensity = 1 / (1 + np.exp(-propensity))  # Sigmoid function for probability

    # Make a prediction based on threshold
    prediction = propensity > threshold
    
    return bool(prediction), float(propensity)  


