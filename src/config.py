import os

# Model configuration
MODEL_SERVICE = "http://localhost:63887/v1/"
API_KEY = "sk-no-key-required"

# Data directory
ANNUAL_REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'annual_reports')

# Evaluation criteria
CRITERIA = {
    "relevance": "The response should be directly relevant to stock recommendations for medium-term appreciation.",
    "specificity": "The response should provide specific stock recommendations, not just general advice.",
    "justification": "The response should include clear justifications for each stock recommendation.",
    "diversity": "The recommendations should cover a diverse range of tech stocks, not just the most well-known ones.",
    "risk_awareness": "The response should acknowledge potential risks or uncertainties in the recommendations."
}