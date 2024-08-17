import os
import logging

# Model and API configuration
MODEL_SERVICE = "http://localhost:53079/v1/"
API_KEY = "sk-no-key-required"

# Directory configuration
ANNUAL_REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'annual_reports')

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Evaluation criteria
CRITERIA = {
    "relevance": "The response should be directly relevant to stock recommendations for medium-term appreciation.",
    "specificity": "The response should provide specific stock recommendations, not just general advice.",
    "justification": "The response should include clear justifications for each stock recommendation.",
    "diversity": "The recommendations should cover a diverse range of tech stocks, not just the most well-known ones.",
    "risk_awareness": "The response should acknowledge potential risks or uncertainties in the recommendations."
}

def setup_logging():
    logging.basicConfig(level=LOG_LEVEL, 
                        format=LOG_FORMAT,
                        datefmt=LOG_DATE_FORMAT,
                        force=True)  # Force reconfiguration of the root logger

# Initialize logging
setup_logging()