from src.evaluator import run_evaluations, calculate_average_scores
from src.config import setup_logging
import json
import logging

# Set up logging
setup_logging()

if __name__ == "__main__":
    logging.info("Starting evaluation process")
    
    results = run_evaluations()
    print("Evaluation Results:")
    print(json.dumps(results, indent=2))
    logging.info("Evaluation results generated")

    average_scores = calculate_average_scores(results)
    print("\nAverage Scores:")
    print(json.dumps(average_scores, indent=2))
    logging.info("Average scores calculated")

    logging.info("Evaluation process completed")