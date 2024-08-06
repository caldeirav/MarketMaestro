from src.evaluator import run_evaluations, calculate_average_scores
from src.config import setup_logging
import json
import logging

# Set up logging
setup_logging()

if __name__ == "__main__":
    logging.info("Starting evaluation process")
    
    results = run_evaluations()
    
    print("\nEvaluation Results:")
    for result in results:
        print(f"\nQuery: {result['query']}")
        print(f"Response: {result['response']}")
        print("Evaluation:")
        for criterion, evaluation in result['evaluation'].items():
            print(f"  {criterion}:")
            print(f"    Score: {evaluation['score']}/10")
            print(f"    Reasoning: {evaluation['reasoning']}")
    
    logging.info("Evaluation results generated")

    average_scores = calculate_average_scores(results)
    print("\nAverage Scores:")
    print(json.dumps(average_scores, indent=2))
    logging.info("Average scores calculated")

    logging.info("Evaluation process completed")