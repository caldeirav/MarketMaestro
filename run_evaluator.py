from src.evaluator import run_evaluations, calculate_average_scores
import json

if __name__ == "__main__":
    results = run_evaluations()
    print(json.dumps(results, indent=2))

    average_scores = calculate_average_scores(results)
    print("\nAverage Scores:")
    print(json.dumps(average_scores, indent=2))