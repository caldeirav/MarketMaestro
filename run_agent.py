from src.agent import StockRecommendationAgent
from src.config import setup_logging
import logging

# Ensure logging is set up
setup_logging()

def main():
    # Create an instance of the StockRecommendationAgent
    agent = StockRecommendationAgent()

    query = input("Enter your stock recommendation query: ")
    logging.info(f"Received query: {query}")

    # Execute the task using the agent
    result = agent.execute_task(query, context=None, tools=None)

    logging.info("Recommendation process completed. Printing results:")
    print(result)

if __name__ == "__main__":
    main()