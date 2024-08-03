from src.agent import get_stock_recommendations
from src.config import setup_logging
import logging

# Ensure logging is set up
setup_logging()

def main():
    query = input("Enter your stock recommendation query: ")
    logging.info(f"Received query: {query}")
    result = get_stock_recommendations(query)
    logging.info("Recommendation process completed. Printing results:")
    print(result)

if __name__ == "__main__":
    main()