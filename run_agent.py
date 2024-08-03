from src.agent import get_stock_recommendations

if __name__ == "__main__":
    query = input("Enter your stock recommendation query: ")
    result = get_stock_recommendations(query)
    print(result['output'])