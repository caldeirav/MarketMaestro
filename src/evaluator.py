from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from typing import List, Dict, Any
import json

from src.config import MODEL_SERVICE, API_KEY, CRITERIA
from src.agent import get_stock_recommendations

# Initialize the LLM
llm = ChatOpenAI(base_url=MODEL_SERVICE, api_key=API_KEY)

# Create a custom evaluator that provides reasoning and a normalized score
def create_custom_evaluator(criterion: str, criterion_description: str):
    prompt = PromptTemplate(
        input_variables=["input", "output", "criterion", "criterion_description"],
        template="""You are an expert evaluator of stock recommendations. Your task is to evaluate the following response based on the given criterion.

Human Input: {input}
AI Response: {output}

Criterion: {criterion}
Criterion Description: {criterion_description}

Please provide your evaluation in the following steps:
1. Analyze the response thoroughly.
2. Provide a detailed chain of thought reasoning for your evaluation.
3. Based on your reasoning, assign a score from 1 to 10, where 1 is the lowest and 10 is the highest.

Chain of Thought Reasoning:

Score (1-10):

Ensure that your reasoning is comprehensive and that your score accurately reflects your analysis."""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    def evaluate(input: str, output: str) -> Dict[str, Any]:
        result = chain.run(input=input, output=output, criterion=criterion, criterion_description=criterion_description)
        reasoning, score = result.split("Score (1-10):")
        return {
            "reasoning": reasoning.strip(),
            "score": int(score.strip())
        }
    
    return evaluate

# Create evaluators
evaluators = {
    criterion: create_custom_evaluator(criterion, description)
    for criterion, description in CRITERIA.items()
}

# Function to run evaluation
def evaluate_agent_response(query: str, response: str) -> Dict[str, Dict[str, Any]]:
    results = {}
    for criterion, evaluator in evaluators.items():
        result = evaluator(query, response)
        results[criterion] = result
    return results

# Example queries for evaluation
example_queries: List[str] = [
    "What are the top 3 US tech stocks you would recommend for medium term appreciation over the next 3 years?",
    "Can you suggest 5 undervalued tech stocks with high growth potential?",
    "What are some promising AI-focused stocks that could see significant gains in the next 2-3 years?"
]

# Run evaluations
def run_evaluations() -> List[Dict[str, Any]]:
    all_results = []
    for query in example_queries:
        response = get_stock_recommendations(query)['output']
        evaluation = evaluate_agent_response(query, response)
        all_results.append({
            "query": query,
            "response": response,
            "evaluation": evaluation
        })
    return all_results

# Calculate average scores across all evaluations
def calculate_average_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
    total_scores = {criterion: 0 for criterion in CRITERIA}
    for result in results:
        for criterion, evaluation in result['evaluation'].items():
            total_scores[criterion] += evaluation['score']
    return {criterion: score / len(results) for criterion, score in total_scores.items()}