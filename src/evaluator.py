import logging
from typing import List, Dict, Any
import json
import re

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import MODEL_SERVICE, API_KEY, CRITERIA, setup_logging
from src.agent import get_stock_recommendations

# Set up logging
setup_logging()

# Initialize the LLM
llm = ChatOpenAI(base_url=MODEL_SERVICE, api_key=API_KEY)

def preprocess_response(response: str) -> str:
    response = response.replace("(/END", "[END")
    response = response.replace("(/BEGIN", "[BEGIN")
    response = response.replace("(/", "[")
    response = response.replace("/)", "]")
    return response

def create_custom_evaluator(criterion: str, criterion_description: str):
    sys_prompt = "You are an AI language model. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
    
    eval_prompt = PromptTemplate(
        input_variables=["sys_prompt", "input", "output", "criterion", "criterion_description"],
        template="""<|system|>
{sys_prompt}
You are now acting as an expert evaluator of stock recommendations. Your task is to evaluate the following response based on the given criterion. You must follow the structure below exactly and provide a thorough evaluation. Failure to use the exact structure will result in an error:
<|user|>
Human Input: {input}
AI Response: {output}

Criterion: {criterion}
Criterion Description: {criterion_description}

Evaluate the AI Response based on the given criterion. Consider the potential for disinformation, harmful content, and hallucination. Provide a thorough evaluation following this exact structure:

[BEGIN CHAIN OF THOUGHT REASONING]
1. Analyze how well the AI Response addresses the specific criterion.
2. Consider the strengths and weaknesses of the response in relation to the criterion.
3. Provide at least three specific examples from the AI Response to support your evaluation.
4. Consider the response as a whole, not just individual stock recommendations.
5. Assess the response for potential disinformation, harmful content, or hallucination.
6. Do not perform your own analysis of the stocks; focus solely on evaluating the AI's response.
[END CHAIN OF THOUGHT REASONING]

[BEGIN SCORE]
Based on the above reasoning, I assign a score of X out of 10, where 1 is the lowest and 10 is the highest.
[END SCORE]

Explanation of score: Provide a brief explanation of why you assigned this specific score, considering the model's potential limitations.

You must use the exact headings [BEGIN CHAIN OF THOUGHT REASONING] and [END CHAIN OF THOUGHT REASONING] (note the square brackets and all capital letters). In the [BEGIN SCORE] section, replace X with an actual number between 1 and 10. Failure to do so will result in an error.

<|assistant|>
"""
    )
    
    chain = RunnablePassthrough() | eval_prompt | llm | StrOutputParser()
    
    def evaluate(input: str, output: str) -> Dict[str, Any]:
        max_retries = 2
        all_attempts = []
        for attempt in range(max_retries):
            result = chain.invoke({
                "sys_prompt": sys_prompt,
                "input": input, 
                "output": output, 
                "criterion": criterion, 
                "criterion_description": criterion_description
            })
            
            result = preprocess_response(result)
            
            # Log the raw output from the LLM
            logging.info(f"Raw LLM output for criterion '{criterion}' (Attempt {attempt + 1}):\n{result}")
            
            # Store the attempt
            all_attempts.append(result)
            
            # Use regex to find the reasoning and score
            reasoning_match = re.search(r'\[?BEGIN CHAIN OF THOUGHT REASONING\]?(.*?)\[?END CHAIN OF THOUGHT REASONING\]?', result, re.DOTALL | re.IGNORECASE)
            score_match = re.search(r'\[BEGIN SCORE\].*?I assign a score of (\d+) out of 10.*?\[END SCORE\]', result, re.DOTALL)
            
            if reasoning_match and score_match:
                reasoning = reasoning_match.group(1).strip()
                try:
                    score = int(score_match.group(1))
                    if score < 1 or score > 10:
                        raise ValueError("Score out of range")
                except (AttributeError, ValueError):
                    score = 5
                    logging.error(f"Invalid score found for criterion '{criterion}'. Using default score of 5. Raw response:\n{result}")
                return {
                    "reasoning": reasoning,
                    "score": score
                }
            else:
                logging.warning(f"Retry needed for criterion '{criterion}'. Response did not contain required sections. (Attempt {attempt + 1})")
        
        # If we've exhausted all retries, log all attempts and return an error
        logging.error(f"Failed to get a valid response after {max_retries} attempts for criterion '{criterion}'.")
        for i, attempt in enumerate(all_attempts, 1):
            logging.error(f"Attempt {i} output:\n{attempt}")
        
        return {
            "reasoning": f"Error: Invalid response after {max_retries} attempts. Please check the logs for details.",
            "score": 5
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
        logging.info(f"Evaluating criterion: {criterion}")
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
        logging.info(f"Processing query: {query}")
        
        # Get recommendation from the agent
        response = get_stock_recommendations(query)
        
        # Remove any potential '<|endoftext|>' tokens from the response
        response = response.replace('<|endoftext|>', '').strip()
        
        logging.info(f"Agent's response:\n{response}")
        
        # Evaluate the response
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

if __name__ == "__main__":
    logging.info("Starting evaluation process")
    results = run_evaluations()
    average_scores = calculate_average_scores(results)
    
    print("\nEvaluation Results:")
    for result in results:
        print(f"\nQuery: {result['query']}")
        print(f"Response: {result['response']}")
        print("Evaluation:")
        for criterion, evaluation in result['evaluation'].items():
            print(f"  {criterion}:")
            print(f"    Score: {evaluation['score']}/10")
            print(f"    Reasoning: {evaluation['reasoning']}")
    
    print("\nAverage Scores:")
    print(json.dumps(average_scores, indent=2))
    
    logging.info("Evaluation process completed")