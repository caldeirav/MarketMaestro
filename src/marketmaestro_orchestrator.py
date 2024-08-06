import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from crewai import Crew, Task, Agent

from src.agent import StockRecommendationAgent

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrewTaskRequest(BaseModel):
    tasks: List[str]
    context: Optional[Dict[str, Any]] = None
    tools: Optional[List[str]] = None

class CrewTaskResponse(BaseModel):
    results: List[str]
    token_usage: Optional[Dict[str, int]] = None

@app.post("/execute_crew_tasks", response_model=CrewTaskResponse)
async def execute_crew_tasks(request: CrewTaskRequest):
    try:
        logger.info("Initializing StockRecommendationAgent")
        stock_agent = StockRecommendationAgent()
        
        logger.info(f"Creating tasks: {request.tasks}")
        tasks = [
            Task(
                description=task,
                expected_output="A detailed stock recommendation based on the query.",
                agent=stock_agent
            ) for task in request.tasks
        ]

        logger.info("Initializing Crew")
        crew = Crew(
            agents=[stock_agent],
            tasks=tasks,
            verbose=True
        )

        logger.info("Kicking off crew tasks")
        results = crew.kickoff()
        
        # Log the structure of the results
        logger.info(f"Results structure: {type(results)}")
        logger.info(f"Results attributes: {dir(results)}")
        
        # Extract the actual results and token usage
        task_results = []
        for task in results.tasks_output:
            logger.info(f"Task output structure: {type(task)}")
            logger.info(f"Task output attributes: {dir(task)}")
            # Attempt to get the raw output, fallback to string representation
            task_results.append(getattr(task, 'raw', str(task)))
        
        token_usage = getattr(results, 'token_usage', None)
        
        logger.info(f"Crew tasks completed. Results: {task_results}")
        return CrewTaskResponse(results=task_results, token_usage=token_usage)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)