import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from crewai import Crew, Task, Agent

from src.agent import StockRecommendationAgent

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrewTaskRequest(BaseModel):
    tasks: List[str]

class CrewTaskResponse(BaseModel):
    results: List[str]

@app.post("/execute_crew_tasks", response_model=CrewTaskResponse)
async def execute_crew_tasks(request: CrewTaskRequest):
    try:
        logger.info("Initializing StockRecommendationAgent")
        stock_agent = StockRecommendationAgent()
        
        logger.info(f"Creating tasks: {request.tasks}")
        tasks = [
            Task(
                description=task,
                agent=stock_agent
            ) for task in request.tasks
        ]

        logger.info("Initializing Crew")
        crew = Crew(
            agents=[stock_agent],
            tasks=tasks
        )

        logger.info("Kicking off crew tasks")
        results = crew.kickoff()
        logger.info(f"Crew tasks completed. Results: {results}")
        return CrewTaskResponse(results=results)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)