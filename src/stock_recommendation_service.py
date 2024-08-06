from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from src.agent import StockRecommendationAgent

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

class StockRecommendationService:
    def __init__(self):
        self.agent = StockRecommendationAgent()

    async def execute_task(self, task: str) -> str:
        return self.agent.get_stock_recommendations(task)

stock_service = StockRecommendationService()

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    try:
        response = await stock_service.execute_task(request.query)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)