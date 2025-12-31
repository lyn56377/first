from fastapi import FastAPI
from pydantic import BaseModel
from agents import run_pipeline  # this is your logic from all10.py

app = FastAPI()

class AgentRequest(BaseModel):
    problem: str

class AgentResponse(BaseModel):
    result: str

@app.post("/run", response_model=AgentResponse)
def run_agents(req: AgentRequest):
    output = run_pipeline(req.problem)
    return {"result": output}

from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory=".", html=True), name="static")
