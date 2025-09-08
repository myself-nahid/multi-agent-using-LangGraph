import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware

from graph import create_graph
import database
from offer_service import app as offer_app
from agents import AGENT_RUNNABLES

AGENT_NAMES = list(AGENT_RUNNABLES.keys())

def clean_agent_name(name: str) -> str:
    """Helper function to normalize agent names for comparison."""
    return "".join(filter(str.isalnum, name)).lower()

CLEANED_AGENT_NAMES_MAP = {clean_agent_name(name): name for name in AGENT_NAMES}

def validate_agent_type(agent_type: str) -> str:
    """
    Cleans and validates the agent_type from the frontend request.
    Returns the correct agent name or raises an HTTPException if invalid.
    """
    cleaned_type = clean_agent_name(agent_type)
    if cleaned_type in CLEANED_AGENT_NAMES_MAP:
        return CLEANED_AGENT_NAMES_MAP[cleaned_type]
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid agent_type '{agent_type}'. Valid options are: {AGENT_NAMES}"
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    await offer_app.router.startup()
    async with AsyncSqliteSaver.from_conn_string("conversation_memory.sqlite") as memory:
        agentic_graph = create_graph(checkpointer=memory)
        app.state.agentic_graph = agentic_graph
        yield
    await offer_app.router.shutdown()

app = FastAPI(title="Full Multi-Agent AI Platform", lifespan=lifespan)

origins = ["http://localhost", "http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4200", "http://127.0.0.1:4200", "http://127.0.0.1:3901"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/offers-api", offer_app, name="offers_api")

class UserQuery(BaseModel):
    message: str
    session_id: str | None = None
    agent_type: str

@app.get("/workflows", response_model=List[Dict[str, Any]])
async def get_workflows():
    return database.get_all_workflows()

@app.post("/chat")
async def handle_chat(request: Request, query: UserQuery):
    agentic_graph = request.app.state.agentic_graph
    session_id = query.session_id or str(uuid.uuid4())
    
    validated_agent_name = validate_agent_type(query.agent_type)
    
    config = {"configurable": {"thread_id": session_id}}
    inputs = {
        "messages": [HumanMessage(content=query.message)],
        "agent_name": validated_agent_name  
    }
    
    if not database.workflow_exists(session_id):
        database.create_workflow(session_id, validated_agent_name)

    final_state = await agentic_graph.ainvoke(inputs, config=config)
    ai_response_message = final_state["messages"][-1]
    response_content = ai_response_message.content
    agent_name = final_state.get("agent_name", validated_agent_name)
    response_data = {"session_id": session_id, "response": response_content, "agent_name": agent_name}
    return JSONResponse(content=response_data)