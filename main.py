import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import HumanMessage

from graph import create_graph
import database

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to set up the agentic graph on startup and clean up on shutdown.
    """
    database.init_db()
    async with AsyncSqliteSaver.from_conn_string("conversation_memory.sqlite") as memory:
        agentic_graph = create_graph(checkpointer=memory)
        app.state.agentic_graph = agentic_graph
        yield

app = FastAPI(title="Full Multi-Agent AI Platform", lifespan=lifespan)

class UserQuery(BaseModel):
    message: str
    session_id: str | None = None
    agent_type: str

@app.get("/workflows", response_model=List[Dict[str, Any]])
async def get_workflows():
    """Endpoint for the 'Workflow Overview' UI component."""
    return database.get_all_workflows()

@app.post("/chat")
async def handle_chat(request: Request, query: UserQuery):
    """Main chat endpoint that handles a full request/response cycle and returns a single JSON object."""
    agentic_graph = request.app.state.agentic_graph
    session_id = query.session_id or str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": session_id}}
    inputs = {"messages": [HumanMessage(content=query.message)]}
    
    if not database.workflow_exists(session_id):
        database.create_workflow(session_id, query.agent_type)
        inputs["agent_name"] = query.agent_type

    final_state = await agentic_graph.ainvoke(inputs, config=config)

    ai_response_message = final_state["messages"][-1]
    response_content = ai_response_message.content
    
    agent_name = final_state.get("agent_name", query.agent_type)

    response_data = {
        "session_id": session_id,
        "response": response_content
    }
    print(f"--- Session {session_id} handled by {agent_name} ---")  
    print(f"Response: {response_content}")
    return JSONResponse(content=response_data)