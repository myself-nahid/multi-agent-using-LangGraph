import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, Any, List
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from fastapi.responses import StreamingResponse
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

app = FastAPI(title="Multi-Agent AI Platform", lifespan=lifespan)

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
    """Main chat endpoint that handles persistent, multi-turn conversations."""
    agentic_graph = request.app.state.agentic_graph
    session_id = query.session_id or str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": session_id}}

    inputs = {"messages": [HumanMessage(content=query.message)]}
    
    if not database.workflow_exists(session_id):
        database.create_workflow(session_id, query.agent_type)
        inputs["agent_name"] = query.agent_type

    response_stream = agentic_graph.astream_events(inputs, config=config, version="v1")
    
    async def stream_generator():
        """Generator to stream only the content chunks back to the client."""
        yield f"session_id: {session_id}\n\n"
        async for event in response_stream:
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield chunk.content

    return StreamingResponse(stream_generator(), media_type="text/plain")
