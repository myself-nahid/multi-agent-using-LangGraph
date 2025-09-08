from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from agents import AGENT_RUNNABLES
from tools import all_tools 

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    agent_name: str

def agent_node(state: AgentState):
    """Invokes the agent selected by the router."""
    agent_name = state["agent_name"]
    result = AGENT_RUNNABLES[agent_name].invoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name=agent_name)]}

tool_node = ToolNode(all_tools)
agent_names = list(AGENT_RUNNABLES.keys())

def router(state: AgentState) -> str:
    """Directs the conversation to the agent whose name is in the state."""
    return state["agent_name"]

def create_graph(checkpointer):
    """Factory function to create and compile the stateful graph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("tools", tool_node)
    for agent_name in agent_names:
        workflow.add_node(agent_name, agent_node)

    workflow.set_conditional_entry_point(
        router,
        {name: name for name in agent_names},
    )

    for agent_name in agent_names:
        workflow.add_conditional_edges(
            agent_name,
            tools_condition,
            {END: END, "tools": "tools"}
        )
        
    workflow.add_conditional_edges(
        "tools",
        router,
        {name: name for name in agent_names}
    )

    return workflow.compile(checkpointer=checkpointer)