from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from agents import AGENT_RUNNABLES
from tools import booking_tools, email_agent_tools
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

all_tools = booking_tools + email_agent_tools

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    agent_name: str

def agent_node(state: AgentState):
    """Invokes the agent selected by the router."""
    agent_name = state["agent_name"]
    result = AGENT_RUNNABLES[agent_name].invoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result["output"], name=agent_name)]}

tool_node = ToolNode(all_tools)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

agent_names = list(AGENT_RUNNABLES.keys())
supervisor_prompt = PromptTemplate.from_template(
    """You are a supervisor managing a team of assistants: {agent_names}.
Based on the initial user request, select the best assistant to handle the entire conversation.
User's request: {initial_request}
Respond with only the name of the assistant you are choosing."""
).partial(agent_names=", ".join(agent_names))

supervisor_chain = supervisor_prompt | llm | (lambda x: x.content)

def get_initial_agent(state: AgentState):
    """Calls the supervisor to select the first agent."""
    print("---SUPERVISOR: Selecting initial agent---")
    initial_request = state["messages"][-1].content
    agent_name = supervisor_chain.invoke({"initial_request": initial_request})
    return {"agent_name": agent_name}

def router(state: AgentState) -> str:
    """
    This is the conditional logic that directs the conversation.
    It checks if an agent has been assigned. If not, it routes to the supervisor.
    Otherwise, it routes to the assigned agent.
    """
    if "agent_name" not in state or not state["agent_name"]:
        return "supervisor"
    else:
        return state["agent_name"]

def create_graph(checkpointer):
    """Factory function to create and compile the stateful graph."""
    workflow = StateGraph(AgentState)
    
    # 1. Add all the nodes to the graph
    workflow.add_node("supervisor", get_initial_agent)
    workflow.add_node("tools", tool_node)
    for agent_name in agent_names:
        workflow.add_node(agent_name, agent_node)

    # 2. Define the entry point
    workflow.set_conditional_entry_point(
        router,
        {
            "supervisor": "supervisor",
            **{name: name for name in agent_names},
        }
    )

    # 3. Define the edges
    
    # The supervisor's choice is routed to the selected agent
    workflow.add_conditional_edges(
        "supervisor",
        # The 'router' function is now used here to direct the flow
        lambda state: state["agent_name"],
        {name: name for name in agent_names}
    )
    
    # An agent can either call a tool or finish
    for agent_name in agent_names:
        workflow.add_conditional_edges(
            agent_name,
            tools_condition,
            {END: END, "tools": "tools"}
        )

    # After a tool is called, control returns to the agent that called it
    # We use the router logic to determine which agent that was.
    workflow.add_conditional_edges(
        "tools",
        router,
        {name: name for name in agent_names}
    )

    return workflow.compile(checkpointer=checkpointer)