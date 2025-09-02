from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import web_search_tool, update_task_status
from langchain_core.messages import AIMessage, HumanMessage

def create_agent(llm: ChatGoogleGenerativeAI, tools: list, system_prompt: str):
    """Factory to create a new agent runnable."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True).with_config({"run_name": "agent"})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
booking_tools = [web_search_tool, update_task_status]
system_prompt_suffix = "\n\nYou must include the source URL of any information you provide."

AGENT_RUNNABLES = {
    "FlightBooking": create_agent(llm, booking_tools,
        "You are a specialized Flight Booking assistant. Find flight options and help the user book. Once complete, update the task status." + system_prompt_suffix),
    "RestaurantBooking": create_agent(llm, booking_tools,
        "You are a specialized Restaurant Booking assistant. Find restaurants and table availability." + system_prompt_suffix),
    "SpaBooking": create_agent(llm, booking_tools,
        "You are a specialized Spa Booking assistant. Find spa services and appointments." + system_prompt_suffix),
    "BirthdayBooking": create_agent(llm, booking_tools,
        "You are a Birthday Planning assistant. Help find venues, gift ideas, and activities." + system_prompt_suffix),
    "ConcertTicketsBooking": create_agent(llm, booking_tools,
        "You are a Concert Tickets assistant. Find tickets for events and artists." + system_prompt_suffix),
    "HotelReservation": create_agent(llm, booking_tools,
        "You are a Hotel Reservation assistant. Help the user find and book hotels for specific dates and guest counts." + system_prompt_suffix),
    # Email Automation is more complex and would use different tools (like a ChromaDB tool)
    "EmailAutomation": create_agent(llm, [],
        "You are an Email Automation assistant. You can summarize and draft emails.")
}