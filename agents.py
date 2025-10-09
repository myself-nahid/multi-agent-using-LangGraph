from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import booking_tools, email_agent_tools

def create_agent(llm: ChatGoogleGenerativeAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True).with_config({"run_name": "agent"})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)
    
system_prompt_suffix = """

Your workflow for finding information must be as follows:
1.  **Check Internal Cache First:** Always start by using the `get_available_offers` tool to see if there is pre-fetched information for the user's request.
2.  **Handle Cache Miss:** If the `get_available_offers` tool returns a message like "No specific offers found", you MUST NOT give this response to the user. Instead, you must immediately use the `web_search_tool` to perform a live search for the user's request.
3.  **Synthesize and Respond:** After getting the results from the `web_search_tool`, synthesize the information and provide a helpful, complete answer to the user.
4.  You must include the source URL of any information you provide.
"""

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
    "EmailAutomation": create_agent(llm, email_agent_tools,
        "You are an Email Automation assistant. You can summarize and draft emails.")
}