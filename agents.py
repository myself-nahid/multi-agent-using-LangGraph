from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import booking_tools, email_agent_tools

def create_agent(llm: ChatGoogleGenerativeAI, tools: list, system_prompt: str):
    """Factory to create a new agent runnable."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True).with_config({"run_name": "agent"})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    
system_prompt_suffix = """

Your workflow for finding information is as follows:
1. First, use the 'get_available_offers' tool to check the internal cache for relevant deals.
2. If the cached offers have an 'image_url' that is null or empty, and the offer looks promising, you MUST then use the 'scrape_page_for_images' tool with the offer's 'url' to find a list of high-quality images.
3. Select the best image from the list returned by the scraper.
4. Present the final result to the user, including the title, summary, the URL of the offer, and the high-quality image URL you found.
You must include the source URL of any information you provide."""

AGENT_RUNNABLES = {
    "FlightBooking": create_agent(llm, booking_tools,
        "You are a specialized Flight Booking assistant." + system_prompt_suffix),
    "RestaurantBooking": create_agent(llm, booking_tools,
        "You are a specialized Restaurant Booking assistant." + system_prompt_suffix),
    "SpaBooking": create_agent(llm, booking_tools,
        "You are a specialized Spa Booking assistant." + system_prompt_suffix),
    "BirthdayBooking": create_agent(llm, booking_tools,
        "You are a Birthday Planning assistant." + system_prompt_suffix),
    "ConcertTicketsBooking": create_agent(llm, booking_tools,
        "You are a Concert Tickets assistant." + system_prompt_suffix),
    "HotelReservation": create_agent(llm, booking_tools,
        "You are a Hotel Reservation assistant." + system_prompt_suffix),
    "EmailAutomation": create_agent(llm, email_agent_tools,
        "You are an Email Automation assistant.")
}