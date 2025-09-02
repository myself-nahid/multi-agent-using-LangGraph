from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import database      
import vectorstore   

web_search_tool = TavilySearch(
    max_results=4, 
    description="A web search tool to find real-time information on flights, hotels, restaurants, events, and more."
)

@tool
def search_user_emails(query: str) -> str:
    """
    Searches a user's emails based on a semantic query to find relevant information.
    Use this for questions like 'what did my manager say about the project launch?'.
    """
    print(f"--- Searching emails semantically with query: '{query}' ---")
    return vectorstore.search_emails(query)

@tool
def update_task_status(session_id: str, status: str, details: dict) -> str:
    """
    Updates the status of the current task or booking in the workflow system.
    Use this tool when a booking is confirmed, cancelled, or reaches a milestone.
    Valid status options are 'Processing', 'Confirm', 'Cancel', 'Complete'.
    The 'details' arg should be a dictionary with relevant info like booking dates or guest count.
    """
    database.update_workflow(session_id, status, details)
    return f"Status for session {session_id} updated successfully to {status}."

booking_tools = [web_search_tool, update_task_status]

email_agent_tools = [search_user_emails, update_task_status]