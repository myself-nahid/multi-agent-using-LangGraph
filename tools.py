import json
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import database
import vectorstore
from offer_service import _offers

web_search_tool = TavilySearch(
    max_results=4, 
    description="A web search tool to find real-time information on flights, hotels, restaurants, events, and more."
)

@tool
def get_available_offers(category: str, location: str) -> str:
    """
    Looks up available deals and offers for a given category (e.g., 'hotel', 'flight', 'concert') and location.
    """
    print(f"--- Searching for offers with category '{category}' in location '{location}' ---")
    
    category_map = {
        "flight": "FlightBooking", "restaurant": "RestaurantBooking", "spa": "SpaBooking",
        "birthday": "BirthdayBooking", "concert": "ConcertTicketsBooking", "event": "ConcertTicketsBooking",
        "hotel": "HotelReservation", "reservation": "HotelReservation"
    }
    
    agent_category = None
    for key, value in category_map.items():
        if key in category.lower():
            agent_category = value
            break
            
    if not agent_category:
        return f"No specific offer category found for '{category}'. Please try a keyword like 'hotel', 'restaurant', or 'concert'."

    print(f"--- Mapped category '{category}' to agent category '{agent_category}' ---")
    try:
        all_offers = _offers
        filtered_offers = [
            offer for offer in all_offers 
            if offer.get("category") == agent_category and 
               location.lower() in offer.get("location", "").lower()
        ]
        if not filtered_offers:
            return f"No specific offers found for {agent_category} in {location}. You can use the web_search_tool for a general search."
        return json.dumps(filtered_offers, indent=2)
    except Exception as e:
        print(f"An unexpected error occurred while accessing offers: {e}")
        return "An unexpected error occurred while fetching offers."

@tool
def search_user_emails(query: str) -> str:
    """Searches a user's emails based on a semantic query to find relevant information."""
    return vectorstore.search_emails(query)

@tool
def update_task_status(session_id: str, status: str, details: dict) -> str:
    """Updates the status of the current task or booking in the workflow system."""
    database.update_workflow(session_id, status, details)
    return f"Status for session {session_id} updated successfully to {status}."

booking_tools = [web_search_tool, update_task_status, get_available_offers]
email_agent_tools = [search_user_emails, update_task_status]

all_tools = booking_tools + email_agent_tools