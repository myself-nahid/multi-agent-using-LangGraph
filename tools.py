# FILE: tools.py

import json
from datetime import datetime
from typing import Optional
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
import database
import vectorstore
from offer_service import _offers

# --- Tool Initialization ---
tavily_search = TavilySearch(
    max_results=4, 
    description="A general web search tool for finding real-time information when other tools don't have the answer."
)

# --- Pydantic Schemas for Action Tools ---
class BookFlightArgs(BaseModel):
    session_id: str = Field(description="The unique session ID for the current conversation.")
    flight_number: str = Field(description="The flight number of the flight to be booked, e.g., 'BA2490'.")
    passenger_name: str = Field(description="The full name of the passenger.")
    date: str = Field(description="The date of the flight in 'YYYY-MM-DD' format.")

class BookHotelArgs(BaseModel):
    session_id: str = Field(description="The unique session ID for the current conversation.")
    hotel_name: str = Field(description="The name of the hotel to book.")
    check_in_date: str = Field(description="The check-in date in 'YYYY-MM-DD' format.")
    num_nights: int = Field(description="The number of nights for the stay.")
    num_guests: int = Field(description="The number of guests.")

class BookRestaurantArgs(BaseModel):
    session_id: str = Field(description="The unique session ID for the current conversation.")
    restaurant_name: str = Field(description="The name of the restaurant for the reservation.")
    date: str = Field(description="The date of the reservation in 'YYYY-MM-DD' format.")
    time: str = Field(description="The time of the reservation, e.g., '7:30 PM'.")
    num_guests: int = Field(description="The number of guests for the reservation.")

class BookSpaArgs(BaseModel):
    session_id: str = Field(description="The unique session ID for the current conversation.")
    spa_name: str = Field(description="The name of the spa.")
    service_type: str = Field(description="The specific spa service being booked, e.g., 'Deep Tissue Massage'.")
    date: str = Field(description="The date of the appointment in 'YYYY-MM-DD' format.")
    time: str = Field(description="The time of the appointment, e.g., '3:00 PM'.")

class BookConcertArgs(BaseModel):
    session_id: str = Field(description="The unique session ID for the current conversation.")
    event_name: str = Field(description="The name of the concert or event.")
    artist_name: str = Field(description="The name of the artist performing.")
    num_tickets: int = Field(description="The number of tickets to purchase.")

# --- NEW: Schema for the Birthday Venue Tool ---
class BookBirthdayVenueArgs(BaseModel):
    session_id: str = Field(description="The unique session ID for the current conversation.")
    venue_name: str = Field(description="The name of the venue to book for the birthday.")
    date: str = Field(description="The date of the event in 'YYYY-MM-DD' format.")
    num_guests: int = Field(description="The number of guests attending.")
    special_requests: Optional[str] = Field(description="Any special requests for the booking, e.g., 'cake needed'.")

# --- Action Tools ---

@tool(args_schema=BookFlightArgs)
def book_flight(session_id: str, flight_number: str, passenger_name: str, date: str) -> str:
    """Books a flight and completes the workflow."""
    print(f"--- SIMULATING FLIGHT BOOKING for session {session_id} ---")
    booking_details = {"flight_number": flight_number, "passenger_name": passenger_name, "date": date, "status": "Confirmed"}
    database.update_workflow(session_id, "Complete", booking_details)
    return f"Booking confirmed! Flight {flight_number} is booked for {passenger_name} on {date}."

@tool(args_schema=BookHotelArgs)
def book_hotel(session_id: str, hotel_name: str, check_in_date: str, num_nights: int, num_guests: int) -> str:
    """Books a hotel and completes the workflow."""
    print(f"--- SIMULATING HOTEL BOOKING for session {session_id} ---")
    booking_details = {"hotel_name": hotel_name, "check_in": check_in_date, "nights": num_nights, "guests": num_guests, "status": "Confirmed"}
    database.update_workflow(session_id, "Complete", booking_details)
    return f"Booking confirmed! A room at {hotel_name} for {num_guests} guests, checking in on {check_in_date} for {num_nights} nights is booked."

@tool(args_schema=BookRestaurantArgs)
def book_restaurant(session_id: str, restaurant_name: str, date: str, time: str, num_guests: int) -> str:
    """Books a restaurant reservation and completes the workflow."""
    print(f"--- SIMULATING RESTAURANT BOOKING for session {session_id} ---")
    booking_details = {"restaurant": restaurant_name, "date": date, "time": time, "guests": num_guests, "status": "Confirmed"}
    database.update_workflow(session_id, "Complete", booking_details)
    return f"Booking confirmed! A table for {num_guests} at {restaurant_name} on {date} at {time} is reserved."

@tool(args_schema=BookSpaArgs)
def book_spa_appointment(session_id: str, spa_name: str, service_type: str, date: str, time: str) -> str:
    """Books a spa appointment and completes the workflow."""
    print(f"--- SIMULATING SPA BOOKING for session {session_id} ---")
    booking_details = {"spa": spa_name, "service": service_type, "date": date, "time": time, "status": "Confirmed"}
    database.update_workflow(session_id, "Complete", booking_details)
    return f"Booking confirmed! A {service_type} appointment at {spa_name} on {date} at {time} is booked."

@tool(args_schema=BookConcertArgs)
def book_concert_tickets(session_id: str, event_name: str, artist_name: str, num_tickets: int) -> str:
    """Books concert tickets and completes the workflow."""
    print(f"--- SIMULATING TICKET BOOKING for session {session_id} ---")
    booking_details = {"event": event_name, "artist": artist_name, "tickets": num_tickets, "status": "Confirmed"}
    database.update_workflow(session_id, "Complete", booking_details)
    return f"Booking confirmed! {num_tickets} tickets for {artist_name} at {event_name} have been purchased."

# --- NEW: The 'book_birthday_venue' tool ---
@tool(args_schema=BookBirthdayVenueArgs)
def book_birthday_venue(session_id: str, venue_name: str, date: str, num_guests: int, special_requests: str = "None") -> str:
    """Books a venue for a birthday party and completes the workflow."""
    print(f"--- SIMULATING BIRTHDAY VENUE BOOKING for session {session_id} ---")
    booking_details = {"venue": venue_name, "date": date, "guests": num_guests, "requests": special_requests, "status": "Confirmed"}
    database.update_workflow(session_id, "Complete", booking_details)
    return f"Booking confirmed! The venue '{venue_name}' is booked for {num_guests} guests on {date}."

# --- Information Gathering Tools ---
@tool
def get_available_offers(category: str, location: str) -> str:
    """Looks up available deals from the pre-fetched internal cache. This should be the first tool you use."""
    # ... (code is unchanged)

@tool
def search_user_emails(query: str) -> str:
    """Searches a user's emails based on a semantic query."""
    return vectorstore.search_emails(query)

@tool
def update_task_status(session_id: str, status: str, details: dict) -> str:
    """Updates the status of the current task or booking in the workflow system."""
    database.update_workflow(session_id, status, details)
    return f"Status for session {session_id} updated successfully to {status}."

# --- Tool Lists for Agents ---
booking_tools = [
    tavily_search, get_available_offers, update_task_status, 
    book_flight, book_hotel, book_restaurant, book_spa_appointment, book_concert_tickets,
    book_birthday_venue # Add the new tool to the list
]
email_agent_tools = [search_user_emails, update_task_status]
all_tools = booking_tools + email_agent_tools