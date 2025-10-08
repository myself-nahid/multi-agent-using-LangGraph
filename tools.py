import os
import json
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import database
import vectorstore
from offer_service import _offers
from playwright.async_api import async_playwright
import asyncio

web_search_tool = TavilySearch(max_results=4)

@tool
async def scrape_page_for_images(url: str) -> str:
    """
    Scrapes a single webpage to find and return a list of up to 10 high-quality image URLs.
    Use this specialized tool when a previous search returned a good result but with no image URL.
    This is a powerful but slow tool, so use it only when necessary.
    """
    print(f"--- Starting Playwright to scrape images from: {url} ---")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=15000)
            
            # Find all <img> tags, get their 'src' attribute, and filter for valid URLs
            image_urls = await page.eval_on_all("img", """(images) =>
                images.map(img => img.src).filter(src => src.startsWith('http'))
            """)
            
            await browser.close()
            
            if not image_urls:
                return "No usable images found on the page."
            
            # Return up to the first 10 images found
            unique_urls = list(dict.fromkeys(image_urls)) # Remove duplicates
            print(f"--- Found {len(unique_urls)} images from {url} ---")
            return json.dumps(unique_urls[:10])

    except Exception as e:
        print(f"--- Playwright scraping failed for {url}: {e} ---")
        return f"Error scraping page: {e}"

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

booking_tools = [web_search_tool, get_available_offers, update_task_status, scrape_page_for_images]
email_agent_tools = [search_user_emails, update_task_status]
all_tools = booking_tools + email_agent_tools