import os
import time
import json
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from tavily import TavilyClient
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()

TAVILY_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

print("ENV CHECK: TAVILY_API_KEY present?", bool(TAVILY_KEY))
print("ENV CHECK: GOOGLE_API_KEY present?", bool(GOOGLE_KEY))

if not TAVILY_KEY or not GOOGLE_KEY:
    raise RuntimeError("Please set TAVILY_API_KEY and GOOGLE_API_KEY in .env")

class Price(BaseModel):
    original_price: Optional[float] = Field(description="The numerical value of the standard/list/original price. Null if not found.")
    offer_price: Optional[float] = Field(description="The numerical value of the sale/discounted/'starting from' price. Null if not found.")
    currency: Optional[str] = Field(description="The 3-letter ISO currency code for the prices found (e.g., 'USD', 'EUR', 'SAR').")

summarize_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)
structured_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0).with_structured_output(Price)

CACHE_FILE = "offers_cache.json"
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "3600"))
app = FastAPI(title="Offer Library Service (Gemini + Tavily)")
_offers: List[Dict[str, Any]] = []

def load_cache():
    global _offers
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f: _offers = json.load(f)
        except Exception as e: print(f"Could not load cache: {e}")

def save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f: json.dump(_offers, f, ensure_ascii=False, indent=2)
    except Exception as e: print(f"Could not save cache: {e}")

async def summarize_with_gemini_async(text: str) -> str:
    if not text: return ""
    prompt = f"Summarize the following in one concise sentence for a listing:\n\n{text}"
    try:
        resp = await summarize_llm.ainvoke(prompt)
        return resp.content.strip()
    except Exception as e:
        print(f"Gemini async summarization failed: {e}")
        return ""

async def extract_prices_with_gemini_async(text: str, retries: int = 2) -> dict:
    default_price = {"original_price": None, "offer_price": None, "currency": None}
    if not text: return default_price
    prompt = f"""
    You are an expert data extraction assistant. Analyze the text to identify price information.
    RULES:
    1. Find 'offer_price' (e.g., "from $99", "sale €50"). This is most important.
    2. Find 'original_price' if it is also mentioned (e.g., a crossed-out list price).
    3. Extract only numerical values (e.g., 99.0, 50).
    4. You MUST identify the 3-letter ISO currency code (e.g., 'USD', 'EUR', 'SAR'). If you find a price but cannot determine the currency, all fields must be null.
    5. Ignore non-monetary values like "points". If a value is not found, it must be null.
    Text to analyze: --- {text} ---
    """
    for attempt in range(retries):
        try:
            price_model = await structured_llm.ainvoke(prompt)
            if price_model: return price_model.dict()
        except Exception as e:
            print(f"Gemini price extraction attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1: await asyncio.sleep(1)
    return default_price

async def fetch_offers_for_async(agent_name: str, search_term: str, location: str, max_results: int = 2) -> List[Dict[str, Any]]: # Reduced to 2 for more diversity
    q = f'"{search_term}" in {location}'
    try:
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, 
            lambda: tavily.search(q, search_depth="advanced", include_raw_content=True, max_results=max_results, include_images=True)
        )
        results = resp.get("results", [])
        top_level_images = resp.get("images", [])
        if results and top_level_images:
            for i in range(min(len(results), len(top_level_images))):
                results[i]['image_url'] = top_level_images[i]
    except Exception as e:
        print(f"Tavily search error: {e}")
        results = []
    for item in results:
        item['category'] = agent_name
        item['original_location'] = location
    return results

async def update_loop():
    load_cache()
    # --- NEW, EXPANDED, AND WORLDWIDE QUERIES LIST ---
    queries = [
        # == HotelReservation ==
        # Middle East
        ("HotelReservation", "luxury 5-star hotels", "Riyadh"),
        ("HotelReservation", "Jeddah corniche hotels with sea view", "Jeddah"),
        ("HotelReservation", "AlUla desert resorts deals", "AlUla, Saudi Arabia"),
        ("HotelReservation", "luxury hotels near Burj Khalifa", "Dubai"),
        # Europe
        ("HotelReservation", "luxury hotels with Eiffel Tower view", "Paris"),
        ("HotelReservation", "boutique hotels in central", "London"),
        ("HotelReservation", "hotels near the Colosseum", "Rome"),
        ("HotelReservation", "canal view hotels", "Amsterdam"),
        # North America
        ("HotelReservation", "5-star hotels in Times Square", "New York"),
        ("HotelReservation", "luxury hotels on the Strip", "Las Vegas"),
        ("HotelReservation", "all-inclusive beach resorts", "Cancun, Mexico"),
        # Asia & Oceania
        ("HotelReservation", "budget hotel deals near Shibuya Crossing", "Tokyo"),
        ("HotelReservation", "hotels with rooftop pool", "Singapore"),
        ("HotelReservation", "beach villas", "Maldives"),
        ("HotelReservation", "hotels with Sydney Opera House view", "Sydney"),

        # == RestaurantBooking ==
        # Middle East & Europe
        ("RestaurantBooking", "fine dining restaurants", "Riyadh"),
        ("RestaurantBooking", "celebrity chef restaurants", "Dubai"),
        ("RestaurantBooking", "michelin star restaurants", "London"),
        ("RestaurantBooking", "best pasta restaurants", "Rome, Italy"),
        # North America & Asia
        ("RestaurantBooking", "omakase sushi experience", "Tokyo"),
        ("RestaurantBooking", "rooftop restaurants with city view", "Bangkok"),
        ("RestaurantBooking", "best steakhouse", "New York"),
        
        # == SpaBooking ==
        ("SpaBooking", "luxury spa packages for women", "Riyadh"),
        ("SpaBooking", "luxury spa and wellness retreats", "Bali, Indonesia"),
        ("SpaBooking", "day spa packages", "New York"),
        ("SpaBooking", "thermal baths and spa", "Budapest, Hungary"),

        # == ConcertTicketsBooking ==
        ("ConcertTicketsBooking", "Riyadh Season event tickets", "Riyadh"),
        ("ConcertTicketsBooking", "upcoming concerts and music festivals", "Los Angeles"),
        ("ConcertTicketsBooking", "tickets for broadway shows", "New York"),
        ("ConcertTicketsBooking", "concerts at the O2 Arena", "London"),
        ("ConcertTicketsBooking", "K-pop concerts", "Seoul"),

        # == BirthdayBooking ==
        ("BirthdayBooking", "private yacht party rental", "Miami"),
        ("BirthdayBooking", "rooftop birthday party venue", "Sydney"),
        ("BirthdayBooking", "desert safari private dinner", "Dubai"),
        ("BirthdayBooking", "castle rental for events", "Scotland, UK"),

        # == FlightBooking ==
        ("FlightBooking", "Saudia business class deals from Riyadh to London", "Saudi Arabia"),
        ("FlightBooking", "Emirates first class deals from Dubai to New York", "UAE"),
        ("FlightBooking", "British Airways cheap flights from London to New York", "UK"),
        ("FlightBooking", "Qantas flights from Sydney to Los Angeles", "Australia"),
        ("FlightBooking", "Singapore Airlines suites from Singapore to Tokyo", "Singapore")
    ]
    
    while True:
        print("\n--- Starting concurrent offer fetch cycle ---")
        start_time = time.time()

        fetch_tasks = [fetch_offers_for_async(agent, term, loc) for agent, term, loc in queries]
        list_of_results_lists = await asyncio.gather(*fetch_tasks)
        all_raw_results = [item for sublist in list_of_results_lists for item in sublist]
        print(f"Fetched {len(all_raw_results)} raw results from Tavily.")

        if not all_raw_results:
            print("--- No raw results found. Ending cycle early. ---")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            continue

        summary_tasks = [summarize_with_gemini_async(item.get("content", "")) for item in all_raw_results]
        price_tasks = [extract_prices_with_gemini_async(item.get("raw_content", item.get("content", ""))) for item in all_raw_results]

        all_gathered_results = await asyncio.gather(*summary_tasks, *price_tasks)
        num_results = len(all_raw_results)
        summaries = all_gathered_results[:num_results]
        prices = all_gathered_results[num_results:]
        
        print(f"Generated {len(summaries)} summaries from Gemini.")
        print(f"Extracted {len(prices)} price data points from Gemini.")

        new_offers = []
        for i, item in enumerate(all_raw_results):
            if item.get("url") and not any(existing["id"] == item["url"] for existing in new_offers):
                image_url = item.get("image_url", None)
                price_data = prices[i]

                original_price = price_data.get("original_price")
                offer_price = price_data.get("offer_price")
                currency = price_data.get("currency")

                if (not isinstance(offer_price, (int, float)) or offer_price <= 0 or 
                    not isinstance(currency, str) or len(currency) != 3):
                    original_price, offer_price, currency = None, None, None
                
                if offer_price and offer_price > 1_000_000:
                    original_price, offer_price, currency = None, None, None

                if not isinstance(original_price, (int, float)) or original_price <= 0:
                    original_price = None

                if original_price and offer_price and original_price < offer_price:
                    original_price = None
                
                if offer_price is not None:
                    new_offers.append({
                        "id": item.get("url"), "title": item.get("title", ""), "summary": summaries[i],
                        "url": item.get("url"), "image_url": image_url, "category": item['category'],
                        "location": item['original_location'], 
                        "price": original_price,
                        "offer_price": offer_price,
                        "currency": currency, 
                        "source": "tavily", "fetched_at": int(time.time()),
                    })

        if new_offers:
            global _offers
            _offers = new_offers
            save_cache()
            print(f"--- Cycle complete. Filtered down to and saved {len(_offers)} high-quality offers. ---")
        else:
            print("--- Cycle complete. No new offers with valid prices were found. ---")
        
        end_time = time.time()
        print(f"--- Offer fetch cycle took {end_time - start_time:.2f} seconds. ---")
        
        await asyncio.sleep(POLL_INTERVAL_SECONDS)

@app.on_event("startup")
async def startup_event():
    load_cache()
    loop = asyncio.get_event_loop()
    loop.create_task(update_loop())
    print("Offer service background task started. Initial offers loaded:", len(_offers))

@app.get("/offers")
async def get_offers():
    return JSONResponse(content={"offers": _offers})

@app.get("/health")
async def health():
    return {"ok": True, "offers": len(_offers)}