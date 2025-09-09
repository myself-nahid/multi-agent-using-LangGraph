import os
import time
import json
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

TAVILY_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

print("ENV CHECK: TAVILY_KEY present?", bool(TAVILY_KEY))
print("ENV CHECK: GOOGLE_KEY present?", bool(GOOGLE_KEY))

if not TAVILY_KEY or not GOOGLE_KEY:
    raise RuntimeError("Please set TAVILY_API_KEY and GOOGLE_API_KEY in .env")

tavily = TavilyClient(api_key=TAVILY_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

CACHE_FILE = "offers_cache.json"
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))

app = FastAPI(title="Offer Library Service (Gemini + Tavily)")

_offers: List[Dict[str, Any]] = []

def load_cache():
    global _offers
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _offers = json.load(f)
        except Exception as e:
            print("Could not load cache:", e)
    else:
        _offers = []

def save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_offers, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Could not save cache:", e)

async def summarize_with_gemini_async(text: str) -> str:
    if not text: return ""
    prompt = f"Summarize the following offer/description in one concise sentence suitable for a listing:\n\n{text}"
    messages_batch = [[HumanMessage(content=prompt)]]
    try:
        resp = await llm.agenerate(messages=messages_batch)
        content = getattr(resp.generations[0][0].message, "content", None)
        return content.strip() if isinstance(content, str) else str(resp.generations[0][0].message)
    except Exception as e:
        print(f"Gemini async summarization failed: {e}")
        return (text[:140] + "...") if len(text) > 140 else text

async def fetch_offers_for_async(agent_name: str, search_term: str, location: str, max_results: int = 4) -> List[Dict[str, Any]]:
    q = f'"{search_term}" in {location}'
    try:
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, 
            lambda: tavily.search(q, include_raw_content=True, max_results=max_results, include_images=True)
        )
        
        print(f"\n--- RAW TAVILY RESPONSE for query: '{q}' ---")
        print(json.dumps(resp, indent=2))
        print("--- END RAW TAVILY RESPONSE ---\n")

        results = resp.get("results", [])
    except Exception as e:
        print(f"Tavily search error: {e}")
        results = []
    
    for item in results:
        item['category'] = agent_name
        item['original_location'] = location
        
    return results

async def update_loop():
    load_cache()
    queries = [
        ("HotelReservation", "hotel deals", "Paris"),
        ("HotelReservation", "luxury hotels", "Dubai"),
        ("RestaurantBooking", "michelin star restaurant offers", "Tokyo"),
        ("RestaurantBooking", "fine dining restaurants", "New York"),
        ("SpaBooking", "spa and wellness packages", "Bali"),
        ("ConcertTicketsBooking", "concert tickets", "London"),
        ("ConcertTicketsBooking", "live music events", "Los Angeles"),
        ("BirthdayBooking", "birthday party venues", "Sydney"),
        ("BirthdayBooking", "weekend getaway deals", "from Berlin"),
        ("FlightBooking", "cheap flight deals", "from Singapore to Bangkok")
    ]
    
    while True:
        print("\n--- Starting concurrent offer fetch cycle ---")
        start_time = time.time()

        fetch_tasks = [fetch_offers_for_async(agent, term, loc) for agent, term, loc in queries]
        list_of_results_lists = await asyncio.gather(*fetch_tasks)
        
        all_raw_results = [item for sublist in list_of_results_lists for item in sublist]
        print(f"Fetched {len(all_raw_results)} raw results from Tavily.")

        summary_tasks = [summarize_with_gemini_async(item.get("content", "")) for item in all_raw_results]
        summaries = await asyncio.gather(*summary_tasks)
        print(f"Generated {len(summaries)} summaries from Gemini.")

        new_offers = []
        for i, item in enumerate(all_raw_results):
            if item.get("url") and not any(existing["id"] == item["url"] for existing in new_offers):
                images = item.get("images", [])
                image_url = images[0] if images else None

                new_offers.append({
                    "id": item.get("url"),
                    "title": item.get("title", ""),
                    "summary": summaries[i],
                    "image_url": image_url,
                    "category": item['category'],
                    "location": item['original_location']
                })

        if new_offers:
            global _offers
            _offers = new_offers
            save_cache()
            print(f"--- Cycle complete. Updated cache with {len(_offers)} new offers. ---")
        
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