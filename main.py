import os
import json
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Tuple
import instructor
from openai import OpenAI
from dotenv import load_dotenv
import random

load_dotenv()
# --- CONFIGURATION ---
api_key = os.getenv("OPENAI_API_KEY")

client = instructor.from_openai(
    OpenAI(api_key=api_key),
    mode=instructor.Mode.JSON,
)

app = FastAPI()

# --- 1. LOAD DATA ---
def load_knowledge_base():
    try:
        with open("data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

KNOWLEDGE_BASE = load_knowledge_base()

# --- 2. ROBUST SEARCH ENGINE ---
def search_database(query: str, tastes: str):
    keywords = query.lower().split() + tastes.lower().replace(",", "").split()
    stop_words = ["the", "and", "a", "for", "is", "of", "in", "to", "my", "i", "want", "get", "budget"]
    keywords = [w for w in keywords if w not in stop_words]

    results = []

    # Helper to calculate score and add item
    def add_if_match(item, service, store):
        name = str(item.get("name", "")).lower()
        desc = str(item.get("description", "")).lower()
        score = 0
        
        # Keyword Match
        for k in keywords:
            if k in name or k in desc:
                score += 1
        
        # CRITICAL: Force Boost the Boat for Birthday Queries
        if "birthday" in query.lower() and "fällä" in name:
            score += 1000  # Impossible to ignore
        
        if score > 0:
            results.append({
                "service": service,
                "store": store,
                "item_name": item.get("name"),
                "price": item.get("price", "N/A"),
                "score": score
            })

    # Scan Database
    for entry in KNOWLEDGE_BASE:
        service_name = entry.get("service", "General")
        root_data = entry.get("data", {})

        # CASE 1: The 'data' is the item itself (e.g. The Boat Event)
        if isinstance(root_data, dict) and "name" in root_data:
            add_if_match(root_data, service_name, "N/A")

        # CASE 2: The 'data' is a Store with Categories (e.g. Music Store)
        elif isinstance(root_data, dict):
            for category, items in root_data.items():
                if isinstance(items, list):
                    for item in items:
                        add_if_match(item, service_name, category)
        
        # CASE 3: The 'data' is a List of items
        elif isinstance(root_data, list):
            for item in root_data:
                add_if_match(item, service_name, "N/A")

    # Sort and return Top 10
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10]

# --- 3. PERSONA ---
USER_PROFILE = {
    "name": "Dika",
    "orders_frequency": "Erratic, always trying new things, daring, healthy",
    "tastes": "Likes Chocolate and Vanilla, Likes Music, Likes Guitars, Does not like blueberries",
    "habits": "Orders tamwin every friday, and also asks for his thobes to be taken for laundry every friday as well"
}

# --- 4. STRUCTURE ---
class AppResponse(BaseModel):
    display_tags: List[List[str]] = Field(
        ..., 
        description="List of lists: [['Service', 'Store', 'Item Name']]. MUST MATCH TEXT."
    )
    assistant_message: str = Field(..., description="Helpful response.")

class UserQuery(BaseModel):
    query: str
    current_date: str = "2026-01-16"

# --- 5. ENDPOINT ---
@app.post("/chat", response_model=AppResponse)
def chat(user_input: UserQuery):
    
    # 1. Search with Fixed Logic
    relevant_items = search_database(user_input.query, USER_PROFILE["tastes"])
    
    random_coins = random.randint(10, 50) 
    
    response = client.chat.completions.create(
        model="gpt-4o",
        response_model=AppResponse,
        messages=[
            {
                "role": "system", 
                "content": f"""
                You are a Lifestyle Consultant for {USER_PROFILE['name']}.
                
                MATCHING ITEMS (From Database):
                {json.dumps(relevant_items)}
                
                --- INSTRUCTIONS ---
                
                **SCENARIO A: Birthday Plan** (Query: "Birthday", "Jan 23")
                   1. REQUIRED ITEMS:
                      - The "Fällä" Boat Event.
                      - One Guitar (Gift).
                      - One Cake (No Blueberries).
                   2. MESSAGE: Pitch the plan + "I moved Tamwin & Laundry to Saturday."
                
                **SCENARIO B: Normal Search**
                   1. Select best matches.
                   2. MESSAGE: "Buy the first item to earn {random_coins} Snoonu Coins!"
                
                **CONSISTENCY RULE:**
                Every item mentioned in your `assistant_message` MUST be included in `display_tags`.
                Do not talk about the Boat if you do not output the Boat tag.
                
                **TAG FORMAT:** List of lists `[["Service", "Store", "ItemName"]]` using EXACT strings from the MATCHING ITEMS list.
                """
            },
            {"role": "user", "content": f"Today is {user_input.current_date}. {user_input.query}"},
        ],
    )
    return response