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

# --- 2. RELEVANCE SEARCH ENGINE (Pure Scoring) ---
def search_database(query: str, tastes: str):
    """
    Scans everything, scores it based on keyword matches, 
    and returns the Top 15 items regardless of category.
    """
    # 1. Keywords
    # We combine the User Query + User Tastes into one big search bag
    keywords = query.lower().split() + tastes.lower().replace(",", "").split()
    
    # Filter out useless words to improve accuracy
    stop_words = ["the", "and", "a", "for", "is", "of", "in", "to", "my", "i", "want", "get", "budget", "nice", "gift", "something", "unsure", "or"]
    keywords = [w for w in keywords if w not in stop_words]

    results = []

    def add_match(item, service, store):
        name = str(item.get("name", "")).lower()
        desc = str(item.get("description", "")).lower()
        
        # Calculate Match Score
        score = 0
        for k in keywords:
            # Full word match gets higher score, partial gets lower
            if k in name: score += 3
            if k in desc: score += 1
        
        # Super-Boost for the specific Birthday Event if asked
        if "birthday" in query.lower() and "fällä" in name:
            score += 1000

        if score > 0:
            results.append({
                "service": service,
                "store": store,
                "item_name": item.get("name"),
                "price": item.get("price", "N/A"),
                "description": item.get("description", "")[:100], # Snippet for context
                "score": score
            })

    # 2. Scan The Database
    for entry in KNOWLEDGE_BASE:
        service_name = entry.get("service", "General")
        root_data = entry.get("data", {})

        # Check: Is data a specific Item?
        if isinstance(root_data, dict) and "name" in root_data:
            add_match(root_data, service_name, "N/A")

        # Check: Is data a Store with Categories?
        elif isinstance(root_data, dict):
            for category, items in root_data.items():
                if isinstance(items, list):
                    for item in items:
                        add_match(item, service_name, category)
        
        # Check: Is data a List of Items?
        elif isinstance(root_data, list):
            for item in root_data:
                add_match(item, service_name, "N/A")

    # 3. Sort by Score (Highest First) & Limit
    # We don't care about diversity here. If the top 10 are all food, so be it.
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:15]

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
        description="List of lists: [['Service', 'Store', 'Item Name']]."
    )
    assistant_message: str = Field(..., description="Helpful response.")

class UserQuery(BaseModel):
    query: str
    current_date: str = "2026-01-16"

# --- 5. ENDPOINT ---
@app.post("/chat", response_model=AppResponse)
def chat(user_input: UserQuery):
    
    # 1. Get Top 15 Matches (Pure Relevance)
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
                
                MATCHING ITEMS (Sorted by Relevance):
                {json.dumps(relevant_items)}
                
                --- INSTRUCTIONS ---
                
                **SCENARIO A: Birthday Plan** ("Jan 23", "Friend's birthday")
                   1. REQUIRED: "Fällä" Boat + 1 Guitar + 1 Cake.
                   2. MESSAGE: Pitch plan + "Moved Tamwin & Laundry to Saturday."
                
                **SCENARIO B: Standard Search**
                   (e.g., "Find me a nice gift", "I want food", "Guitars")
                   1. Review the 'MATCHING ITEMS'.
                   2. **SELECTION:** Select the items that truly fit the user's intent. 
                      - If the user asks for "Food or Gifts", showing 3-4 varied items is great.
                      - If the user asks for "Amps", showing 3 Amps is correct.
                      - **Do not limit yourself to 1 item.**
                   3. **PROMO:** "Buy the first item to earn {random_coins} Snoonu Coins!"
                
                **TAG FORMAT:** List of lists `[["Service", "Store", "ItemName"]]`.
                """
            },
            {"role": "user", "content": f"Today is {user_input.current_date}. {user_input.query}"},
        ],
    )
    return response