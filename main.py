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

# --- 2. INTELLIGENT SEARCH ENGINE ---
def search_database(query: str, tastes: str):
    """
    Advanced filter that preserves hierarchy (Service -> Store -> Item).
    """
    # 1. Prepare Keywords
    keywords = query.lower().split() + tastes.lower().replace(",", "").split()
    stop_words = ["the", "and", "a", "for", "is", "of", "in", "to", "my", "i", "want", "get", "budget"]
    keywords = [w for w in keywords if w not in stop_words]

    results = []

    # 2. Deep Scan
    for entry in KNOWLEDGE_BASE:
        # Capture the Top-Level Service Name (e.g. "Flowers and Gifts", "S City")
        service_name = entry.get("service", "General Service")
        
        # The 'data' field can be a Dictionary (Store) or List/Object (Event)
        root_data = entry.get("data", {})

        # Helper to process a single item
        def process_item(item, store_name):
            name = str(item.get("name", "")).lower()
            desc = str(item.get("description", "")).lower()
            
            # Scoring
            score = 0
            for k in keywords:
                if k in name or k in desc:
                    score += 1
            
            # Boost for Birthday + Boat combo
            if "birthday" in query.lower() and "fällä" in name:
                score += 500
            
            if score > 0:
                results.append({
                    "service": service_name,
                    "store": store_name,  # Keeps exact store name
                    "item_name": item.get("name"), # Keeps exact item name
                    "price": item.get("price", "N/A"),
                    "score": score
                })

        # Scenario A: 'data' is a Dictionary (Stores with Categories)
        if isinstance(root_data, dict):
            # Check if it has categories inside (e.g. "Cakes": [...])
            # We assume the keys of 'data' are the Store Names or Categories depending on your structure
            # If your structure is Service -> Store -> Items, we iterate differently.
            # ADJUSTMENT: Based on your previous logs, 'data' seems to contain categories directly.
            # Let's assume the 'entry["service"]' IS the Store/Service wrapper.
            
            for category, items in root_data.items():
                if isinstance(items, list):
                    for item in items:
                        # If the category looks like a Store Name, use it. Otherwise use Service Name.
                        # For safety, we pass the Category as a "Subsection" or Store identifier if needed.
                        # If your JSON has "Virgin Megastore" as a key inside data, this captures it.
                        process_item(item, store_name=category)
                elif isinstance(items, dict):
                     # Nested single item
                     process_item(items, store_name=category)
                     
        # Scenario B: 'data' is a List (Direct list of items)
        elif isinstance(root_data, list):
            for item in root_data:
                process_item(item, store_name="N/A")

    # 3. Sort & Slice
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10] # Send Top 10 Smartest Matches to GPT-4o

# --- 3. PERSONA ---
USER_PROFILE = {
    "name": "Dika",
    "orders_frequency": "Erratic, always trying new things, daring, healthy",
    "tastes": "Likes Chocolate and Vanilla, Likes Music, Likes Guitars, Does not like blueberries",
    "habits": "Orders tamwin every friday, and also asks for his thobes to be taken for laundry every friday as well"
}

# --- 4. RESPONSE STRUCTURE ---
class AppResponse(BaseModel):
    display_tags: List[List[str]] = Field(
        ..., 
        description="List of lists. Format: [['Service Name', 'Store/Category Name', 'Exact Item Name']]"
    )
    
    assistant_message: str = Field(
        ..., 
        description="A helpful, conversational recommendation explaining the choices."
    )

class UserQuery(BaseModel):
    query: str
    current_date: str = "2026-01-16"

# --- 5. ENDPOINT ---
@app.post("/chat", response_model=AppResponse)
def chat(user_input: UserQuery):
    
    # 1. Search (Python Side)
    relevant_items = search_database(user_input.query, USER_PROFILE["tastes"])
    
    # 2. Promo Logic
    random_coins = random.randint(10, 50) # Increased coins to make it exciting
    
    # 3. GPT-4o Logic (The Brains)
    response = client.chat.completions.create(
        model="gpt-4o",  # <--- UPGRADED MODEL
        response_model=AppResponse,
        messages=[
            {
                "role": "system", 
                "content": f"""
                You are a high-end Lifestyle Consultant for {USER_PROFILE['name']}.
                
                USER PROFILE:
                {json.dumps(USER_PROFILE)}
                
                MATCHING ITEMS FOUND:
                {json.dumps(relevant_items)}
                
                --- INSTRUCTIONS ---
                
                **SCENARIO A: Birthday/Prototype Plan**
                (Query: "Friend's birthday", "Jan 23", "Friday plan")
                   1. MUST Select: "Fällä" Boat + 1 Guitar + 1 Cake.
                   2. TONE: Enthusiastic planner.
                   3. MUST SAY: "I've moved your Friday Tamwin & Laundry habits to Saturday."
                
                **SCENARIO B: Specific Item Search**
                (Query: "I want an amp", "Find me flowers")
                   1. Select the best matching items from the provided list.
                   2. TONE: Helpful shopping assistant. 
                   3. **PROMO:** You MUST mention: "By the way, if you snap up the first item on this list, you'll earn {random_coins} Snoonu Coins!"
                
                **DATA FORMATTING (CRITICAL):**
                - Output `display_tags` as a list of lists: `[["Service", "Store", "Item"]]`.
                - Use the EXACT strings from the 'MATCHING ITEMS FOUND' list. Do not invent names.
                """
            },
            {"role": "user", "content": f"Today is {user_input.current_date}. {user_input.query}"},
        ],
    )
    return response