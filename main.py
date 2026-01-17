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

# --- 2. SEARCH ENGINE (Fixes the 383k Token Crash) ---
def search_database(query: str, tastes: str):
    """
    Scans the database locally to find matches BEFORE sending to AI.
    This reduces token usage from 380,000 -> 2,000.
    """
    # 1. Keywords to look for
    keywords = query.lower().split() + tastes.lower().replace(",", "").split()
    stop_words = ["the", "and", "a", "for", "is", "of", "in", "to", "my", "i", "want", "show", "me"]
    keywords = [w for w in keywords if w not in stop_words]

    matches = []

    # 2. Scan every item in the JSON
    for entry in KNOWLEDGE_BASE:
        # Get Service Name (e.g. "Music Store")
        service_name = entry.get("service", "General")
        store_name = entry.get("store", "N/A")
        
        # Dig into the data categories
        data_content = entry.get("data", {})
        
        # Handle "Event" structure (flat dict) vs "Store" structure (nested dict)
        # We normalize everything into a list of items to check
        items_to_check = []
        
        if isinstance(data_content, dict):
            # It's a store with categories (e.g. "Guitars": [...])
            for category, item_list in data_content.items():
                if isinstance(item_list, list):
                    for item in item_list:
                        item["_category"] = category
                        items_to_check.append(item)
                # Handle single event object case
                elif isinstance(item_list, (str, int, float)):
                    # This might be an event object directly in 'data'
                    items_to_check.append(data_content)
                    break 
        
        # 3. Check each item for keywords
        for item in items_to_check:
            # Safely get string fields
            i_name = str(item.get("name", "")).lower()
            i_desc = str(item.get("description", "")).lower()
            i_cat = str(item.get("_category", "")).lower()
            
            # Simple Scoring
            score = 0
            for k in keywords:
                if k in i_name or k in i_desc or k in i_cat:
                    score += 1
            
            # Special Rule: "Birthday" queries always grab the Boat Event
            if "birthday" in query.lower() and "fällä" in i_name:
                score += 100
                
            if score > 0:
                # Add to results with metadata
                matches.append({
                    "service": service_name,
                    "store": store_name,
                    "item_data": item,
                    "score": score
                })

    # 4. Sort by score and take Top 15
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:15]

# --- 3. PERSONA ---
USER_PROFILE = {
    "name": "Dika",
    "orders_frequency": "Erratic, always trying new things, daring, healthy",
    "tastes": "Likes Chocolate and Vanilla, Likes Music, Likes Guitars, Does not like blueberries",
    "habits": "Orders tamwin every friday, and also asks for his thobes to be taken for laundry every friday as well"
}

# --- 4. STRUCTURE ---
class AppResponse(BaseModel):
    # We use List[List[str]] which is easier for the AI than List[Tuple]
    display_tags: List[List[str]] = Field(
        ..., 
        description="List of items. Each item is a list of 3 strings: [Service, Store, ItemName]."
    )
    
    assistant_message: str = Field(
        ..., 
        description="Natural language response including Snoonu Coins promotion."
    )

class UserQuery(BaseModel):
    query: str
    current_date: str = "2026-01-16"

# --- 5. ENDPOINT ---
@app.post("/chat", response_model=AppResponse)
def chat(user_input: UserQuery):
    
    # 1. RUN SEARCH (Filter data from 380k -> 2k tokens)
    relevant_context = search_database(user_input.query, USER_PROFILE["tastes"])
    
    random_coins = random.randint(1, 5)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=AppResponse,
        messages=[
            {
                "role": "system", 
                "content": f"""
                You are a smart assistant for {USER_PROFILE['name']}.
                
                USER PROFILE:
                {json.dumps(USER_PROFILE)}
                
                RELEVANT ITEMS FOUND IN DATABASE:
                {json.dumps(relevant_context)}
                
                --- LOGIC CONTROLLER ---
                
                **SCENARIO A: "Birthday/Prototype" Request**
                (Query mentions "Friend's birthday", "Jan 23", "Friday plan")
                   1. SELECT "Fällä" Boat + 1 Guitar + 1 Cake (No Blueberries).
                   2. MESSAGE: "I moved Tamwin & Laundry to Saturday."
                
                **SCENARIO B: Normal Search Request**
                   1. Pick the best matching items from the RELEVANT ITEMS list.
                   2. Filter by budget if asked.
                   3. MESSAGE: "Buy the first item to earn {random_coins} Snoonu Coins!"
                
                --- FORMATTING RULES ---
                OUTPUT FORMAT: A list of lists.
                [
                  ["Service Name", "Store Name", "Exact Item Name"],
                  ["Service Name", "Store Name", "Exact Item Name"]
                ]
                
                IMPORTANT:
                - Use the 'service' and 'store' fields EXACTLY as they appear in the RELEVANT ITEMS list.
                - If 'store' is N/A, keep it as "N/A".
                """
            },
            {"role": "user", "content": f"Today is {user_input.current_date}. {user_input.query}"},
        ],
    )
    return response