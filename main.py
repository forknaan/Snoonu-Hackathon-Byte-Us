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

# --- 2. PERSONA ---
USER_PROFILE = {
    "name": "Dika",
    "orders_frequency": "Erratic, always trying new things, daring, healthy",
    "tastes": "Likes Chocolate and Vanilla, Likes Music, Likes Guitars, Does not like blueberries",
    "habits": "Orders tamwin every friday, and also asks for his thobes to be taken for laundry every friday as well"
}

# --- 3. STRUCTURE ---
class AppResponse(BaseModel):
    display_tags: List[Tuple[str, str, str]] = Field(
        ..., 
        description="List of [Service, Store, ItemName] tuples. MUST use exact strings from the JSON."
    )
    
    assistant_message: str = Field(
        ..., 
        description="Natural language response. MUST include the Snoonu Coins promotion for normal searches."
    )

class UserQuery(BaseModel):
    query: str
    current_date: str = "2026-01-16"

# --- 4. ENDPOINT ---
@app.post("/chat", response_model=AppResponse)
def chat(user_input: UserQuery):
    
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
                
                FULL DATABASE:
                {json.dumps(KNOWLEDGE_BASE)}
                
                --- LOGIC CONTROLLER ---
                
                **SCENARIO A: The "Birthday/Prototype" Request**
                IF user asks about "Friend's birthday", "Jan 23", etc:
                   1. FIND "Fällä" boat event (Jan 23).
                   2. SELECT 1 Gift (Guitar) + 1 Cake (No Blueberries).
                   3. MESSAGE: Pitch the plan + "I moved Tamwin & Laundry to Saturday."
                
                **SCENARIO B: Normal Search Request**
                IF user asks for specific items:
                   1. SEARCH the database.
                   2. OUTPUT: Relevant tags.
                   3. MESSAGE: Confirm items + "Buy the first item to earn {random_coins} Snoonu Coins!"
                
                --- DATA EXTRACTION RULES (STRICT) ---
                1. When you select an item, look at its **Root Parent Object** in the JSON.
                2. Copy the `service` value EXACTLY. (e.g. "Music Store", not "Market").
                3. Copy the `store` value EXACTLY. (e.g. "Virgin Megastore"). 
                   - If `store` key is missing in the JSON object, return "N/A".
                4. Copy the `name` value EXACTLY.
                
                **FINAL FORMAT:** [Exact Service String, Exact Store String, Exact Item Name]
                """
            },
            {"role": "user", "content": f"Today is {user_input.current_date}. {user_input.query}"},
        ],
    )
    return response