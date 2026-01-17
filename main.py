import os
import json
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Tuple
import instructor
from openai import OpenAI
from dotenv import load_dotenv

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

# --- 3. FLEXIBLE STRUCTURE ---
class AppResponse(BaseModel):
    # CHANGED: Now a List of Tuples (which become Arrays in JSON)
    display_tags: List[Tuple[str, str, str]] = Field(
        ..., 
        description="List of item details. Each item is a tuple of 3 strings: [Service Name, Store Name, Exact Item Name]."
    )
    
    assistant_message: str = Field(
        ..., 
        description="Natural language response. If Birthday Scenario, explain the full plan + reschedule. If normal search, just describe the items found."
    )

class UserQuery(BaseModel):
    query: str
    current_date: str = "2026-01-16"

# --- 4. ENDPOINT ---
@app.post("/chat", response_model=AppResponse)
def chat(user_input: UserQuery):
    
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
                IF the user asks about "Friend's birthday next Friday" or "Jan 23 plan":
                   1. FIND "Fällä" boat event (Jan 23).
                   2. SELECT 1 Gift (Guitar) + 1 Cake (No Blueberries).
                   3. MESSAGE: Pitch the "Daring Plan" and explicitly state: "I moved Tamwin & Laundry to Saturday."
                
                **SCENARIO B: Normal Search Request**
                IF the user asks for a specific item (e.g., "I want an amp", "Show me flowers"):
                   1. SEARCH the database for that specific item.
                   2. FILTER by price/specs if mentioned (e.g. "Budget 2000 QAR").
                   3. OUTPUT: Relevant display tags.
                   4. MESSAGE: Simple confirmation.
                   5. DO NOT reschedule habits or add cakes unless asked.
                   6. For every time an item is shown, for the first item reccomended choose a number between 1-5 and then mention that if they purchase the first item mentioned they would gain that many number of snoonu coins.
                
                --- TAG FORMATTING RULES (TUPLE FORMAT) ---
                1. Look at the parent object of the selected item in the JSON.
                2. Extract the "service" field and the "store" field.
                3. If "store" is missing (e.g. for Events), use "N/A" or the Service name.
                4. **FINAL FORMAT:** [Service Name, Store Name, Exact Item Name]
                   Example: ["Snooflower", "Snooflower Store", "Red Roses"]
                """
            },
            {"role": "user", "content": f"Today is {user_input.current_date}. {user_input.query}"},
        ],
    )
    return response