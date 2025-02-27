from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai
import os
from faiss_search import search_faiss  # Ensure this module exists

# ✅ Load API Key from environment variables
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY. Set it as an environment variable.")

# ✅ Initialize Gemini AI Model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# ✅ FastAPI App Setup
app = FastAPI()

# ✅ Enable CORS for Flutter/Web requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Root Endpoint
@app.get("/")
def root():
    return {"message": "AI Document Search API with Gemini is running!"}

# ✅ Request Model for Queries
class QueryRequest(BaseModel):
    query: str

# ✅ Handle AI Queries with FAISS + Gemini AI
@app.post("/ask")
def post_ask(request: QueryRequest):
    try:
        # 🔍 Step 1: Search FAISS for relevant data
        status, retrieved_chunks = search_faiss(request.query)

        # 🚫 If no relevant data is found, return default response
        if status == "Out of context":
            return {"response": "I cannot answer that based on the provided information."}

        # ✨ Step 2: Construct the prompt for Gemini AI
        prompt = f"""
        You are an AI assistant. Based on the following retrieved information, answer the query:
        \n\n{retrieved_chunks}\n\n
        Question: {request.query}
        """

        # 🔥 Step 3: Generate response using Gemini AI
        response = model.generate_content(prompt)

        # ✅ Extract response text properly
        if response and hasattr(response, "candidates") and response.candidates:
            answer = response.candidates[0].text
        else:
            answer = "I couldn't generate a response."

        return {"response": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Run the FastAPI Server (for Render Deployment)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render assigns a dynamic port
    uvicorn.run(app, host="0.0.0.0", port=port)
