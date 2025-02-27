# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from faiss_search import ask_question, search_faiss, query_company_ai
# import uvicorn
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# # ✅ Enable CORS (Fixes XMLHttpRequest error in Flutter Web)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace "*" with your frontend URL if needed
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Root Endpoint
# @app.get("/")
# def root():
#     return {"message": "AI Document Search API is running!"}

# # GET request for asking a question
# @app.get("/ask")
# def get_ask(query: str):
#     try:
#         return {"response": ask_question(query)}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Request model for POST request
# class QueryRequest(BaseModel):
#     query: str

# # POST request for asking a question
# @app.post("/ask")
# def post_ask(request: QueryRequest):
#     try:
#         status, retrieved_chunks = search_faiss(request.query)

#         if status == "Out of context":
#             return {"response": "I cannot answer that based on the provided information."}

#         ai_response = query_company_ai(request.query, retrieved_chunks)
#         return {"response": ai_response}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Run the FastAPI server
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)



# import json
# import requests
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from faiss_search import ask_question, search_faiss

# # Load configuration from JSON
# with open("config.py") as config_file:
#     config = json.load(config_file)

# GEMINI_API_KEY = config["GEMINI_API_KEY"]
# ENDPOINT_URL = config["ENDPOINT_URL"]
# RELEVANCE_THRESHOLD = config["RELEVANCE_THRESHOLD"]

# app = FastAPI()

# # ✅ Enable CORS (Fixes XMLHttpRequest error in Flutter Web)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace "*" with your frontend URL if needed
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Root Endpoint
# @app.get("/")
# def root():
#     return {"message": "AI Document Search API is running!"}

# # Request model for POST request
# class QueryRequest(BaseModel):
#     query: str

# # Function to query Gemini API
# def query_gemini(query: str, context: str):
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {GEMINI_API_KEY}",
#     }
#     payload = {
#         "messages": [
#             {"role": "system", "content": "You are a helpful AI assistant."},
#             {"role": "user", "content": f"Based on this context: {context}, answer: {query}"}
#         ],
#         "temperature": 0.7
#     }

#     try:
#         response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
#         response.raise_for_status()
#         return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
#     except requests.exceptions.RequestException as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # POST request for asking a question
# @app.post("/ask")
# def post_ask(request: QueryRequest):
#     try:
#         status, retrieved_chunks = search_faiss(request.query)

#         if status == "Out of context":
#             return {"response": "I cannot answer that based on the provided information."}

#         ai_response = query_gemini(request.query, retrieved_chunks)
#         return {"response": ai_response}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Run the FastAPI server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai
from faiss_search import search_faiss  # Keeping FAISS search for context retrieval

# ✅ Replace this with your actual Gemini API Key
API_KEY = "AIzaSyAuof-veYPuBB9bplKO-54A0PIgG0Mjkdo"

# ✅ Initialize Google Gemini AI
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")  # Using Gemini Pro model

app = FastAPI()

# ✅ Enable CORS (Fixes XMLHttpRequest error in Flutter Web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root Endpoint
@app.get("/")
def root():
    return {"message": "AI Document Search API with Gemini is running!"}

# Request model for POST request
class QueryRequest(BaseModel):
    query: str

# ✅ POST request for asking a question with Gemini AI
@app.post("/ask")
def post_ask(request: QueryRequest):
    try:
        # 🔍 Step 1: Retrieve relevant chunks from FAISS
        status, retrieved_chunks = search_faiss(request.query)

        # 🚫 If no relevant data is found, return a response
        if status == "Out of context":
            return {"response": "I cannot answer that based on the provided information."}

        # ✨ Step 2: Construct prompt with FAISS results
        prompt = f"""
        You are an AI assistant. Based on the following retrieved information, answer the query:
        \n\n{retrieved_chunks}\n\n
        Question: {request.query}
        """

        # 🔥 Step 3: Generate a response using Gemini AI
        response = model.generate_content(prompt)

        return {"response": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
