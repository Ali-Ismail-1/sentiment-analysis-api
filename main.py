from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI(title="Sentiment Analysis API")

@app.get("/")
def root():
    return {"message": "Sentiment API is running!"}

@app.get("/analyze")
def analyze(text: str):
    result = sentiment(text)
    return {"text": text, "sentiment": result}