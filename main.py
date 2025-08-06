from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI(title="Sentiment Analysis API")

# Initialize Hugging Face sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define request body model
class TextInput(BaseModel):
    text: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API! Use /docs for API documentation."}

# Sentiment analysis endpoint
@app.post("/analyze")
def analyze_sentiment(input: TextInput):
    result = sentiment_analyzer(input.text)[0]
    return {
        "text": input.text,
        "sentiment": result["label"].lower(),
        "confidence": round(result["score"], 4)
    }