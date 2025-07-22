from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import requests

MODEL_PATH = "bert_model"
TOKENIZER_PATH = "tokenizer"

MODEL_FILE = "bert_model.pkl"
MODEL_URL = "https://drive.google.com/file/d/1PAZk8v16f5-7vwzECxPQ1CH0CRnh5H6x/view?usp=drive_link"  # Replace with real ID

def download_model():
    if not os.path.exists(MODEL_FILE):
        print("Downloading model...")
        r = requests.get(MODEL_URL)
        with open(MODEL_FILE, "wb") as f:
            f.write(r.content)
        print("Model downloaded.")

download_model()

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device("cpu")))
model.eval()

id2label = {
    0: "cold",
    1: "fever",
    2: "headache",
    3: "anxiety",
    4: "depression"
}

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/predict")
def predict_question(data: QuestionRequest):
    inputs = tokenizer(data.question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    label = id2label[prediction]
    confidence = round(probs[0][prediction].item(), 3)

    return {"prediction": label, "confidence": confidence}
