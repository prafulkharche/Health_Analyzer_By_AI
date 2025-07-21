# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("tokenizer")
model = BertForSequenceClassification.from_pretrained("bert_model")
model.eval()

# Class labels
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
