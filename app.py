from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import lime
from lime.lime_text import LimeTextExplainer

app = FastAPI()

# Load model and tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))
model.eval()

# Risk Labels
labels = ["Low", "Moderate", "High"]

# Request model
class TextInput(BaseModel):
    text: str

# Predict Function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, predicted = torch.max(probs, dim=1)
    return labels[predicted.item()], confidence.item()

# Explanation Function
def get_explanation(text):
    class_names = labels

    def predict_proba(texts):
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probs.numpy()

    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_proba, num_features=5)
    return [word for word, weight in exp.as_list()]

# API Endpoint
@app.post("/predict")
def analyze_text(data: TextInput):
    try:
        prediction, confidence = predict(data.text)
        explanation = get_explanation(data.text)
        return {
            "risk_level": prediction,
            "confidence": round(confidence, 2),
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))