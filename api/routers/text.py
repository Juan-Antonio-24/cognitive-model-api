from fastapi import APIRouter
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

router = APIRouter(prefix="/text", tags=["Text Emotion Detection"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent

route_final_model = (BASE_DIR.parent.parent / "model" / "Text Model" / "final_model")

tokenizer = AutoTokenizer.from_pretrained(route_final_model)

model = AutoModelForSequenceClassification.from_pretrained(route_final_model)

model.to(device)

model.eval()

id2label = {0: 'joyful', 
            1: 'mad', 
            2: 'peaceful', 
            3: 'powerful', 
            4: 'sad', 
            5: 'scared'}

class TextInput(BaseModel):
    text: str

@router.post("/predict")

async def predict(data: TextInput):

    text = data.text

    with torch.no_grad():
        inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt').to(device)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = probs.squeeze().cpu().numpy()

    pred_result = {}

    for i in range(len(preds)):
        pred_result[id2label[i]] = round(float(preds[i]) * 100, 2)

    pred_result = dict(sorted(pred_result.items(), key= lambda x: x[1], reverse=True))

    return {"Emotions": pred_result}