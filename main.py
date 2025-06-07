from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://veridict.vercel.app"],  # your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your tokenizer and model ONCE when app starts
tokenizer = BertTokenizer.from_pretrained('final_bertCopy_tokenizer')
model = TFBertForSequenceClassification.from_pretrained('final_bertCopy_model')

def encode_texts(tokenizer, texts, max_length=256):
    encoding = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='tf'
    )
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'token_type_ids': encoding.get('token_type_ids', tf.zeros_like(encoding['input_ids']))
    }

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextRequest):
    inputs = encode_texts(tokenizer, [request.text])
    predictions = model.predict(inputs)
    predicted_class = np.argmax(predictions.logits, axis=1)[0]
    label = "AI-generated" if predicted_class == 1 else "Human-written"
    return {"prediction": label}
