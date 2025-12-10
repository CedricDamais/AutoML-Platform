import os
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json

app = FastAPI(title="ML Deployment Inference API")

model = None
label_mapping = None

@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        print("MODEL_PATH environment variable not set.")
        return

    try:
        print(f"Loading model from {model_path}...")
        model = mlflow.pyfunc.load_model(model_path)
        print("Model loaded successfully.")
        
        mapping_path = os.path.join(model_path, "labels.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                label_mapping = json.load(f)
            label_mapping = {int(k): v for k, v in label_mapping.items()}
            print(f"Loaded label mapping: {label_mapping}")
        else:
            print("No label mapping found.")
    except Exception as e:
        print(f"Error loading model: {e}")
        pass

class InferenceInput(BaseModel):
    data: list[dict]

@app.post("/predict")
def predict(input_data: InferenceInput):
    global model, label_mapping
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame(input_data.data)
        predictions = model.predict(df)
        
        if hasattr(predictions, "values"):
            preds = predictions.values
        elif hasattr(predictions, "tolist"):
            preds = np.array(predictions)
        else:
            preds = np.array(list(predictions))

        if len(preds.shape) == 2 and preds.shape[1] > 1:
            is_logits = np.any((preds < 0) | (preds > 1.0))
            if is_logits:
                exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
                probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
                
                class_ids = np.argmax(probs, axis=1)
                
                response = {
                    "predictions": probs.tolist(), 
                    "predicted_classes": class_ids.tolist(),
                    "status": "success",
                    "note": "Logits converted to probabilities via Softmax"
                }
                
                if label_mapping:
                    response["predicted_labels"] = [label_mapping.get(c, str(c)) for c in class_ids]
                
                return response

        response = {"predictions": preds.tolist()}
        
        if label_mapping and preds.ndim == 1 and np.issubdtype(preds.dtype, np.integer):
             response["predicted_labels"] = [label_mapping.get(int(c), str(c)) for c in preds]
        elif label_mapping and preds.ndim == 1 and np.issubdtype(preds.dtype, np.floating):
             if np.all(np.mod(preds, 1) == 0):
                  response["predicted_labels"] = [label_mapping.get(int(c), str(c)) for c in preds]
             
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
