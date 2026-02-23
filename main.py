from fastapi import FastAPI, HTTPException
import joblib
import uvicorn
import pandas as pd
from models import ModelFeature, PredictionOut

app = FastAPI()

model_path = 'tuned_model.joblib'
try:
    model = joblib.load(model_path)
    print(f'Model loaded successfully from {model_path}')
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.post('/predict', response_model=PredictionOut)
async def predict_target(features: ModelFeature):
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded or unavailable.")

        input_df = pd.DataFrame([features.model_dump()])

        try:
             prediction = model.predict(input_df)
             prediction = prediction[0]
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
        
        return PredictionOut(
             predicted_value = prediction
        )

if __name__ == "__main__":
    uvicorn.run('main:app')