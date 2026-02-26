from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import joblib
import uvicorn
import pandas as pd
from models import ModelFeature, PredictionOut

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

model_path = 'tuned_model.joblib'
try:
    model = joblib.load(model_path)
    print(f'Model loaded successfully from {model_path}')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post('/predict', response_model=PredictionOut)
async def predict_target(features: ModelFeature):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    input_df = pd.DataFrame([features.model_dump()])

    try:
        prediction = model.predict(input_df)
        return PredictionOut(predicted_value=float(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    uvicorn.run('Fast:app', host='0.0.0.0', port=8000)