from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# load model
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "Penguin Prediction API"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    # align with training columns
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0]),
        "name": "Harshada Raut",
        "roll_no": "2022bcd0053"
    }