import pickle
import boto3
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Initialize FastAPI App
app = FastAPI()

# Create S3 Client
s3_client = boto3.client("s3")

# 🔹 S3 Bucket and Model File Key (Directly specified)
S3_BUCKET_NAME = "your-bucket-name"  # 🔴 Replace with actual S3 bucket name
S3_MODEL_KEY = "health_recommendation_model.pkl"  # 🔴 Replace with actual model file name

# Load Model from S3
def load_model_from_s3():
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_MODEL_KEY)
        model_data = response["Body"].read()
        model = pickle.loads(model_data)
        print("✅ Model loaded successfully from S3.")
        return model
    except Exception as e:
        print(f"❌ Error loading model from S3: {str(e)}")
        return None

# Load Model at Startup
model = load_model_from_s3()
if model is None:
    raise Exception("⚠️ Failed to load the model from S3. API will not function properly.")

# Request Model for API
class HealthRecommendationRequest(BaseModel):
    healthIssue: str
    symptoms: List[str]
    age: int
    gender: str
    diet: str
    exercise: str
    weightChange: bool
    smokeAlcohol: bool
    medications: bool
    severity: str
    stress: str
    sleepIssues: bool
    energyLevel: int
    symptomWorsening: str
    symptomTrend: str
    consultedDoctor: bool

# 🏠 Home Route
@app.get("/")
def home():
    return {"message": "🔥 Health Recommendation API is running!"}

# 🏥 Health Recommendation Endpoint
@app.post("/get_recommendation/")
def get_recommendation(data: HealthRecommendationRequest):
    try:
        if model is None:
            return {"error": "Model not loaded from S3."}

        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict().values()])

        # 🔥 Predict recommendations using AI model
        prediction = model.predict(input_data)

        # 🔹 Ensure prediction is structured correctly
        if isinstance(prediction, list):
            prediction = prediction[0]  # Extract first result if list

        # Return AI-generated response
        return {
            "health_tip": prediction.get("health_tip", "No recommendation available"),
            "diet_schedule": prediction.get("diet_schedule", {
                "morning": "No recommendation",
                "afternoon": "No recommendation",
                "evening": "No recommendation"
            }),
            "severity": data.severity,
        }
    except Exception as e:
        return {"error": str(e)}
