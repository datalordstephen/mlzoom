from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn


app = FastAPI()

# Define request schema
class LeadInput(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# res model
class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool

# switch pipeline v1 -> v2 for q6
with open('pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)

@app.post("/predict")
def predict(customer: LeadInput) -> PredictResponse:
    prob = predict_single(customer.model_dump())

    return PredictResponse(
        churn_probability=prob,
        churn=prob >= 0.5
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
