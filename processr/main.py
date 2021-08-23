import os
import uvicorn
import requests
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from utils import process_data

TRAINR_ENDPOINT = os.getenv("TRAINR_ENDPOINT")

# defining the main app
app = FastAPI(title="processr", docs_url="/")

# class which is expected in the payload while training
class DataIn(BaseModel):
    Status_of_existing_account: int
    Duration_of_Credit_month: int
    Payment_Status_of_Previous_Credit: int
    Purpose_of_loan: int
    Credit_Amount:int
    Value_of_Savings_accountbonds:int
    Years_of_Present_Employment:int
    Percentage_of_disposable_income:int
    Sex_Marital_Status:int
    Guarantors_Debtors:int
    Duration_in_Present_Residence:int
    Property:int
    Age_in_years:int
    Concurrent_Credits:int
    Housing:int
    No_of_Credits_at_this__Bank:int
    Occupation:int
    No_of_dependents:int
    Telephone:int
    Foreign_Worker:int
    Cost_Matrix_Risk: str


# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/process", status_code=200)
# Route to take in data, process it and send it for training.
def process(data: List[DataIn]):
    processed = process_data(data)
    # send the processed data to trainr for training
    response = requests.post(f"{TRAINR_ENDPOINT}/train", json=processed)
    return {"detail": "Processing successful"}


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
