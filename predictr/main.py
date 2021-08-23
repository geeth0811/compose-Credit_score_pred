import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from utils import load_model, predict

# defining the main app
app = FastAPI(title="predictr", docs_url="/")

# class which is expected in the payload
class QueryIn(BaseModel):
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


# class which is returned in the response
class QueryOut(BaseModel):
    Cost_Matrix_Risk: str


# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/predict_creditscore", response_model=QueryOut, status_code=200)
# Route to do the prediction using the ML model defined.
# Payload: QueryIn containing the parameters
# Response: QueryOut containing the Credit_score predicted (200)
def predict_creditscore(query_data: QueryIn):
    output = {"Cost_Matrix_Risk": predict(query_data)}
    return output


@app.post("/reload_model", status_code=200)
# Route to reload the model from file
def reload_model():
    load_model()
    output = {"detail": "Model successfully loaded"}
    return output


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=9999, reload=True)
