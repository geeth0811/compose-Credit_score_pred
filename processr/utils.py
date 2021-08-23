import os
import pickle
from sklearn.naive_bayes import GaussianNB

# define the class encodings and reverse encodings
classes = {0: "Bad Risk", 1: "Good Risk"}
r_classes = {y: x for x, y in classes.items()}

# function to process data and return it in correct format
def process_data(data):
    processed = [
        {
            "Status_of_existing_account": d.Status_of_existing_account,
            "Duration_of_Credit_month": d.Duration_of_Credit_month,
            "Payment_Status_of_Previous_Credit": d.Payment_Status_of_Previous_Credit,
            "Purpose_of_loan": d.Purpose_of_loan,
            "Credit_Amount": d.Credit_Amount,
            "Value_of_Savings_accountbonds": d.Value_of_Savings_accountbonds,
            "Years_of_Present_Employment": d.Years_of_Present_Employment,
            "Percentage_of_disposable_income": d.Percentage_of_disposable_income,
            "Sex_Marital_Status": d.Sex_Marital_Status,
            "Guarantors_Debtors": d.Guarantors_Debtors,
            "Duration_in_Present_Residence": d.Duration_in_Present_Residence,
            "Property": d.Property,
            "Age_in_years": d.Age_in_years,
            "Concurrent_Credits": d.Concurrent_Credits,
            "Housing": d.Housing,
            "No_of_Credits_at_this__Bank": d.No_of_Credits_at_this__Bank,
            "Occupation": d.Occupation,
            "No_of_dependents": d.No_of_dependents,
            "Telephone": d.Telephone,
            "Foreign_Worker": d.Foreign_Worker,
            "Cost_Matrix_Risk": d.Cost_Matrix_Risk,
        }
        for d in data
    ]

    return processed
