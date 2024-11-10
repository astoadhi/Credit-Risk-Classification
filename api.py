from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src import utils, preprocessing
import json

app = FastAPI()

# Load your model and OneHotEncoder objects
model = utils.deserialized_data("models/RandomForestClassifier_model.pkl")
ohe_home_ownership = utils.deserialized_data('models/ohe_home_ownership.pkl')
ohe_loan_intent = utils.deserialized_data('models/ohe_loan_intent.pkl')
ohe_loan_grade = utils.deserialized_data('models/ohe_loan_grade.pkl')
ohe_default_on_file = utils.deserialized_data('models/ohe_default_on_file.pkl')

class Item(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: int
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

@app.post("/predict")
async def predict(item: Item):
    # Convert the input data to DataFrame
    data = pd.DataFrame([item.dict()])  # Create DataFrame from the item

    # Preprocess the data
    processed_data = preprocessing.ohe_transform(data, 'person_home_ownership', 'home_ownership', ohe_home_ownership)
    processed_data = preprocessing.ohe_transform(processed_data, 'loan_intent', 'loan_intent', ohe_loan_intent)
    processed_data = preprocessing.ohe_transform(processed_data, 'loan_grade', 'loan_grade', ohe_loan_grade)
    processed_data = preprocessing.ohe_transform(processed_data, 'cb_person_default_on_file', 'default_onfile', ohe_default_on_file)

    # Predict probabilities
    proba = model.predict_proba(processed_data)
    # Load best threshold
    with open("models/best_threshold.json", 'r') as f:
        best_thresholds=json.load(f)
        print(best_thresholds)
    random_forest_info = best_thresholds[1]  # Get the second dictionary (index 1)
    random_forest_threshold = random_forest_info['best_threshold']

    # Determine prediction based on threshold
    
    prediction = (proba[:, 1] >= random_forest_threshold).astype(int)  # Assuming positive class is at index 1
    predicted_proba = proba[:, 1].tolist()  # Get probabilities for the positive class

    return {
        "prediction": int(prediction[0]),  # Convert to int for JSON serialization
        "probability": predicted_proba[0]  # Return the probability of the positive class
    }