import streamlit as st
import requests

# Set the page configuration
st.set_page_config(
    page_title="Credit Risk Prediction Web App",
    page_icon="💳",
    layout="wide",
)

# Title and description
st.title("Credit Risk Prediction Web App")
st.markdown("""
    This application predicts the probability of credit default based on user input.
    Please fill in the details below to get your prediction.
""")

# Create a sidebar for input fields
st.sidebar.header("User  Input")

# Input fields
person_age = st.sidebar.number_input("Person Age", min_value=20, max_value=80, value=20)
person_income = st.sidebar.number_input("Person Income", min_value=0, value=5000)
person_home_ownership = st.sidebar.selectbox("Home Ownership", options=["OWN", "RENT", "MORTGAGE","OTHER"])
person_emp_length = st.sidebar.number_input("Employment Length (Years)", min_value=0, value=5)
loan_intent = st.sidebar.selectbox("Loan Intent", options=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_grade = st.sidebar.selectbox("Loan Grade", options=["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.sidebar.number_input("Loan Amount", min_value=0, value=1000)
loan_int_rate = st.sidebar.number_input("Interest Rate (%)", min_value=5.0,max_value=23.0, value=5.0)
loan_percent_income = st.sidebar.number_input("Loan Percentage of Income", min_value=0.0,max_value=0.9, value=0.0)
cb_person_default_on_file = st.sidebar.selectbox("Default on File", options=["Yes", "No"])
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length", min_value=0, value=0)

# Button to predict
if st.sidebar.button("Predict"):
    # Prepare the data for the API request
    data = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": "Y" if cb_person_default_on_file == "Yes" else "N",
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
    }
    
    # Make the API request
    response = requests.post("http://localhost:8000/predict", json=data)
    
    # Display the result
    if response.status_code == 200:
        result = response.json()
        prediction = result['prediction']  
        probability = result['probability']

        # Set the color based on the prediction
        if prediction == 0:
            color = "green"
            message = "Prediction: Non-Default"
        else:
            color = "red"
            message = "Prediction: Default"

        # Display the prediction result with the corresponding color
        st.markdown(f"<h1 style='color: {color};'>{message}</h1>", unsafe_allow_html=True)
        st.markdown(f"**Probability:** {probability:.2f}")
    else:
        st.error("Error in prediction. Please check your input values.")
st.markdown("---")
st.header("Input Data Table")

# Display the input data in a table
input_data = {
    "Age": person_age,
    "Income": person_income,
    "Home Ownership": person_home_ownership,
    "Employment Length": person_emp_length,
    "Loan Intent": loan_intent,
    "Loan Grade": loan_grade,
    "Loan Amount": loan_amnt,
    "Interest Rate": loan_int_rate,
    "Loan Percentage of Income": loan_percent_income,
    "Default on File": cb_person_default_on_file,
    "Credit History Length": cb_person_cred_hist_length,
}

# Create a DataFrame for better visualization
import pandas as pd

input_df = pd.DataFrame(input_data.items(), columns=["Feature", "Value"])
st.table(input_df)

# Footer
st.markdown("---")
st.markdown("Made by Triasto Adhinugroho")