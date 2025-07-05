import streamlit as st
import pickle
import tensorflow as tf
import pandas as pd 

model = tf.keras.models.load_model('model.h5')

with open('transformer.pkl', 'rb') as file:
    transformer = pickle.load(file)

st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer is likely to churn based on their account information.")

CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
Geography = st.selectbox("Geography", options=['France', 'Spain', 'Germany'])
Gender = st.selectbox('Gender', options = ['Male', 'Female'])
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Tenure = st.number_input("Tenure (in years)", min_value=0, max_value=10, value=1)
Balance = st.number_input("Balance", min_value=0.0,  value=5000.0)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
HasCrCard = st.selectbox("Has Credit Card", options=['Yes', 'No'])
IsActiveMember = st.selectbox("Is Active Member", options=['Yes', 'No'])    
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

HasCrCard = 1 if HasCrCard == 'Yes' else 0
IsActiveMember = 1 if IsActiveMember == 'Yes' else 0

if st.button("Predict"):
    input_data = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Geography': [Geography],
        'Gender' : [Gender],
        'Age': [Age], 
        'Tenure': [Tenure],
        'Balance': [Balance],      
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'EstimatedSalary': [EstimatedSalary]
    })         
    transformed_input = transformer.transform(input_data)
    prediction = model.predict(transformed_input)
    result = "Churn" if prediction[0][0] > 0.5 else "Not Churn"
    st.subheader("Prediction Result")
    probability = prediction[0][0] 
    st.write(f"The customer is likely to **{result}**.")
    st.write(f"**Probability of churn:** {probability:.2%}")
