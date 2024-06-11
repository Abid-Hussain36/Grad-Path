import streamlit as st

import numpy as np
import pandas as pd
import sklearn
import pickle

pickle_in = open("xgb_clf.pkl", "rb")
model = pickle.load(pickle_in)

st.title("University Student Dropout Predictor :mortar_board:")
first_sem_credits = st.number_input("Number of Credits taken in the First Semester", 0, 20)
first_sem_grade = st.number_input("Grade Average in the First Semester (1 - 20)", 0, 20)
second_sem_credits = st.number_input("Number of Credits taken in the Second Semester", 0, 20)
second_sem_grade = st.number_input("Grade Average in the Second Semester (1 - 20)", 0, 20)
age_at_enrollment = st.number_input("Age at Enrollment", 0)
tuition_upto_date = st.radio("Are Tuition Fee Payments up to Date?", ["Yes", "No"])
tuition_upto_date = 1 if tuition_upto_date == "Yes" else 0
has_scholarship = st.radio("Is the Student a Scholarship Holder?", ["Yes", "No"])
has_scholarship = 1 if has_scholarship == "Yes" else 0
has_debt = st.radio("Is the Student in Debt?", ["Yes", "No"])
has_debt = 1 if has_debt == "Yes" else 0
gender = st.radio("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0
applied_first_round = st.radio("Did the Student Apply in the First Round of Applications to their University without Special Circumstances?", ["Yes", "No"])
applied_first_round = 1 if applied_first_round == "Yes" else 0
applied_after_23 = 1 if age_at_enrollment > 23 else 0
in_nursing = st.radio("Is the Student in a Nursing Program?", ["Yes", "No"])
in_nursing = 1 if in_nursing == "Yes" else 0

if 'output' not in st.session_state:
    st.session_state.output = ""

def predict():
    input_data = np.array([second_sem_grade, second_sem_credits, first_sem_grade, 
                  first_sem_credits, tuition_upto_date, has_scholarship,
                  age_at_enrollment, has_debt, gender,
                  applied_after_23, applied_first_round, in_nursing]).reshape((1, -1))
    prediction = model.predict(input_data)
    if(prediction == 1):
        st.session_state.output = "Student is at Risk for Dropping Out"
    else:
        st.session_state.output = "Student is not at Risk for Dropping Out"

st.button("Predict", on_click=predict)
st.write(st.session_state.output)
