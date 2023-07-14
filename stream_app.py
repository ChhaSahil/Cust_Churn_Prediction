import pickle
import streamlit as st 
import pandas as pd
from PIL import Image
import numpy as np


model_file = 'model_save.sav'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

def cat_score(score):
  if 800<=score<=850:
    return 4
  elif 700<=score<800:
    return 3
  elif 600<=score<700:
    return 2
  else:
    return 1
def cat_country(country):
    if country=='France':
       return 0
    elif country == 'Germany':
       return 1
    else:
       return 2
def cat_gender(gender):
    if gender == 'Male':
        return 1
    else:
        return 0
   
  
def main():
    image = Image.open('Images/cust_churn.png')
    st.image(image,use_column_width=False)
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ('Online','Batch')
    )
    st.sidebar.info('This app is created to predict the Customer Churn')
    st.title("Predicting Customer Churn")
    if add_selectbox == 'Online':
        gender = st.selectbox('Gender:',['Male','Female'])
        country = st.selectbox('Country:',['France','Germany','Spain'])
        active_member = st.selectbox('Active Member :',['Yes','No'])
        credit_card = st.selectbox("Custome has our Company's Credit Card :",['Yes','No'])
        age = st.number_input('Age of the Customer :',min_value=18,max_value=150,value=18)
        estimated_salary = st.number_input('Estimated Salary of Customer :',min_value = 0,max_value=200000000,value=0)
        tenure = st.number_input('Number of years the customer has been with us :',min_value=0,max_value=200,value=0)
        credit_score = st.number_input('Credit Score of a Customer : ',min_value = 0,max_value = 851,value=0)
        balance = st.number_input('Balance of Customer : ',min_value=0, max_value = 250000000,value=0)
        product_number = st.number_input('Number of products from bank : ',min_value=0, max_value=700000000,value=0)
        output = ''
        output_prob = ''
        input_dict = {
           'country' : cat_country(country),
           'gender' : 1 if gender=='Male' else 0,
           'age' : age,
           'tenure' : tenure,
           'balance' : balance,
           'products_number' : product_number,
           'credit_card' : 1 if credit_card == 'Yes' else 0,
           'active_member' : 1 if active_member=='Yes' else 0,
           'estimated_salary' : estimated_salary,
           'cat_score' : cat_score(credit_score),
        }
        if st.button('Predict'):
           x = pd.DataFrame(input_dict,index=[0])
           y_pred = model.predict_proba(x)[0,1]
           churn = y_pred >= 0.5
           output_prob = float(y_pred)
           output = bool(churn)
           st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))
    if add_selectbox == 'Batch':
       file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
       if file_upload is not None:
            data = pd.read_csv(file_upload)
            data.drop(['customer_id'], axis = 1, inplace = True)
            X = data
            X['gender'] = X['gender'].apply(cat_gender)
            X['country'] = X['country'].apply(cat_country)
            X['cat_score'] = X['credit_score'].apply(cat_score)
            X.drop(['credit_score'],axis=1,inplace=True)
            X['products_number'] = X['products_number'].astype('float64')
            X['age'] = X['age'].astype('float64')
            X['tenure'] = X['tenure'].astype('float64')
            y_pred = model.predict_proba(X)[0, 1]
            churn = y_pred >= 0.5
            churn = bool(churn)
            st.write(churn)

if __name__ == '__main__':
   main()