# import pandas as pd
# import numpy as np
# import pickle as pkl
# import streamlit as st
#
# # from analysis import cars_data
#
# model = pkl.load(open('./model/model.pkl', 'rb'))
#
# st.header('Car Price Prediction ML Model')
#
# cars_data = pd.read_csv('./data/Cardetails.csv')
#
# def get_brand_name(car_name):
#     car_name = car_name.split(' ')[0]
#     return car_name.strip()
# cars_data['name'] = cars_data['name'].apply(get_brand_name)
#
# name = st.selectbox('Select Car Brand', cars_data['name'].unique())
# year = st.slider('Car Manufactured Year', 1994, 2024)
# km_driven = st.slider('No of kms Driven',11,200000)
# fuel = st.selectbox('Fuel Type',cars_data['fuel'].unique())
# seller_type = st.selectbox('Seller Type',cars_data['seller_type'].unique())
# transmission = st.selectbox('Transmission Type',cars_data['transmission'].unique())
# owner = st.selectbox('Owner',cars_data['owner'].unique())
# mileage = st.slider('Car Mileage',10,40)
# engine = st.slider('Engine CC',700,5000)
# max_power = st.slider('Max Power',0,200)
# seats= st.slider('No of seats',5,10)
#
#
# if st.button('Predict'):
#     input_data_model = pd.DataFrame(
#         [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine,
#                        max_power, seats
#           ]], columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine',
#                        'max_power', 'seats'])
#
#     input_data_model['owner'].replace(['First Owner', 'Second Owner','Third Owner','Fourth & Above Owner', 'Test Drive Car'],[1,2,3,4,5],inplace=True)
#     input_data_model['fuel'] = cars_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4])
#     input_data_model['seller_type'] = cars_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3])
#     input_data_model['transmission'] = cars_data['transmission'].replace(['Manual', 'Automatic'], [1, 2])
#     input_data_model['name'] = cars_data['name'].replace(
#         ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra', 'Tata',
#          'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW',
#          'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
#          'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
#         list(range(1, 32))
#     )
#     st.write(input_data_model)
#
#     Car_Price_Prediction = model.predict(input_data_model)
#     st.markdown('Car Price is going to be '+str(Car_Price_Prediction[0]))


# import pandas as pd
# import pickle
# import streamlit as st
# from preprocessing.encoder import encode_features
#
# # Load model
# model = pickle.load(open('./model/model.pkl', 'rb'))
#
# st.title('🚗 Car Price Prediction App')
#
# # Load dataset
# cars_data = pd.read_csv('./data/Cardetails.csv')
# cars_data['name'] = cars_data['name'].apply(lambda x: x.split(' ')[0].strip())
#
# # Input Fields
# name = st.selectbox('Select Car Brand', sorted(cars_data['name'].unique()))
# year = st.slider('Car Manufactured Year', 1994, 2024)
# km_driven = st.slider('Kilometers Driven', 11, 200000)
# fuel = st.selectbox('Fuel Type', cars_data['fuel'].unique())
# seller_type = st.selectbox('Seller Type', cars_data['seller_type'].unique())
# transmission = st.selectbox('Transmission Type', cars_data['transmission'].unique())
# owner = st.selectbox('Owner', cars_data['owner'].unique())
# mileage = st.slider('Mileage (km/l)', 10, 40)
# engine = st.slider('Engine CC', 700, 5000)
# max_power = st.slider('Max Power (bhp)', 0, 200)
# seats = st.slider('Number of Seats', 5, 10)
#
# # Predict Button
# if st.button('Predict Price'):
#     input_df = pd.DataFrame([{
#         'name': name,
#         'year': year,
#         'km_driven': km_driven,
#         'fuel': fuel,
#         'seller_type': seller_type,
#         'transmission': transmission,
#         'owner': owner,
#         'mileage': mileage,
#         'engine': engine,
#         'max_power': max_power,
#         'seats': seats
#     }])
#
#     encoded_input = encode_features(input_df, cars_data)
#
#     try:
#         prediction = max(0, model.predict(encoded_input)[0])
#         st.success(f'💰 Estimated Car Price: ₹{prediction:,.2f}')
#     except Exception as e:
#         st.error(f'⚠️ Error during prediction: {e}')


import pandas as pd
import pickle as pkl
import streamlit as st
from preprocessing.encoder import encode_features

# Load dataset and preprocess brand names
cars_data = pd.read_csv('./data/Cardetails.csv')
cars_data['name'] = cars_data['name'].apply(lambda x: x.split(' ')[0].strip())

st.title('🚗 Car Price Prediction App')

# Input Form
name = st.selectbox('Select Car Brand', sorted(cars_data['name'].unique()))
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('Kilometers Driven', 11, 200000)
fuel = st.selectbox('Fuel Type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller Type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission Type', cars_data['transmission'].unique())
owner = st.selectbox('Owner', cars_data['owner'].unique())
mileage = st.slider('Mileage (km/l)', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power (bhp)', 0, 200)
seats = st.slider('Number of Seats', 5, 10)

# Load all models from disk
model_paths = {
    'Linear Regression': 'models/linear_regression_model.pkl',
    'Random Forest': 'models/random_forest_model.pkl',
    'Decision Tree': 'models/decision_tree_model.pkl',
}

loaded_models = {name: pkl.load(open(path, 'rb')) for name, path in model_paths.items()}

# Predict Button
if st.button('Predict Price'):
    input_df = pd.DataFrame([{
        'name': name,
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats
    }])

    encoded_input = encode_features(input_df, cars_data)

    st.markdown("### Model Predictions:")
    for model_name, model in loaded_models.items():
        try:
            prediction = max(0, model.predict(encoded_input)[0])
            st.success(f"**{model_name}: ₹{prediction:,.2f}**")
        except Exception as e:
            st.error(f"{model_name} failed: {e}")
