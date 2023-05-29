import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style = 'darkgrid')
import streamlit as st
import pyttsx3
import speech_recognition as sr
import time 
import os
import torch
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection  import train_test_split 


#Instructions
#1.Use the Request library to retrieve weather data from the OpenWeatherMap API.
#2.Preprocess the weather data to prepare it for machine learning.
#3.Use data visualization and statistical techniques to explore and analyze the weather data.
#4.Use XGBoost as your machine learning algorithm for weather prediction.
#5.Train your XGBoost model using the preprocessed data.
#6.Evaluate the performance of your XGBoost model using various metrics such as accuracy, precision, recall, and F1-score.
#7.Store the weather data in an SQLite database for future use.
#8.Create a chatbot using Python and integrate speech recognition.
#9.Design a Streamlit UI for your weather prediction app.
#10.Deploy your app on a cloud server such as streamlit share

#Using the Meteostat API to retrieve and connect the station ID of each states in Nigeria and the start/end dates
from meteostat import Stations, Daily
from datetime import datetime
current_date = datetime.today().strftime('%Y-%m-%d')

#Get data and Filter stations by country code (NG for Nigeria) also fetch stations info

def get_data(state):
    station = Stations()
    nig_stations = station.region('NG')
    nig_stations = nig_stations.fetch()
    global available_states
    #Clean up the data because some of the states have symbols in them
    nig_stations['name'] = nig_stations['name'].apply(lambda x: x.split('/', 1)[0])
    nig_stations.drop_duplicates(subset = ['name'], keep = 'first', inplace =True)
    available_states = nig_stations.name
    # Collect the data from the mentioned state if state in the list of available states 
    try:
        state_stations = nig_stations[nig_stations['name'].str.contains(state)]
        station_id = state_stations.index[0]
    except IndexError:
        error = f'Sorry, {state} is not among the available states. Please choose another neighboring state'
        raise ValueError(error)

    # Connect the API and fetch the data 
    data = Daily(station_id, str(state_stations.hourly_start[0]).split(' ')[0], str(current_date))
    data = data.fetch()

# Collect the necessary features needed for the model
    data['avg_temp'] = data[['tmin', 'tmax']].mean(axis=1)
    temp = data['avg_temp']
    press = data['pres']
    wind_speed = data['wspd']
    precip = data['prcp']
    rain_df = pd.concat([temp, press, wind_speed, precip], axis=1)

         # From the collected data, Create a Dataframe for training of modeland Classify Precipitation To Ascertain Rainfall.
            # Light rain — when the precipitation rate is < 2.5 mm (0.098 in) per hour. 
            # Moderate rain — when the precipitation rate is between 2.5 mm (0.098 in) – 7.6 mm (0.30 in) or 10 mm (0.39 in) per hour. 
            # Heavy rain — when the precipitation rate is > 7.6 mm (0.30 in) per hour, or between 10 mm (0.39 in) and 50 mm (2.0 in)
   
    rainfall = []
    for i in rain_df.prcp:
        if i < 0.1:
            rainfall.append('no rain')
        elif i < 2.5:
            rainfall.append('light rain')
        elif i >2.5 and i < 7.6:
            rainfall.append('moderate rain')
        else: 
            rainfall.append('heavy rain')
    model_data = rain_df.copy()
    rainfall = pd.Series(rainfall)
    model_data.reset_index(inplace = True)
    model_data['raining'] = rainfall
    model_data.index = model_data['time']
    model_data.drop(['time'], axis = 1, inplace = True)
    model_data.dropna(inplace=True)

    return data, temp, press, wind_speed, precip, model_data

  
# List out the states

nigerian_states = ['Sokoto', 'Gusau', 'Kaduna', 'Zaria', 'Kano', 'Maiduguri',
       'Ilorin', 'Bida', 'Minna', 'Abuja', 'Jos', 'Yola', 'Lagos',
       'Ibadan', 'Oshogbo', 'Benin City', 'Port Harcourt', 'Enugu',
       'Calabar', 'Makurdi', 'Uyo', 'Akure ', 'Imo ', 'Minna ']


#Modelling For Precipitation (Liquid from the atmosphere)
def modelling(data, target):
    # preprocess the data
    scaler = StandardScaler()
    encode = LabelEncoder()
    x = data.drop(target, axis = 1)
    for i in x.columns:
        x[[i]] = scaler.fit_transform(x[[i]])
    y = encode.fit_transform(data[target])

    # find the best random-state
    model_score = []
    for i in range(1, 100):
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = i)
        xgb_model = XGBClassifier()
        xgb_model.fit(xtrain, ytrain)
        prediction = xgb_model.predict(xtest)
        score = accuracy_score(ytest, prediction)
        model_score.append(score)

    best_random_state = [max(model_score), model_score.index(max(model_score))]

    # Train the dataset after getting the best random_state
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = best_random_state[1] + 1)
    xgb_model = XGBClassifier()
    xgb_model.fit(xtrain, ytrain)
    prediction = xgb_model.predict(xtest)
    score = accuracy_score(ytest, prediction)
    report = classification_report(ytest, prediction)
    report = classification_report(ytest, prediction, output_dict=True)
    report = pd.DataFrame(report).transpose()
    print(report)
    print(f"Accuracy Score: {score}")

    return xgb_model, best_random_state, report


# ARIMA TIME SERIES FOR TEMPERATURE, PRESSURE AND WIND-SPEED
from statsmodels.tsa.arima_model import ARIMA

# Get time and response variable of the original data
def response_data(df, pressure, temperature, wind, precipitation):
    pressure = df[['pres']]
    temp = df[['avg_temp']]
    wind = df[['wspd']]
    precip = df[['prcp']]
    return pressure, temp, wind, precip

# Plot
def plotter(dataframe):
    sns.set
    plt.figure(figsize = (15, 3))
    plt.subplot(1,1,1)
    sns.lineplot(dataframe)

# Collection of data
def item_collector(data, selected_date):
    selected_row = data.loc[data.index == selected_date]
    return float(selected_row.values[0])

# Determining lag value for our time series model by looping through possible numbers using GRID SEARCH method
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm

def best_parameter(data):
    # Create a grid search of possible values of p,d,and q values
    p_values = range(0, 5)
    d_values = range(0, 2)
    q_values = range(0, 4)

    # Create a list to store the best AIC values and the corresponding p, d, and q values
    best_aic = np.inf
    best_pdq = None

    # Loop through all possible combinations of p, d, and q values
    for p in p_values:
        for d in d_values:
            for q in q_values:
                # Fit the ARIMA model
                model = sm.tsa.arima.ARIMA(data, order=(p, d, q))
                try:
                    model_fit = model.fit()
                    # Update the best AIC value and the corresponding p, d, and q values if the current AIC value is lower
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_pdq = (p, d, q)
                except:
                    continue
    return best_pdq


# --------- MODELLING FOR TIME SERIES USING ARIMA MODEL -------------------
import statsmodels.api as sm

# ...Create a function that models, predict, and returns the model and the predicted values
def arima_model(data, best_param):
    plt.figure(figsize = (14,3))
    model = sm.tsa.ARIMA(data, order = best_param)
    model = model.fit()

    # Plot the Future Predictions according to the model
    future_dates = pd.date_range(start = data.index[-1], periods = 5, freq = 'D')
    forecast = model.predict(start = len(data), end = len(data) + (5 - 1))

    # get the dataframes of the original/prediction
    data_ = data.copy()
    data_['predicted'] = model.fittedvalues

    # Get the dataframe for predicted values
    predictions_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
    predictions_df.set_index('Date', drop = True, inplace = True)

    return predictions_df


# EMBEDDING TALKBACK
# Create a function for text Talkback
def Text_Speaker(your_command):
    speaker_engine = pyttsx3.init() #................ initiate the talkback engine
    speaker_engine.say(your_command) #............... Speak the command
    speaker_engine.runAndWait() #.................... Run the engine

directions = "Follow the instructions below to effectively use the app: 1. Launch the App: Open the Weather Prediction App on your device. 2.Enable Voice Input: Make sure your devices microphone is enabled and properly working. Grant necessary permissions if prompted. 3. Speak Naturally: Communicate with the app using natural language while making sure there is no overbearing background noise. You can ask questions or provide instructions using voice commands. 4. Mention Weather-related Keywords: When you want to inquire about the weather or climate, use words like 'weather' or 'climate' in your conversation. For example: Whats the weather like today?, Tell me the climate forecast for tomorrow., Is it going to rain in Lagos?. 5. Provide Location Information: Specify the location for which you want the weather prediction. You can mention the city name, ZIP code, or any other relevant location details. 6. Listen to the Response: The app will process your voice command, retrieve the necessary weather information, and provide a response. You can listen to the apps voice output or read the displayed information on your devices screen. 7. End the Conversation: When you have received the desired weather prediction or want to exit the app, you can end the conversation by saying 'Goodbye', 'Thank you', or any other appropriate closing remark. Enjoy Your Usage !!!"


# EMBEDDING SPEECH RECOGNITION
# Create a function for Speech Trancription
def transcribe_speech():
    # Initialize recognizer class
    r = sr.Recognizer()

    # Reading Microphone as source
    with sr.Microphone() as source:

        # create a streamlit spinner that shows progress
        with st.spinner(text='Silence pls, Caliberating background noise.....'):
            time.sleep(3)

        r.adjust_for_ambient_noise(source, duration = 1) # ..... Adjust the sorround noise
        st.info("Speak now...")

        audio_text = r.listen(source) #........................ listen for speech and store in audio_text variable
        with st.spinner(text='Transcribing your voice to text'):
            time.sleep(2)

        try:
            # using Google speech recognition to recognise the audio
            text = r.recognize_google(audio_text)
            # print(f' Did you say {text} ?')
            return text
        except:
            return "Sorry, I did not get that."

# STREAMLIT IMPLEMENTATION
st.markdown("<h1 style = 'color: #D21312'>WEATHER FORECAST APP</h1> ", unsafe_allow_html = True)
st.markdown("<h6 style = 'top_margin: 0rem;color: #FF6000'>Built by Odugbesi Nofisat</h6>", unsafe_allow_html = True)
st.image('image4.jpg', width =500)

st.subheader("Do you need weather information to help plan your day better? Ask Tito.")

if st.sidebar.button('Direction to use'):
    st.sidebar.markdown("<h4 style = 'text-decoration: underline;  color: #2A2F4F'>Read The Users Directions</h4> ", unsafe_allow_html = True)
    st.sidebar.write(['Launch the App: Open the Weather Prediction App on your device', 'Enable Voice Input: Make sure your devices microphone is enabled and working properly. Grant necessary permissions if prompted', 'Speak Naturally: Communicate with the app using natural language while making sure there is no overbearing background noise. You can ask questions or provide instructions using voice commands', "Mention Weather-related Keywords: When you want to inquire about the weather or climate, use words like 'weather' or 'climate' in your conversation. For example: Whats the weather like today?, Tell me the climate forecast for tomorrow., Is it going to rain in Lagos?", 'Provide Location Information: Specify the location for which you want the weather prediction. You can mention the city name, ZIP code, or any other relevant location details', 'Listen to the Response: The app will process your voice command, retrieve the necessary weather information, and provide a response. You can listen to the apps voice output or read the displayed information on your devices screen', "End the Conversation: When you have received the desired weather prediction or want to exit the app, you can end the conversation by saying 'Goodbye', 'Thank you', or any other appropriate closing remark"])
if st.sidebar.button('Listen To The Users Directions'):
    Text_Speaker(directions)


# Next, we will initaite a speech recognition button that takes speech 
# Run the get_data function to fetch data 
# Pick out mentions of weather condition and state in the speech input 
# Get the necessary data for these mentions 

if st.button("Talk to me"):
    your_words_in_text = transcribe_speech()
    st.write("Transcription: ", your_words_in_text)
    place = [i for i in your_words_in_text.capitalize().split() if i in nigerian_states]
# Set the default value to 'Lagos' if no match is found
    if not place:
        place = 'Lagos'
    place_data, temp, press, wind_speed, precip, model_data = get_data(place[0])# ....... Call the get_data function to fetch place data.

    selected_date = st.date_input("Select a date within a 5 days spectrum for prediction") # ------------ Collect the date
    date_string = selected_date.strftime("%Y-%m-%d") # ----------------------- Turn the date to a string
    #Get repsonse data for time series 
    pressure_data, temperature_data, wind_data, precipitation_data =  response_data(place_data, press, temp, wind_speed, precip)
    #Get Best Time Series Parameters
    pressure_param = best_parameter(pressure_data)
    wind_param = best_parameter(wind_data)
    temp_param = best_parameter(temperature_data)
    precip_param = best_parameter(precipitation_data)
    # Time Series Prediction
    pressure_predict = arima_model(pressure_data,  pressure_param)
    windSpeed_predict = arima_model(wind_data,  wind_param)
    temperature_predict = arima_model(temperature_data,  temp_param)
    precipitation_predict = arima_model(precipitation_data, precip_param)
    # Collect the future data in the particular date mentioned by the user
    if date_string in pressure_predict.index:
        future_pressure = item_collector(pressure_predict, date_string)
        future_windSpeed = item_collector(windSpeed_predict, date_string)
        future_temperature = item_collector(temperature_predict, date_string)
        future_precipitation = item_collector(precipitation_predict, date_string)
    else: st.warning('Oops, the date you chose is beyond 5days. Pls choose a date 5days from now')
            
    if 'rainfall' in your_words_in_text.lower().strip():
        print(f"You mentioned to predict Rainfall for {date_string}. Pls wait while I get to my destination")
        #Modelling
        xgb_model, best_random_state, report = modelling(model_data, 'raining')
        # Testing the Rainfall Model (In order of Temp, Pressure, Windspeed, Precipitation)
        input_value = [[future_temperature, future_pressure, future_windSpeed, future_precipitation	]]
        scaled = StandardScaler().fit_transform(input_value)
        prediction = xgb_model.predict(scaled)
        if int(prediction) == 0:
            st.success('There Wont be Rain')
            st.image('sunny2.jpg', width =500)
            st.text('Have a beautiful sunny day, This is a day to plan an outdoor event')
        elif int(prediction) == 1:
            st.success('There will be light rain')
            st.image('lightrain3.jpg', width =500)
            st.text('The sky will be gloomy, Its okay to pack an umbrella and a light jacket')
        elif int(prediction) == 2:
            st.success('There will be moderate level of rain')
            st.image('moderaterain2.jpg', width =500)
            st.text('There will be showers of blessings, Please dress warm and plan for a ride while going out')
        else:
            st.success('There will be heavy rain')
            st.image('heavyrain3.jpg', width= 500)
            st.text('It will be a stormy day, This might be the best day to stay indoors' )
        
        
    elif 'temperature' in your_words_in_text.lower().strip():
        st.success(f"you mentioned to predict Temperature for {date_string}. Pls wait while I get to my destination")
        st.success(f"The temperature for {place} at {date_string} is {future_temperature}")
    elif 'wind speed' in your_words_in_text.lower().strip():
        st.success(f"you mentioned to predict Temperature for {date_string}. Pls wait while I get to my destination")
        st.success(f"The temperature for {place} at {date_string} is {future_windSpeed}")
    elif 'pressure' in your_words_in_text.lower().strip():
        st.success(f"you mentioned to predict Temperature for {date_string}. Pls wait while I get to my destination")
        st.success(f"The temperature for {place} at {date_string} is {future_pressure}")
    elif 'precipitation' in your_words_in_text.lower().strip():
        st.success(f"you mentioned to predict Temperature for {date_string}. Pls wait while I get to my destination")
        st.success(f"The temperature for {place} at {date_string} is {future_precipitation}")
    else:
        st.error('couldnt find an action')
    st.warning('Pls note that I will automatically select Lagos in the absence of a location')
    
