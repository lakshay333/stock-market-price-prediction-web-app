import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model

start = '2013-11-01'
end = '2023-12-12'

st.title('Stock Price Prediction')

user_input = st.text_input('Enter stock ticker', 'TSLA')
# df = data.get_data_yahoo(user_input, start=start, end=end)
df = yf.download(user_input, start=start, end=end)

#DESCRIBING DATA
st.subheader('Data of past 10 years')
st.write(df.describe())

#visualisation
st.subheader('Closing price vs Time Chart')
fig= plt.figure(figsize= (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig= plt.figure(figsize= (12,6))
plt.plot(ma100 , 'g')
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing price vs Time Chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig= plt.figure(figsize= (12,6))
plt.plot(ma100, 'g')
plt.plot(ma200 , 'r')
plt.plot(df.Close )

plt.xlabel('Time')
plt.ylabel('Price')

st.pyplot(fig)



# Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print (data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



#load model
model= load_model('keras_model.h5')


#testing
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test= []
y_test= []

for i in range (100, input_data.shape[0]):
    x_test.append (input_data[i-100:i])
    y_test.append (input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted= model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#final graph

st.subheader('Predictions vs Original')



fig2= plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label ='Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price' )
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)