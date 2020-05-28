#!/usr/bin/env python
# coding: utf-8

# # Gold Price Prediction
# Predicts Gold Prices from [Yahoo Finance](https://query1.finance.yahoo.com/v7/finance/download/GOOG?period1=1092873600&period2=1589414400&interval=1d&events=history)

# ## Imports

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[2]:


plt.rcParams['figure.figsize'] = (14, 10)


# In[3]:


print(tf.__version__)


# ## Constants

# In[4]:


DATA_URL = "https://query1.finance.yahoo.com/v7/finance/download/GOLD?period1=476323200&period2=1590019200&interval=1d&events=history"
TRAIN_DATE_BOUNDARY = '2019-01-01'
EPOCHS = 60
BATCH_SIZE = 31
STEPS = 7


# ## Download data
# Download data from Yahoo finance and partition to test and training

# In[5]:


# data = pd.read_csv(DATA_URL, index_col="Date", parse_dates=["Date"])
data = pd.read_csv(DATA_URL)


# In[6]:


data.head()


# In[7]:


data_training = data[data['Date'] < TRAIN_DATE_BOUNDARY]
training_data = data_training.drop(['Date', 'Adj Close', 'Volume'], axis=1)
data_test = data[data['Date'] >= TRAIN_DATE_BOUNDARY]


# Scale data to improve training efficiency

# In[8]:


scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)


# A function for converting data into time steps dataset

# In[9]:


def data_timesteps(dataset, steps = 1):
    x_data = []
    y_data = []
    for i in range(steps, dataset.shape[0]):
        x_data.append(dataset[(i - steps): i])
        y_data.append(dataset[i, 0])
    return np.array(x_data), np.array(y_data)
    


# In[10]:


X_train, y_train = data_timesteps(training_data, STEPS)
X_train.shape, y_train.shape


# ## Model Configuration

# In[11]:


# model = keras.models.Sequential()
# model.add(keras.layers.LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.LSTM(units=60, activation='relu', return_sequences=True))
# model.add(keras.layers.Dropout(0.3))
# model.add(keras.layers.LSTM(units=80, activation='relu', return_sequences=True))
# model.add(keras.layers.Dropout(0.4))
# model.add(keras.layers.LSTM(units=120, activation='relu', return_sequences=True))
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(1))

# model.summary()


# In[12]:


model = keras.Sequential()
model.add(
  keras.layers.Bidirectional(
    keras.layers.LSTM(
      units=128, 
      input_shape=(X_train.shape[1], X_train.shape[2])
    )
  )
)
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))


# Compile and Train model

# In[13]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[14]:


X_train.shape, y_train.shape


# In[15]:


model_results = model.fit(
    X_train, 
    y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    validation_split=0.1, 
    shuffle=False
)


# Plot for training and validation

# In[16]:


plt.plot(model_results.history['loss'], label='Train', color='green')
plt.plot(model_results.history['val_loss'], label='Validation', color='red')
plt.show()


# ## Test Model

# In[17]:


past_steps_days = data_training.tail(STEPS)
df = past_steps_days.append(data_test, ignore_index=True)
df = data_test
df = df.drop(['Date', 'Adj Close', 'Volume'], axis=1)
inputs = scaler.transform(df)
    
X_test, y_test = data_timesteps(inputs, STEPS)
X_test.shape, y_test.shape


# Predict test data

# In[18]:


y_predict = model.predict(X_test)
# y_predict = np.argmax(y_predict, axis=1)
y_predict.shape


# Scale data back to original form

# In[19]:


scale = 1/scaler.scale_[0]
y_predict = y_predict * scale
y_test = y_test * scale
# y_predict = scaler.inverse_transform(y_predict)
# y_test = scaler.inverse_transform(y_test)


# ## Visualize Test Prediction

# In[20]:


plt.plot(y_test, color="green", label="Real Price")
plt.plot(y_predict, color="blue", label="Predicted Price")
plt.title("Gold Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()


# ## Credits
# 1. [KGP Talkie](https://youtu.be/arydWPLDnEc) LSTM Tutorial

# In[ ]:




