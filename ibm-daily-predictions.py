from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


# Define Constants
BASE_URL = "https://query1.finance.yahoo.com/v7/finance/download/IBM"
STEPS = 5  # Previous days depends on time steps used to train the model


def load_data(base_url, steps=7, current_date=datetime.today()):
    """
    Loads past 7 days data from yahoo finance
    :return: pandas data frame
    """
    # Account for weekends
    steps = steps + 2
    start_date = current_date - timedelta(days=steps)

    start_date = int(datetime.timestamp(start_date))
    current_date = int(datetime.timestamp(current_date))

    api_url = base_url + "?period1=" + str(start_date) + "&period2=" + str(current_date) + "&interval=1d&events=history"

    return pd.read_csv(api_url)


def data_timesteps(features, steps=1):
    """
    Convert data to 3 dimension array

    :param features: data
    :param steps: steps
    :return: numpy array
    """
    x_data = [features[0: steps]]
    # for i in range(steps, features.shape[0]):
    #     x_data.append(features[(i - steps): i])
    return np.array(x_data)


# Load data
current_date = datetime.strptime("2019/11/07 00:00", "%Y/%m/%d %H:%M")
data = load_data(BASE_URL, STEPS, current_date)
feature_data = data[['Open', 'High', 'Low']]

label_data = data[['Close']]

# Scale date
features_scaler = MinMaxScaler()
feature_data = features_scaler.fit_transform(feature_data)
feature_data = data_timesteps(feature_data, STEPS)

labels_scaler = MinMaxScaler()
labels_scaler.fit_transform(label_data)


# Load model
model = keras.models.load_model("ibm-prediction-model.h5")
prediction = model.predict(feature_data)

print(labels_scaler.inverse_transform(prediction))
