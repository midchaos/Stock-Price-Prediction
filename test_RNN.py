# import required packages
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Dropout
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

def load_data(dir_path: str, file: str):
    data = pd.read_csv(dir_path + file)

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1:]

    return X, y

if __name__ == "__main__":

	dir_path = os.path.dirname(__file__) 

	# 1. Load your saved model
	RNN_model = tf.keras.models.load_model(dir_path + r"/models/Group81_RNN_model.h5")

	# 2. Load your testing data
	X_test, y_test = load_data(dir_path= dir_path, file = r"/data/test_data_RNN.csv")

	# 3. Run prediction on the test data and output required plot and loss
	y_pred = RNN_model.predict(X_test)
	y_pred = y_pred.flatten()
	y_test = np.array(y_test).flatten()

	print(r2_score(y_test, y_pred))
	