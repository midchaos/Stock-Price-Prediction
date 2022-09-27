# import required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import pickle
import tensorflow as tf


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

def create_dataset(dir_path,file):
    q2_dataset = pd.read_csv(dir_path + file)

    dataset_list = []
    next_row = []
    rows = pd.DataFrame()

    for i in range(3, len(q2_dataset)):
        rows = q2_dataset.iloc[i-3:i+1,:]

        for j in range(3):
            row = rows.iloc[3-j]

            next_row.append(row[' Volume'])
            next_row.append(row[' Open'])
            next_row.append(row[' High'])
            next_row.append(row[' Low'])

        next_row.append((rows.iloc[0])[' Open'])
        dataset_list.append(next_row)
        next_row =[]

    dataset = pd.DataFrame(dataset_list)

    # scaling the data before splitting to avoid two different ranges.
    scalar = MinMaxScaler()
    scalar.fit(dataset)
    dataset = scalar.transform(dataset)

    dataset = pd.DataFrame(dataset,  columns=['Volume_1', 'Open_1', 'High_1', 'Low_1', 
                                                'Volume_2', 'Open_2', 'High_2', 'Low_2', 
                                                'Volume_3', 'Open_3', 'High_3', 'Low_3', 
                                                'Next_Day_Opening'])

    train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=81, shuffle=True)

    train_data.to_csv(dir_path + r'\data\train_data_RNN.csv', header=True, index_label=False)
    test_data.to_csv(dir_path + r'\data\test_data_RNN.csv', header=True, index_label=False)
    return

def load_data(dir_path: str, file: str):
    data = pd.read_csv(dir_path + file)

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1:]

    return X, y

def build_RNN_model(input_shape: tuple):
    model = Sequential()

    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model


if __name__ == "__main__":
    dir_path = os.path.dirname(__file__) 
    
    # create_dataset(dir_path, r"\data\q2_dataset.csv")
	
    # 1. load your training data
    X_train, y_train = load_data(dir_path = dir_path, file = r"/data/train_data_RNN.csv")

	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss
    RNN_model = build_RNN_model(input_shape = (X_train.shape[1],1))

    #training model with training dataset which is already scaled and storing the training parameters into a "history" variable.
    history = RNN_model.fit(X_train, y_train, 
                            epochs=100, batch_size=32, verbose=1)

    # prinitng the final training loss below
    print("Final Training Loss: "+str(history.history['loss'][-1]))

	# 3. Save your model
    # saving the model to be used in test_RNN.py file
    RNN_model.save(dir_path + r"\models\Group81_RNN_model.h5")
