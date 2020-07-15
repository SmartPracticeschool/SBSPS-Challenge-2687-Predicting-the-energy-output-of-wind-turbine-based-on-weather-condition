import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf

df = pd.read_csv("dataset/T1.csv")

training_set = df.iloc[:40000, :].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set[:, 1:])

# Creating a data structure with 432 timesteps and 1 output
X_train = []
y_train = []
for i in range(432, 40000):
    X_train.append(training_set_scaled[i - 432:i, 1:])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Part 2 - Building the RNN

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 3)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=1, batch_size=32)


test_set = df.iloc[40000:, 1:].values
test_set_scaled = sc.transform(test_set)

# Creating a data structure with 60 timesteps and 1 output
X_test = []
y_test = []
for i in range(432, 10097):
    X_test.append(test_set_scaled[i - 432:i, 1:])
    y_test.append(test_set_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

predicted_out = regressor.predict(X_test)


a = np.zeros((9665, 1))
predicted_out = np.concatenate((predicted_out, a, a, a), axis=1)
act = df.iloc[40432:, 1:2].values
predicted_out = sc.inverse_transform(predicted_out)



# Visualising the results

plt.plot(predicted_out[0:, 0:1], color='blue', label='Predicted ')
plt.plot(df.iloc[40432:-433, 1:2].values, color='red', label='Real ')
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('output')
plt.legend()
plt.show()



model_json = regressor.to_json()
with open("regressor.json", "w") as json_file:
    json_file.write(model_json)

regressor.save_weights("model.h5")






