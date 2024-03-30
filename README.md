# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Develop a Recurrent Neural Network (RNN) model to predict the stock prices of Google. The goal is to train the model using historical stock price data and then evaluate its performance on a separate test dataset. The prediction accuracy of the model will be assessed by comparing its output with the true stock prices from the test dataset.

Dataset:
![image](https://github.com/S-Priyadharshan/rnn-stock-price-prediction/assets/145854138/60136a9d-f216-4d79-ba78-94ee52b1a13a)


## Design Steps

### Step 1:
Import the necessary modules 

### Step 2:
import the training dataset csv file

### Step 3:
Allocate the first column as the main training data set

### Step 4:
Perform the necessary pre-processing on the training data set

### Step 5:
Build your Deep Learning model which makes the use of SimpleRNN layer to learn

### Step 6:
Fit and compile the model with the necessary loss functions, epochs and batch size

### Step 7:
Scale the testing data and use it to test your model

### Step 8:
Check the output graph to verify your models effectiveness and make the required changes

## Program
#### Name: Priyadharshan S
#### Register Number: 212223240127

```
import tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras import models
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

dataset_train = pd.read_csv('trainset.csv')

dataset_train.columns

dataset_train.head()

train_set = dataset_train.iloc[:,1:2].values

train_set

type(train_set)

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

length = 60
n_features = 1

model=Sequential([
    layers.SimpleRNN(50,input_shape=(length,n_features)),
    layers.Dense(1),
])
model.compile(optimizer='adam',loss='mse')

model.summary()

model.fit(X_train1,y_train,epochs=100, batch_size=32)

dataset_test = pd.read_csv('testset.csv')

test_set = dataset_test.iloc[:,1:2].values

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

print("Name: Priyadharshan S          Register Number: 212223240127    ")
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

```

## Output

### True Stock Price, Predicted Stock Price vs time
![image](https://github.com/S-Priyadharshan/rnn-stock-price-prediction/assets/145854138/dd01ca4e-415f-40eb-9bc3-cea25fb58a1c)


### Mean Square Error

![image](https://github.com/S-Priyadharshan/rnn-stock-price-prediction/assets/145854138/ba3c2466-7045-4ff2-835b-7d83108122ed)

![image](https://github.com/S-Priyadharshan/rnn-stock-price-prediction/assets/145854138/ac870891-34c1-4443-92fa-cac68ba30316)

## Result
Hence a nerual network based on Simple RNN to predict Google stock prices is successfully established.
