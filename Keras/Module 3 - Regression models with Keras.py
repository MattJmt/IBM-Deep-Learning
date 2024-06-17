
import pandas as pd
import numpy as np

# download data
concrete_data = pd.read_csv('concrete_data.csv')
concrete_data.head()
concrete_data.shape
concrete_data.describe()
concrete_data.isnull().sum()

# split data into predictors and targets
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column
predictors.head()
target.head()
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()
n_cols = predictors_norm.shape[1] # number of predictors

import keras
from keras.models import Sequential # sequential is a neural network with layers
from keras.layers import Dense # several layers and nodes


# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,))) # 50 = number of neurons, input shape - number of input nodes
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# build the model
model = regression_model()

# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)