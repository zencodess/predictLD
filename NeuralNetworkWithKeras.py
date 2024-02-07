import keras 
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
import numpy as np
from numpy import genfromtxt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# import keras_metrics as km
from keras import metrics

# Data loading
my_data = genfromtxt('not16nm_Delays_process_temp_pvdd_cqloadNEWLOAD.csv', delimiter=',')
my_data = np.array(my_data)
Train = my_data[1:3750,:]
Test  = my_data[3750:5000,:]
Train_X = Train[:,0:20]
Train_y =Train[:,20:23]
Test_X = Test[:,0:20]
Test_y =  Test[:,20:23]
# print(Train)
# def larger_model():

# print(Test.shape)
	# create model
model = Sequential()
model.add(Dense(10, input_dim=20, kernel_initializer='normal', activation='tanh'))

model.add(Dense(5, input_dim=10, kernel_initializer='normal', activation='tanh'))
model.add(Dense(3, kernel_initializer='normal', activation='tanh'))

	# Compile model
model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=[metrics.mae, metrics.categorical_accuracy])


# print(Test_X)
# Train the model
model.fit(x=Train_X, y=Train_y, batch_size=20, epochs=128,  verbose=2)

scores = model.evaluate(x=Test_X, y=Test_y, batch_size=None, verbose=2, sample_weight=None, steps=None)
	
print('Test loss:', scores)	

# print('Test accuracy:', scores[1])

# Train the model, iterating on the data in batches of 32 samples
# model.fit(data, labels, epochs=10, batch_size=32))