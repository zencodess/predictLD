import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
import pandas as pd
import numpy as np
from numpy import genfromtxt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.regularizers import l1_l2
from keras import optimizers
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
#from sklearn.linear_model import Ridge
# import keras_metrics as km
from keras import metrics
# Data loading
my_data = genfromtxt('dvd_data_1.csv', delimiter=',')
my_data = np.array(my_data)
Train = my_data[1:37500,:]
Test  = my_data[37500:50000,:]
x_train = Train[:,0:20]
y_train =Train[:,20:23]
x_test= Test[:,0:20]
y_test =  Test[:,20:23]
# print(Train)
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.

model.add(Dense(20, activation='relu', input_dim=20))
model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=1.)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=100)
score = model.evaluate(x_test, y_test)
##################model = Sequential()
##################model.add(Dense(3,  # output dim is 3, one score per each class
#                activation='tanh',
#                kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
#                input_dim=20))  # input dimension = number of features your data has
#keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,clipnorm=1.)
#sgd=keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
##model.compile(optimizer='rmsprop',loss='mean_absolute_error',metrics=['accuracy'])
#########################model.compile(optimizer='rmsprop',
    #          loss='categorical_crossentropy',
    #          metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer='Adam', metrics=[metrics.mae, metrics.categorical_accuracy])

######################model.fit(Train_X, Train_y, epochs=100, validation_data=(Test_X, Test_y))

##############scores = model.evaluate(x=Test_X, y=Test_y, batch_size=None, verbose=2, sample_weight=None, steps=None)
print(x_test)
print('Test loss:', score)

# print('Test accuracy:', scores[1])

# Train the model, iterating on the data in batches of 32 samples
# model.fit(data, labels, epochs=10, batch_size=32))
