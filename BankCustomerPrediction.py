# Artificial Neural Network
# Rihad Variawa, Data Scientist

#Part 1 - Data Preprocessing
#...........................

#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the dataset
df = pd.read_csv('BankCustomers.csv')
X = df.iloc[3:13].values
y = df.iloc[:13].values

#Convert categorical variables into dummy variables
states=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

#Concatenate the remaining dummies columns
X=pd.concat([X,states,gender],axis=1)

#Drop the columns as it is no longer required
X=X.drop(['Geography','Gender'],axis=1)

#Splitting the dataset into Training & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
#To normalize variables so measuring units do not affect the algorithm
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part 2 - Now let's make the ANN!
#................................

#Import the Keras libraries & packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

#Adding the second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

#Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

#Part 3 - Making the predictions and evaluating the model
#........................................................

#Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Construct a confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test,y_pred)
