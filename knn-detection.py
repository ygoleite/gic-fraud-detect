import numpy as np 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

filename = "D:/Repositorios/gic-fraud-detection/data/creditcard.csv"

data = pd.read_csv(filename)
print("file read successfuly")

X = data[["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]]
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(X,y)
print("train and test sets created")

knn = KNeighborsClassifier(n_neighbors = 5,n_jobs=16)
print ("Fitted Data")
data_fitted = knn.fit(X_train,y_train)
#print(data_fitted())
print("classifier created")
score = knn.score(X_test,y_test)
predicted_data = knn.predict(X_test)
print("model evaluated")
print(score)

print("Result")

print(predicted_data)

plt(predicted_data).show()