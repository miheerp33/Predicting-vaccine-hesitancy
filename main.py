
import matplotlib.pyplot as plt
import numpy as np

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
url = "vh_data15.csv"
names = ['Trust_Science_Community', 'Fear_Needles', 'Trust_National', 'Vaccine_Trust_Index', 'Household_Income', 'Vaccine_Hesitant']
dataset = read_csv(url, usecols=names)
dataset = dataset.dropna(axis=0)
array = dataset.values

X = array[:,0:4]
y = array[:,5]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)


for i in range(len(predictions)):
    print(predictions[i], '-||-', Y_validation[i])

score = accuracy_score(predictions, Y_validation)
print(score*100,'%')
