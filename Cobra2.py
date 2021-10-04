import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycobra.classifiercobra import ClassifierCobra
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
 
loans = pd.read_csv("loan_data.csv")
purpose_c = pd.get_dummies(loans['purpose'], drop_first=True)
loans_f = pd.concat([loans, purpose_c], axis=1).drop('purpose', axis=1)
 
# loans_f = loans_f.iloc[1:1000,1:19]
print(loans_f.head())
y = loans_f['not.fully.paid'] 
X = loans_f.drop(columns=['not.fully.paid'])
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=40)
 
# print(loans_f.info())
COBRA = ClassifierCobra(random_state=0, machine_list='basic')
# COBRA.X_ = X
# COBRA.y_ = y
# COBRA = COBRA.split_data()
COBRA.fit(X_train,Y_train,default=True)
 
prediction = COBRA.predict(X_test)
print(prediction)
print(confusion_matrix(Y_test, prediction))
print(classification_report(Y_test, prediction))
 
# COBRA.__init__