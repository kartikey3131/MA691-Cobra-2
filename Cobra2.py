import numpy as np
from numpy.random import randint
import pandas as pd
import matplotlib.pyplot as plt
from pycobra.classifiercobra import ClassifierCobra
from pycobra.cobra import Cobra
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
 
loans = pd.read_csv("loan_data.csv")
# df = pd.DataFrame(loans)
# purpose_c = pd.get_dummies(loans['purpose'], drop_first=True)
# loans_f = pd.concat([loans, purpose_c], axis=1).drop('purpose', axis=1)
loans = loans.drop('purpose',1)
# print(loans)
 
# loans_f = loans_f.iloc[1:1000,1:25]
# print(loans.head())
# y = loans['DEATH_EVENT'] 
# X = loans.drop('DEATH_EVENT',1)

# y = loans['Outcome'] 
# X = loans.drop('Outcome',1)
l =[]
# for c in range(6000):

loans=loans.sample(frac=1)

for i in range(len(loans)):
    if(loans.loc[i,'not.fully.paid']==0):
        l.append(i)
    if(len(l)>=6500):
        break
# # c=0

# # while(c<6000):
# #     r = randint(0,len(loans)-1)
# #     if(r in l):
# #         p=1
# #     else:
# #         l.append(r)
# #         c=c+1
loans=loans.drop(labels=l)
loans=loans.sample(frac=1)

# print(loans)
# print(len(loans))

y = loans['not.fully.paid'] 
X = loans.drop('not.fully.paid',1)


# print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3,shuffle=True)
print(sum(Y_train))
# print(loans_f.info())
COBRA = ClassifierCobra(random_state=0, machine_list='basic')
# COBRA = Cobra()
# COBRA.X_ = X
# COBRA.y_ = y
# COBRA = COBRA.split_data()
COBRA.fit(X_train,Y_train,default=True)
 
prediction = COBRA.predict(X_test)
print(sum(prediction))
print(confusion_matrix(Y_test, prediction))
print(classification_report(Y_test, prediction))
 