import numpy as np
from numpy.random import randint
import pandas as pd
import matplotlib.pyplot as plt
import Cobra

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

loans = pd.read_csv("new_loans.csv")

# l =[]
# for i in range(len(loans)):
#     if(loans.loc[i,'not.fully.paid']==0):
#         l.append(i)
#     if(len(l)>=6500):
#         break

# loans=loans.drop(labels=l)
loans=loans.sample(frac=1)
loans = loans.drop('purpose',1)

y = loans['not.fully.paid'] 
X = loans.drop('not.fully.paid',1)


COBRA = Cobra.ClassifierCobra(random_state=0, machine_list='basic')

X_test,Y_test = COBRA.fit(X,y,split_ratio=0.3)

prediction = COBRA.predict(X_test)

print(confusion_matrix(Y_test, prediction))
print(classification_report(Y_test, prediction))
# print(sum(prediction))

 