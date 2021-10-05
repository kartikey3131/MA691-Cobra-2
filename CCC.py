from sklearn import neighbors, tree, svm
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from math import fabs

import math
import numpy as np
import random
import logging
import numbers


class ClassifierCobra(BaseEstimator):

    def __init__(self, random_state=None,machine_list = 'basic'):
        self.random_state = random_state
        self.machines ={} 
        if machine_list == 'basic':
            self.machine_list = ['sgd', 'tree', 'knn', 'svm']
        if machine_list == 'advanced':
            self.machine_list = ['tree', 'knn', 'svm', 'logreg', 'naive_bayes', 'lda', 'neural_network']        


    def __split_shuffle_data(self, X, y, split_ratio=0.3,shuffle_data = True,k=None):
        k = int(len(self.X_) / 2)
        l = int(len(self.X_))

        X_k = self.X_[:k]
        X_l = self.X_[k:l]
        y_k = self.y_[:k]
        y_l = self.y_[k:l]

        return X_k, y_k, X_l, y_l

    # def split_data(self, k=None, l=None, shuffle_data=True):

    #     if shuffle_data:
    #         self.X_, self.y_ = shuffle(self.X_, self.y_, random_state=self.random_state)

    #     if k is None and l is None:
    #         k = int(len(self.X_) / 2)
    #         l = int(len(self.X_))

    #     if k is not None and l is None:
    #         l = len(self.X_) - k

    #     if l is not None and k is None:
    #         k = len(self.X_) - l

    #     self.X_k_ = self.X_[:k]
    #     self.X_l_ = self.X_[k:l]
    #     self.y_k_ = self.y_[:k]
    #     self.y_l_ = self.y_[k:l]

    #     return self

 
    def fit(self, X, y, split_ratio = 0.3):

        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        X_k, y_k, X_l, y_l = self.__split_shuffle_data(self.X_, self.y_, split_ratio)
        # self.split_data()
        self.X_k_ = X_k
        self.X_l_ = X_l
        self.y_k_ = y_k
        self.y_l_ = y_l

        for machine in self.machine_list:
            try:
                if machine == 'svm':
                    self.machines['svm'] = svm.SVC().fit(self.X_k_, self.y_k_)
                if machine == 'knn':
                    self.machines['knn'] = neighbors.KNeighborsClassifier().fit(self.X_k_, self.y_k_)
                if machine == 'tree':
                    self.machines['tree'] = tree.DecisionTreeClassifier().fit(self.X_k_, self.y_k_)
                if machine == 'logreg':
                    self.machines['logreg'] = LogisticRegression(random_state=self.random_state).fit(self.X_k_, self.y_k_)
                if machine == 'naive_bayes':
                    self.machines['naive_bayes'] = GaussianNB().fit(self.X_k_, self.y_k_)
                if machine == 'lda':
                    self.machines['lda'] = LinearDiscriminantAnalysis().fit(self.X_k_, self.y_k_)
                if machine == 'neural_network':
                    self.machines['neural_network'] = MLPClassifier(random_state=self.random_state).fit(self.X_k_, self.y_k_)
            except ValueError:
                print("Invalid Machine Found")
                continue


            self.machine_predictions_ = {}
            for machine in self.machines:
                self.machine_predictions_[machine] = self.machines[machine].predict(self.X_l_)

            # self.all_predictions_ = np.array([])
            # for machine_name in self.machine_list:
            #     self.machine_predictions_[machine_name] = self.machines[machine_name].predict(X_l)
            #     self.all_predictions_ = np.append(self.all_predictions_, self.machine_predictions_[machine_name])

            return self

    def predict(self, X, M=None, info=False):


        X = check_array(X)

        if M is None:
            M = len(self.machines)
        if X.ndim == 1:
            return self.pred(X.reshape(1, -1), M=M)

        result = np.zeros(len(X))
        avg_points = 0
        index = 0
        for vector in X:
            if info:
                result[index], points = self.pred(vector.reshape(1, -1), M=M, info=info)
                avg_points += len(points)
            else:
                result[index] = self.pred(vector.reshape(1, -1), M=M)
            index += 1
        
        if info:
            avg_points = avg_points / len(X_array)
            return result, avg_points
        
        return result

    def pred(self, X, M, info=False):
        # dictionary mapping machine to points selected
        select = {}
        for machine in self.machines:
            # machine prediction
            label = self.machines[machine].predict(X)
            select[machine] = set()
            # iterating from l to n
            # replace with numpy iteration
            for count in range(0, len(self.X_l_)):
                if self.machine_predictions_[machine][count] == label:
                    select[machine].add(count)

        points = []
        # count is the indice number.
        for count in range(0, len(self.X_l_)):
            # row check is number of machines which picked up a particular point
            row_check = 0
            for machine in select:
                if count in select[machine]:
                    row_check += 1
            if row_check == M:
                points.append(count)

        # if no points are selected, return 0
        # if len(points) == 0:
        #     if info:
        #         logger.info("No points were selected, prediction is 0")
        #         return (0, 0)
        #     logger.info("No points were selected, prediction is 0")
        #     return 0

        # aggregate
        classes = {}
        for label in np.unique(self.y_l_):
            classes[label] = 0

        for point in points:
            classes[self.y_l_[point]] += 1

        result = int(max(classes, key=classes.get))
        if info:
            return result, points
        return result

