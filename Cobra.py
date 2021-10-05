from sklearn import neighbors, tree, svm
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

import math
import numpy as np
import random
import logging
import numbers


logger = logging.getLogger('pycobra.classifiercobra')


class ClassifierCobra(BaseEstimator):

    def __init__(self, random_state=None, machine_list='basic'):
        self.random_state = random_state
        self.machine_list = machine_list
        self.estimators_ = {}
        self.machine_list ={}
        if machine_list == 'basic':
            self.machine_list = ['sgd', 'tree', 'knn', 'svm']
        if machine_list == 'advanced':
            self.machine_list = ['tree', 'knn', 'svm', 'logreg', 'naive_bayes', 'lda', 'neural_network']
    

    def shuffle_split_data(self, k=None, l=None, shuffle_data=True,split_ratio = 0.3):

        if shuffle_data:
            self.X_, self.y_ = shuffle(self.X_, self.y_, random_state=self.random_state)

        # if k is None and l is None:
        #     k = int(len(self.X_) / 2)
        #     l = int(len(self.X_))

        # if k is not None and l is None:
        #     l = len(self.X_) - k

        # if l is not None and k is None:
        #     k = len(self.X_) - l
        k = int(split_ratio*len(self.X_))
        l = len(self.X_) - k

        self.X_k_ = self.X_[:k]
        self.X_l_ = self.X_[k:l]
        self.y_k_ = self.y_[:k]
        self.y_l_ = self.y_[k:l]

        return self


    def fit(self, X, y, default=True, X_k=None, X_l=None, y_k=None, y_l=None,machine_list='basic',split_ratio =0.3):

        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.X_k_ = X_k
        self.X_l_ = X_l
        self.y_k_ = y_k
        self.y_l_ = y_l

        self.shuffle_split_data(split_ratio =split_ratio)

        for machine in self.machine_list:
            try:
                if machine == 'svm':
                    self.estimators_['svm'] = svm.SVC().fit(self.X_k_, self.y_k_)
                if machine == 'knn':
                    self.estimators_['knn'] = neighbors.KNeighborsClassifier().fit(self.X_k_, self.y_k_)
                if machine == 'tree':
                    self.estimators_['tree'] = tree.DecisionTreeClassifier().fit(self.X_k_, self.y_k_)
                if machine == 'logreg':
                    self.estimators_['logreg'] = LogisticRegression(random_state=self.random_state).fit(self.X_k_, self.y_k_)
                if machine == 'naive_bayes':
                    self.estimators_['naive_bayes'] = GaussianNB().fit(self.X_k_, self.y_k_)
                if machine == 'lda':
                    self.estimators_['lda'] = LinearDiscriminantAnalysis().fit(self.X_k_, self.y_k_)
                if machine == 'neural_network':
                    self.estimators_['neural_network'] = MLPClassifier(random_state=self.random_state).fit(self.X_k_, self.y_k_)
            except ValueError:
                continue

        self.machine_predictions_ = {}
        for machine in self.estimators_:
            self.machine_predictions_[machine] = self.estimators_[machine].predict(self.X_l_)

        return self.X_l_,self.y_l_

    def pred(self, X, M):

        # dictionary mapping machine to points selected
        select = {}
        for machine in self.estimators_:
            # machine prediction
            label = self.estimators_[machine].predict(X)
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
        if len(points) == 0:
            logger.info("No points were selected, prediction is 0")
            return 0

        # aggregate
        classes = {}
        for label in np.unique(self.y_l_):
            classes[label] = 0

        for point in points:
            classes[self.y_l_[point]] += 1

        result = int(max(classes, key=classes.get))
        return result


    def predict(self, X):

        X = check_array(X)

        M = len(self.estimators_)
        if X.ndim == 1:
            return self.pred(X.reshape(1, -1), M=M)

        result = np.zeros(len(X))
        index = 0
        for vector in X:
            result[index] = self.pred(vector.reshape(1, -1), M=M)
            index += 1
        
        return result
