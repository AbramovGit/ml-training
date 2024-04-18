import math
import random

import numpy as np
import pandas as pd


class MyLineReg:
    metric_value = 0

    def __init__(self, weights: pd.Series = None, metric=None, n_iter=100, learning_rate=0.5, reg=None, l1_coef=0,
                 l2_coef=0, sgd_sample=None, random_state=42):
        self.random_state = random_state
        self.sgd_sample = sgd_sample
        self.reg = reg
        self.l2_coef = l2_coef
        self.l1_coef = l1_coef
        self.weights = weights
        self.metric = metric
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def __repr__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        random.seed(self.random_state)
        X.insert(0, "ones", pd.Series([1] * X.shape[0]))
        is_learning_rate_number = isinstance(self.learning_rate, (int, float, complex))

        batch_X = X
        batch_y = y

        self.weights = pd.Series(np.ones(batch_X.shape[1]))

        for i in range(1, self.n_iter + 1):
            if isinstance(self.sgd_sample, int):
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
                batch_X = X.iloc[sample_rows_idx].reset_index(drop=True)
                batch_y = y.iloc[sample_rows_idx].reset_index(drop=True)
            elif isinstance(self.sgd_sample, float):
                sample_rows_idx = random.sample(range(X.shape[0]), math.trunc(X.shape[0] * self.sgd_sample))
                batch_X = X.iloc[sample_rows_idx].reset_index(drop=True)
                batch_y = y.iloc[sample_rows_idx].reset_index(drop=True)

            if is_learning_rate_number:
                actual_lr = self.learning_rate
            else:
                actual_lr = self.learning_rate(i)

            batch_y_ = np.dot(batch_X, self.weights)
            y_ = np.dot(X, self.weights)
            grad = (2 / batch_y.shape[0]) * np.dot((batch_y_ - batch_y), batch_X) + self.calculate_regularization()
            self.weights = self.weights - actual_lr * grad

            # mse and metric to be computed on entire dataset
            mse = np.sum((y - y_) ** 2) / y.shape[0]
            self.calculate_metric(np.dot(X, self.weights), y)
            if verbose and i != 0 and i % verbose == 0:
                print(f"{i} | loss {mse} | learning rate: {actual_lr} | {self.metric}: {self.metric_value}")

    def get_coef(self):
        return np.delete(np.array(self.weights), 0)

    def predict(self, X: pd.DataFrame):
        X.insert(0, "col1", pd.Series([1] * X.shape[0]))
        return np.dot(X, self.weights)

    def get_best_score(self):
        return self.metric_value

    def calculate_regularization(self):
        l1 = self.l1_coef * self.weights.map(lambda x: self.sgn(x))
        l2 = self.l2_coef * 2 * self.weights
        if self.reg == 'l1':
            return l1
        if self.reg == 'l2':
            return l2
        if self.reg == 'elasticnet':
            return l1 + l2
        return 0

    def sgn(self, number):
        if number < 0:
            return -1
        if number > 0:
            return 1
        return 0

    def calculate_metric(self, y_, y):
        if self.metric == 'mae':
            self.metric_value = np.sum(abs(y - y_)) / y.shape[0]
        if self.metric == 'mse':
            self.metric_value = np.sum((y - y_) * (y - y_)) / y.shape[0]
        if self.metric == 'rmse':
            self.metric_value = (np.sum((y - y_) * (y - y_)) / y.shape[0]) ** (1 / 2)
        if self.metric == 'mape':
            self.metric_value = (100 * np.sum(abs((y - y_) / y))) / y.shape[0]
        if self.metric == 'r2':
            y_a = np.average(y)
            self.metric_value = 1 - (np.sum((y - y_) * (y - y_)) / np.sum((y - y_a) * (y - y_a)))
