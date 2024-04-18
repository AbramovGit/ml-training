from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from linear_reg import MyLineReg


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]
    model = MyLineReg(None, n_iter=50, learning_rate=0.1, reg='l1', sgd_sample=312, metric='mae')
    model.fit(X, y, 10)
    print(np.mean(model.get_coef()))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
