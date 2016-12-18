from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pandas as pd
import numpy as np

#Ordinary Least Squares Linear Regression (Vectorized)
class LinearRegression(object):

    def fit(self, X, y):
        # Fits training data and finds optimal weight
        # to reduce RSS cost function

        #Determine optimized weights
        self.betaHat = inv(X.T.dot(X)).dot(X.T).dot(y)

        #Determine how well our optimized weights do
        self.cost = (y - X.dot(self.betaHat)).T.dot(y - X.dot(self.betaHat))

        return self

    def predict(self, X):
        #Predict y values given our optimized weights
        return X.dot(self.betaHat)

def show(X, y, model):
    plt.scatter(X, y, c = 'blue')
    plt.plot(X, model.predict(X), color = 'red')
    return None

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header = None, sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df.iloc[:, :-1].values
y = df.iloc[:,-1].values

sc_x = StandardScaler()
sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)

lr = LinearRegression()
lr.fit(X_std, y_std)

print(lr.cost)
