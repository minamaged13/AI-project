import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('tmdb_movies_data.csv')
df.head()
df.info()
df['director'].value_counts()
df.drop(['id', 'imdb_id', 'original_title', 'homepage', 'tagline', 'director', 'overview',
        'cast', 'production_companies', 'release_year','release_date', 'keywords','budget','revenue'], axis=1, inplace=True)

df['net_profit'] = df['revenue_adj'] - df['budget_adj']


x = df.drop(['net_profit'], axis=1)
y = df['net_profit']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


y_train = np.array(y_train)
y_train = y_train.reshape(y_train.shape[0], 1)

y_test = np.array(y_test)
y_test = y_test.reshape(y_test.shape[0], 1)

x_train = np.vstack((np.ones((x_train.shape[0], )), x_train.T)).T
x_test = np.vstack((np.ones((x_test.shape[0], )), x_test.T)).T

print("X_train :", x_train.shape)
print("X_test :", x_test.shape)
print("y_train :", y_train.shape)
print("y_test :", y_test.shape)


def model(x, y, learning_rate, iteration):
    m = y.size
    theta = np.zeros((x.shape[1], 1))
    cost_ = []

    for i in range(iteration):
        y_pred = np.dot(x, theta)
        cost = (1/(2*m))*np.sum(np.square(y_pred-y))

        d_theta = (1/m)*np.dot(x.T, y_pred-y)
        theta = theta-learning_rate*d_theta

        cost_.append(cost)

        if((i % iteration/10) == 0):
            print("Cost :", cost)

    return theta, cost_


def Gradient_Descent(X, Y, Theta, alpha, num_iters):

    m = Y.shape[0]

    X1 = np.append(np.ones(X.shape), X, axis=1)
    for i in range(num_iters):
        h = Predictive_Line(X, Theta.reshape((2, 1)))
        Theta = Theta - (alpha / m) * ((h - Y).T).dot(X1)

    return Theta


iteration = 1000
learning_rate = 0.0000005
theta, cost_ = model(
    x_train, y_train, learning_rate=learning_rate, iteration=iteration)
