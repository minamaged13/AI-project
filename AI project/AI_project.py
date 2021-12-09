import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

iterations = 100
data = pd.read_csv('tmdb_movies_data.csv')
print(data.columns)

# add new column for net_profit
data['net_profit'] = data['revenue_adj'] - data['budget_adj']

def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        # nationality, club, position (convert them to numbers 0- nclass-1)
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


def removeUnnecessaryColumns():
    global data
    data = data.drop(['id'], axis=1)
    data = data.drop(['imdb_id'], axis=1)
    data = data.drop(['director'], axis=1)
    data = data.drop(['original_title'], axis=1)
    data = data.drop(['cast'], axis=1)
    data = data.drop(['homepage'], axis=1)
    data = data.drop(['tagline'], axis=1)
    data = data.drop(['keywords'], axis=1)
    data = data.drop(['overview'], axis=1)
    data = data.drop(['production_companies'], axis=1)
    data = data.drop(['release_date'], axis=1)
    data = data.drop(['release_year'], axis=1)


def cleanData():
    global data
    # remove null values
    data = data.dropna()
    # remove duplicates
    data = data.drop_duplicates()

    # Filter and clean the columns and rows
    data = data[data['vote_average'] > 0]
    data = data[data['vote_count'] > 0]
    data = data[data['popularity'] > 0]
    data = data[data['revenue'] > 0]
    data = data[data['runtime'] > 0]
    data = data[data['budget'] > 0]
    data = data[data['genres'] != '[]']


def normalizeData():
    global data
    # normalize the data
    data['revenue'] = data['revenue'] / 1000000000
    data['budget'] = data['budget'] / 1000000000
    data['popularity'] = data['popularity'] / 100
    data['runtime'] = data['runtime'] / 60

# convert categorical columns into numerical using Feature_Encoder
data = Feature_Encoder(data, ['genres'])

removeUnnecessaryColumns()
cleanData()
normalizeData()





print(data.isnull().sum())
print(data.columns)

#print first 5 rows
print(data.head())


X = data['revenue_adj']
Y = data['budget_adj']
print(X.shape)

# plt.scatter(X, Y)
# plt.xlabel('revenue', fontsize = 20)
# plt.ylabel('budget', fontsize = 20)
# plt.show()

# def Predictive_Line(X, Theta):

#    Predictions = None
#    X = np.append(np.ones((1,X.shape[0])), X.reshape((1,X.shape[0])), axis = 0)


#    Predictions = np.dot(Theta.T, X)


#    Predictions = Predictions.T
#    return Predictions

# def Calculate_Cost(X, Theta, Y):
#    m = Y.shape[0]
#    J = 0

#    h = Predictive_Line(X, Theta)
#    J = ((1/(2 * m)) * np.sum(np.square(h - Y), axis = 0))

#    return J

# def Gradient_Descent(X, Y, Theta, alpha, num_iters):

#    m = Y.shape[0]
#    X1 = np.append(np.ones(X.shape) , X, axis = 1)
#    for i in range(num_iters):
#        h = Predictive_Line(X, Theta.reshape((2,1)))
#        Theta = Theta - (alpha / m) * ((h - Y).T).dot(X1)
#    return Theta
