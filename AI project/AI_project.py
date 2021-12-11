import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('tmdb_movies_data.csv')
df.head()
df.info()
df['director'].value_counts()
df.drop(['id', 'imdb_id', 'original_title', 'homepage', 'tagline', 'director', 'overview',
        'cast', 'production_companies', 'release_year', 'release_date', 'keywords', 'budget', 'revenue'], axis=1, inplace=True)

df['net_profit'] = df['revenue_adj'] - df['budget_adj']


def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        # nationality, club, position (convert them to numbers 0- nclass-1)
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X

df = Feature_Encoder(df, ['genres'])


x = df.drop(['net_profit', 'revenue_adj'], axis=1)
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


def cost(x,y,theta):
    m=x.shape[0]
    prediction=x.dot(theta)
    error=np.square(np.subtract(prediction,y))
    
    j=1/(2*m)* np.sum(error)
    return j

def gradient_descent(x,y,theta,alpha,iteration):
    m=x.shape[0]
    cost_=np.zeros(iteration)
    
    for i in range(iteration):
        prediction=np.dot(x,theta)
        errors=np.subtract(prediction,y)
        delta=(alpha/m)*x.T.dot(errors)
        theta=theta-delta
        
        cost_[i]=cost(x,y,theta)
        if((i%iteration/10)==0):
            print("Cost :",cost)
    return theta,cost_

theta= np.array(np.zeros(7)).reshape((1,7)).T
iteration=400
alpha=0.15

theta.shape

theta,cost=gradient_descent(x_train,y_train,theta,alpha,iteration)
print("cost",cost[:40])




