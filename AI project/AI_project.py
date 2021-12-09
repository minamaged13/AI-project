import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



iterations = 100

data = pd.read_csv('tmdb_movies_data.csv')
#print(data.describe())
X=data['revenue_adj']
Y=data['budget_adj']
print(X.shape)
#Plotting
plt.scatter(X, Y)
plt.xlabel('revenue', fontsize = 20)
plt.ylabel('budget', fontsize = 20)
plt.show()



def Predictive_Line(X, Theta):

    Predictions = None
    X = np.append(np.ones((1,X.shape[0])), X.reshape((1,X.shape[0])), axis = 0)

  
    Predictions = np.dot(Theta.T, X)

  
    Predictions = Predictions.T
    return Predictions

def Calculate_Cost(X, Theta, Y):
    m = Y.shape[0]
    J = 0

    h = Predictive_Line(X, Theta)
    J = ((1/(2 * m)) * np.sum(np.square(h - Y), axis = 0))
 
    return J
def Gradient_Descent(X, Y, Theta, alpha, num_iters):

    m = Y.shape[0]
  
    X1 = np.append(np.ones(X.shape) , X, axis = 1)
    for i in range(num_iters):
        h = Predictive_Line(X, Theta.reshape((2,1)))
        Theta = Theta - (alpha / m) * ((h - Y).T).dot(X1)

    
    return Theta
