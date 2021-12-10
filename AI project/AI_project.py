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


data.drop(['id','imdb_id','original_title','homepage','tagline','director','overview','cast','production_companies','release_year','keywords'],axis=1,inplace=True)


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

cleanData()
normalizeData()
print(data.isnull().sum())
print(data.columns)
#print first 5 rows
print(data.head())



