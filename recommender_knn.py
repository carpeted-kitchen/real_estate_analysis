import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta
import time
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score,mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import warnings


training_data = pd.read_csv("Suitability_score_house.csv")
K = 10
X = training_data.drop(["status", "city", "brokered_by", 'prev_sold_date'], axis='columns')
Y = training_data["Suitability"]
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.5, random_state=42, shuffle=True)

train = train_x
print(train.head())
test = test_x

neighbors = []
knneigh = NearestNeighbors(n_neighbors=K)
knneigh.fit(train_x)
first_test_elem = test_x.iloc[0]
#Get the distances of its nearest neighbors
knns = knneigh.kneighbors([first_test_elem], return_distance=True)
print("Nearest neighbors:")
#print(knns)
#second param of knns is the indices of the population matrix
indices = knns[1][0]
#get the average score of the neighbors
avg_score = 0
for idx in indices:
    home = train.iloc[idx]
    #print(home)
    #print("Home suitability score: ", home["Suitability"])
    avg_score = avg_score+ home["Suitability"]
avg_score = avg_score/K
print("Predicted score for the first test property is: ", avg_score)
print("Actual score for this property is: ", test_y.iloc[0])
def predict_score(item):
    k_neigh = knneigh.kneighbors([item], return_distance=True)
    neighbors_idxs = k_neigh[1][0]
    avg = 0
    for i in neighbors_idxs:
        #Get the property at the given index
        prop = train.iloc[i]
        avg = avg + prop["Suitability"]
    avg = avg/K
    return avg
#pred_y = []

#print(test.head())
def score_pred(knn : NearestNeighbors, items):
    pred_y = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx, row in items.iterrows():
            k_nbrs = knn.kneighbors([row], return_distance = True)
            neighbors_idxs = k_nbrs[1][0]
            avg = 0
            for i in neighbors_idxs:
                prop = train.iloc[i]
                avg = avg + prop["Suitability"]

            avg = avg / len(neighbors_idxs)
            pred_y.append(avg)
    return pred_y
#with warnings.catch_warnings():
#        warnings.simplefilter("ignore")
#        for idx,row in test.iloc[:10000].iterrows():

#            predicted_score = predict_score(row)
#            pred_y.append(predicted_score)
test_x2 = test_x.iloc[0:100]
y_predict = score_pred(knneigh, test_x2)
test_y2 = test_y.iloc[0:100]
spearman = stats.spearmanr(y_predict, test_y2)
print("Knn Spearman Correlation:", spearman)