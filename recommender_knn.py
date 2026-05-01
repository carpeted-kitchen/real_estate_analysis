import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta
import time
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score,mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import warnings


training_data = pd.read_csv("Suitability_score_house.csv")
training_data.dropna(inplace = True, axis = 'index')

K = 10
X = training_data.drop(["status", "city", "brokered_by", 'prev_sold_date', "Suitability"], axis='columns')
Y = training_data["Suitability"]
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.2, random_state=42, shuffle=True)
scaler = StandardScaler()
scaler.fit(X)
X_train_scaled = scaler.transform(train_x)
X_test_scaled = scaler.transform(test_x)
train = train_x
print(train.head())
test = test_x

neighbors = []
knn_unscaled = NearestNeighbors(n_neighbors=K)
knn_unscaled.fit(train_x)
first_test_elem = test_x.iloc[0]
#Get the distances of its nearest neighbors
knns = knn_unscaled.kneighbors([first_test_elem], return_distance=True)
print("Nearest neighbors (unscaled):")
#print(knns)
knn_scaled = NearestNeighbors(n_neighbors=K)
knn_scaled.fit(X_train_scaled)

#second param of knns is the indices of the population matrix
indices = knns[1][0]
#get the average score of the neighbors
avg_score = 0
for idx in indices:
    home = train_y.iloc[idx]
    #print(home)
    #print("Home suitability score: ", home["Suitability"])
    avg_score = avg_score+ home
avg_score = avg_score/K
print("Unscaled KNN")
print("Predicted score for the first test property is: ", avg_score)
print("Actual score for this property is: ", test_y.iloc[0])

first_test_elem_scaled = X_test_scaled[0]
k_nearest_scaled = knn_scaled.kneighbors([first_test_elem_scaled])
indices_scaled = k_nearest_scaled[1][0]
avg_score2 = 0
for idx in indices_scaled:
    suit = train_y.iloc[idx] #suitability score
    avg_score2 = avg_score2 + suit
avg_score2 = avg_score2 / K
print("Scaled KNN")
print("predicted:", avg_score2)
print("actual", test_y.iloc[0])
def predict_score(item):
    k_neigh = knn_unscaled.kneighbors([item], return_distance=True)
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
                prop_score = train_y.iloc[i]
                avg = avg + prop_score

            avg = avg / len(neighbors_idxs)
            pred_y.append(avg)
    return pred_y
def score_pred_scaled(knn: NearestNeighbors, items):
    pred_y = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for elem in items:
            k_nbrs = knn.kneighbors([elem], return_distance = True)
            neighbors_idxs = k_nbrs[1][0]
            avg = 0
            for i in neighbors_idxs:
                prop_suit = train_y.iloc[i] #property suitability
                avg = avg + prop_suit

            avg = avg /K
            pred_y.append(avg)
    return pred_y

#with warnings.catch_warnings():
#        warnings.simplefilter("ignore")
#        for idx,row in test.iloc[:10000].iterrows():

#            predicted_score = predict_score(row)
#            pred_y.append(predicted_score)
test_x2 = test_x.iloc[0:100]
y_predict_before = score_pred(knn_unscaled, test_x2)
test_y2 = test_y.iloc[0:100]
spearman = stats.spearmanr(y_predict_before, test_y2)
print("Knn Spearman Correlation before:", spearman)
test_x2_scaled = X_test_scaled[0:100]
y_predict_after = score_pred_scaled(knn_scaled, test_x2_scaled)
spearman_after = stats.spearmanr(y_predict_after, test_y2)
print("Spearman correlation after: ", spearman_after)