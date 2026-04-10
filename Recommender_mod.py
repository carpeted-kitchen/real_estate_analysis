import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import warnings
import numpy as np
from sklearn.model_selection import train_test_split


training_data = pd.read_csv("Suitability_score_house.csv")
K = 10
X = training_data.drop(["status", "city", "brokered_by", 'prev_sold_date'], axis='columns')
Y = training_data["Suitability"]
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.5, random_state=42, shuffle=True)
knn = NearestNeighbors(n_neighbors=K)
train = train_x
test = test_x

def train_knn():
    knn.fit(train)
#predict the score of a list of properties
#parameters:
# items: list of items to score
# knn - KNearest Neighbors module to use - return_distance must be set to 'true'
def predict_score(knn : NearestNeighbors, items):
    pred_y = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx, row in items.iterrows():
            k_nbrs = knn.kneigbors([row], return_distance = True)
            neighbors_idxs = k_nbrs[1][0]
            avg = 0
            for i in neighbors_idxs:
                prop = train.iloc[i]
                avg = avg + prop["Suitability"]

            avg = avg / len(neighbors_idxs)
            pred_y.append(avg)
    return None