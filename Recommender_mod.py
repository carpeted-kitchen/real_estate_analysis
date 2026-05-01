import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



training_data = pd.read_csv("Suitability_score_house.csv")
K = 10
X = training_data.drop(["status", "city", "brokered_by", 'prev_sold_date'], axis='columns')
Y = training_data["Suitability"]
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.2, random_state=42, shuffle=True)
knn = NearestNeighbors(n_neighbors=K)
train = train_x
test = test_x
Scaler = StandardScaler()

def scaler_fit(scaler, data):
    scaler.fit(data)
    return None

#scales data (do this to the data b4 training the KNN model)
def scaler_transform(data):
    return Scaler.transform(data)


#trains the KNN model
#Note that for best results, used scaled training data
def train_knn(x_train_scaled):
    #scale data
    knn.fit(x_train_scaled)
    return None
#predict the score of a list of properties
#parameters:
# items: list of items to score - they don't have to be scaled
# knn - KNearest Neighbors module to use - return_distance must be set to 'true'
def predict_score(knn : NearestNeighbors, items, items_prescaled = False):
    pred_y = []
    #transform items using scaler
    if items_prescaled == False:
            Scaler.transform(items)

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