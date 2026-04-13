import requests
import json
from bs4 import BeautifulSoup as bs
import pandas as pd
import os
from pathlib import Path
from datetime import date, datetime, timedelta
import time
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score,mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import requests

#To calculate the weights:
#Add up the following indicators:
#Environment
#Housing
#Health
#Engagement
#Opportunity
#Transportation

#API Call format: https://api.livabilityindex.aarp.org/api/features/zip/findByLatLng?
#lat=39.07567&lng=-77.002078&state=Maryland+20904&location=Silver+Spring&specifiedGeoLevel=
data_df = pd.read_csv(
    "Datasets/realtor-data.zip.csv",
    dtype={"zip_code": str},
)
data_df = data_df[(data_df["state"] != "Puerto Rico") & (data_df["state"] != "Virgin Islands") 
                  & (data_df["state"] != "Guam")]
data_df = data_df.dropna(subset=['brokered_by', 'status', 'price','bed','bath','acre_lot','street','city','state','zip_code','house_size'])

url = "https://api.livabilityindex.aarp.org/api/features/zip/findByLatLng?lat=39.07567&lng=-77.002078&state=Maryland+20904&location=Silver+Spring&specifiedGeoLevel="
url2 = "https://api.livabilityindex.aarp.org/api/features/zip/20904/scores"

#r = requests.get(url = url2)
#request_content = r.json()
#results = request_content["result"][0] #the "result" field of request_content is an array, where the first element is another object that contains the scores
#print("Got request result")
#print(results["scores"])
max_sq_ft = data_df["house_size"].max()
min_sq_ft = data_df["house_size"].min()
max_price = data_df["price"].max()
min_price = data_df["price"].min()

#scores = results["scores"]
zi_base_current_df = pd.read_csv(
    "Datasets/(Copy) VT/usa_zi_base_currentyear.csv", dtype={"ID": str}
)
zi_base_current_df.set_index("ID", inplace=True)
base_current_stats_df = pd.read_csv(
    "Datasets/(Copy) VT/usa_zi_base_currentyear-statistics.csv"
)

prop_scored = []

for idx, row in zi_base_current_df.loc["01001":"01008"].iterrows():
    curr_zip = idx
    #print out zipcode
    print("Zipcode: ", curr_zip)
    #Call requests.get
    req_url = "https://api.livabilityindex.aarp.org/api/features/zip/" + curr_zip + "/scores"
    req_result = requests.get(req_url, timeout=60)
    print("Get a result: ", req_result.status_code)
    result_json = req_result.json()
    returned_results = result_json["result"][0]
    returned_score = returned_results["scores"]
    houses_at_zip = data_df.loc[data_df["zip_code"] == curr_zip]
    for idx, house in houses_at_zip.iterrows():
        livability_weighted_score = returned_score["score_prox"]
        house_price = house.price
        price_weighted_score = (
            (max_price - house_price) / (max_price - min_price)
        )
        sq_footage = house.house_size
        footage_weighted_score = (
            (max_sq_ft - sq_footage) / (max_sq_ft - min_sq_ft)
        )
    #add up the scores
    total_score = (price_weighted_score + footage_weighted_score +
                   returned_score['score_engage'] + returned_score['score_env'] + returned_score['score_health']
                   + returned_score['score_house'] + returned_score['score_opp'] + returned_score['score_trans']) / 8
    prop_scored.append(total_score)
    