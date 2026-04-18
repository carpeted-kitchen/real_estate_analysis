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

searchplace = "New York, NY"  # Can enter state, region, city, zip

#QoL is 27% of which 30% is based on crime rates and 25%  on weather

home_price_wt = 0.24
wthr_temp_wt = 0.07
wthr_extm_wt = 0.0405
les_wt = 0.027
crime_wt = 0.081
sq_ft_wt = 0.1
#Taxes weight
tax_wt = 0.16

url = "https://health.usnews.com/best-hospitals/search-data?specialty_id=IHQCANC&page=1"

directory = str(Path(__file__).parent.parent) + "/data/"

usn_url = "https://health.usnews.com"
hdr = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Ch-Ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    "Dnt": "1",
    "Upgrade-Insecure-Requests": "1",
}
# Timestamp
timestamp = datetime.now()
print(timestamp)

# Initiate lists
names = []
aha_ids = []
hospital_ids = []
addresses = []
cities = []
states = []
state_abbrevs = []
zips = []
lats = []
longs = []
input_locations = []
input_location_dists = []
phones = []
urls = []
metro_names = []
metro_rec = []
metro_ranks = []
metro_tied_flags = []
region_names = []
region_rec = []
region_ranks = []
region_tied_flags = []
state_rec = []
state_ranks = []
state_tied_flags = []
high_perf_adult = []
national_ranks_adult = []
national_ranks_peds = []
updated = []

#r = requests.get(    "https://httpbin.io/user-agent",    timeout=10)

#print(r.status_code)
#print(r.json())

data_df = pd.read_csv(
    "Datasets/realtor-data.zip.csv",
    dtype={"zip_code": str},
)
data_df = data_df[(data_df["state"] != "Puerto Rico") & (data_df["state"] != "Virgin Islands") 
                  & (data_df["state"] != "Guam")]
data_df = data_df.dropna(subset=['brokered_by', 'status', 'price','bed','bath','acre_lot','street','city','state','zip_code','house_size'])

crime_risk_df = pd.read_csv(
    filepath_or_buffer="Datasets/(Copy) VT/usa_zi_premium_crimerisk.csv",
    dtype={"ID": str},
)
crime_risk_df.set_index("ID", inplace=True)

crime_risk_stats_df = pd.read_csv(
    "Datasets/(Copy) VT/usa_zi_2024 crimerisk-statistics.csv"
)
recent_demo_df = pd.read_csv("Datasets/(Copy) VT/usa_zi_2024_demographic_estimates.csv",dtype={"ID": str})
recent_demo_df.set_index("ID", inplace=True)
zi_base_current_df = pd.read_csv(
    "Datasets/(Copy) VT/usa_zi_base_currentyear.csv", dtype={"ID": str}
)
zi_base_current_df.set_index("ID", inplace=True)
base_current_stats_df = pd.read_csv(
    "Datasets/(Copy) VT/usa_zi_base_currentyear-statistics.csv"
)
base_current_stats_df.set_index("VARIABLE", inplace=True)
max_total_crime = crime_risk_stats_df.loc[9].MAXIMUM
min_total_crime = crime_risk_stats_df.loc[9].MINIMUM

# Get min and max housing price
min_med_housing_value = data_df.price.min()
max_med_housing_value = data_df.price.max()

# Get square foot min and max
sq_ft_max = data_df.house_size.max()
sq_ft_min = data_df.house_size.min()
temp_stat_df = pd.read_csv(
    "Datasets/(Copy) VT/usa_zi_premium_environment-statistics.csv"
)
temp_stat_df.set_index("VARIABLE", inplace=True)
temp_av_max = temp_stat_df.loc["TMPAVEANN"].MAXIMUM
temp_av_min = temp_stat_df.loc["TMPAVEANN"].MINIMUM

temp_df = pd.read_csv(
    "Datasets/(Copy) VT/usa_zi_premium_environment.csv", dtype={"ID": str}
)
temp_df.set_index("ID", inplace=True)

#one-hot encoding for state
statedf = pd.DataFrame({'state': data_df["state"]})
state_ohe = pd.get_dummies(statedf, dtype=int)
data_df = pd.concat([data_df, state_ohe], axis='columns')
data_df.drop(labels=['state'], inplace=True, axis='columns')
# Get the data for zip codes
def calculatesuitability(tablerow):
    zipcode = tablerow.zip_code
    # get square footage
    size_in_sq_ft = tablerow.house_size
    sq_ft_weighted_score = (
        (sq_ft_max - size_in_sq_ft) / (sq_ft_max - sq_ft_min)
    ) * sq_ft_wt
    # Get price
    price = tablerow.price
    price_weighted_score = (
        (max_med_housing_value - price)
        / (max_med_housing_value - min_med_housing_value)
    ) * home_price_wt
    # temp score: take annual temperatures
    avg_temp = temp_df.loc[zipcode].TMPAVEANN
    temp_weighted_score = (
        (avg_temp - temp_av_min) / (temp_av_max - temp_av_min)
    ) * wthr_temp_wt

    # Get crime rate total
    total_crime = crime_risk_df.loc[zipcode].CRMPYTOTC
    crime_weighted_score = (
        (max_total_crime - total_crime) / (max_total_crime - min_total_crime)
    ) * crime_wt
    total_weighted_score = (
        sq_ft_weighted_score
        + price_weighted_score
        + temp_weighted_score
        + crime_weighted_score
    )
    return total_weighted_score

max_sq_ft = data_df["house_size"].max()
min_sq_ft = data_df["house_size"].min()
max_price = data_df["price"].max()
min_price = data_df["price"].min()
# Try it with proprties from New York
#new_york_real_estate = data_df.loc[(data_df.state == "New York")]
#new_york_real_estate.reset_index()

senior_suitability = np.zeros(data_df.shape[0])
data_df["Suitability"] = senior_suitability
data_df["crime"] = np.zeros(data_df.shape[0])
data_df["weather"] = np.zeros(data_df.shape[0])

print("Going thru housing data")
# Go thru zip codes instead
for idx, row in zi_base_current_df.iterrows():
    # Get properties with the given zip code
    zipcode = idx
    houses_at_zip = data_df.loc[data_df["zip_code"] == zipcode]
    crime = crime_risk_df.loc[zipcode].CRMPYTOTC
    temp = temp_df.loc[zipcode].TMPAVEANN
    weather_rsk = temp_df.loc[zipcode].RSKCYRISK

    if houses_at_zip.shape[0] > 0:
        # Get max price
        #max_price = houses_at_zip["price"].max()
        #min_price = houses_at_zip["price"].min()
        #sq_ft_max = houses_at_zip["house_size"].max()
        #sq_ft_min = houses_at_zip["house_size"].min()
        avg_temp = temp_df.loc[zipcode].TMPAVEANN
        temp_weighted_score = (
            (avg_temp - temp_av_min) / (temp_av_max - temp_av_min)
        ) * wthr_temp_wt
        total_crime = crime_risk_df.loc[zipcode].CRMPYTOTC
        crime_weighted_score = (
            (max_total_crime - total_crime) / (max_total_crime - min_total_crime)
        ) * crime_wt

        for idx, house in houses_at_zip.iterrows():
            house_price = house.price
            price_weighted_score = (
                (max_price - house_price) / (max_price - min_price)
            ) * home_price_wt if (max_price > min_price) else home_price_wt

            sq_footage = house.house_size
            footage_weighted_score = (
                (max_sq_ft - sq_footage) / (max_sq_ft - min_sq_ft)
            ) * sq_ft_wt if (max_sq_ft > min_sq_ft) else sq_ft_wt
            
            total_weighted_score = (
                price_weighted_score
                + footage_weighted_score
                + crime_weighted_score
                + temp_weighted_score + weather_rsk
            )/5
            data_df.at[idx,"weather_risk"] = weather_rsk
            data_df.at[idx,"crime_risk"] =crime
            data_df.at[idx,"crime"] = total_crime
            data_df.at[idx,"avg_temp"] = avg_temp
            data_df.at[idx, "Suitability"] = total_weighted_score
            data_df.at[idx,"particulate_matter"] = temp_df.loc[zipcode].PM10
            data_df.at[idx,"carbon_monoxide"] =temp_df.loc[zipcode].CARBMONO
            data_df.at[idx,"walkability"] =temp_df.loc[zipcode].SLD22WALK
            data_df.at[idx,"Nitrogen Dioxde"] =temp_df.loc[zipcode].NO2
            data_df.at[idx,"Median Cash Rent"] = recent_demo_df.loc[zipcode].RNTX4MED
            data_df.at[idx,"Health Care"] = zi_base_current_df.loc[zipcode].XCYHLT

data_df.to_csv("Suitability_score_house.csv")
data_df.dropna(inplace=True)
lin_reg = LinearRegression()
# remove 
#Note to self: run dropna() before fitting
X = data_df.drop(["status", "city", "brokered_by", "zip_code", 'prev_sold_date'], axis='columns')
Y = data_df["Suitability"]
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.3, random_state=42, shuffle=True)

lin_reg.fit(train_x, train_y)
y_pred_lin = lin_reg.predict(test_x)

root_mse = root_mean_squared_error(test_y, y_pred_lin)
r2_lin_pred = r2_score(test_y, y_pred_lin)
print("Score report:")
print("Root MSE: ", root_mse)
print("R-squared: ", r2_lin_pred)