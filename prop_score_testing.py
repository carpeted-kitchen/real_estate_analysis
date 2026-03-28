import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta
import time
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score,mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats

home_price_wt = 0.5
weather_wt = 0.3
crime_wt = 0.1
sq_ft_wt = 0.1

training_data = pd.read_csv("Suitability_score_house.csv")

lin_reg = LinearRegression()

X = training_data.drop(["status", "city", "brokered_by", "zip_code", 'prev_sold_date'], axis='columns')
Y = training_data["Suitability"]
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.3, random_state=42, shuffle=True)

lin_reg.fit(train_x, train_y)
y_pred_lin = lin_reg.predict(test_x)

root_mse = root_mean_squared_error(test_y, y_pred_lin)
r2_lin_pred = r2_score(test_y, y_pred_lin)
res = stats.spearmanr(test_y, y_pred_lin)
print("Score report:")
print("Root MSE: ", root_mse)
print("R-squared: ", r2_lin_pred)
print("Spearman correlation: ", res.correlation)
print("Spearman significance: ", res.pvalue)


# below is from ChatGPT for Add Baseline Comparison
# Since your rule-based baseline = Suitability, simulate comparison:
baseline_preds = (
    -0.5 * test_x["price"] + # negative to rank cheaper houses higher
    0.3 * test_x["weather"] +
    -0.1 * test_x["crime"] + # negative to rank low crime houses higher
    0.1 * test_x["house_size"]
)

ml_corr = stats.spearmanr(test_y, y_pred_lin).correlation
baseline_corr = stats.spearmanr(test_y, baseline_preds).correlation

print("ML Spearman correlation:", ml_corr)
print("Baseline Spearman:", baseline_corr)

# from ChatGPT for Add Simple Filtering Baseline for evaluation
filtered = training_data[
    (training_data["price"] <= 450000) &
    (training_data["crime"] < training_data["crime"].quantile(0.5))
]
# --- Filtering baseline (on test set) ---
test_df = test_x.copy()
test_df["true"] = test_y

filtered_test = test_df[
    (test_df["price"] <= 450000) &
    (test_df["crime"] < test_df["crime"].quantile(0.5))
].copy()

if len(filtered_test) > 5:
    filtered_test["filter_score"] = (
        -filtered_test["price"] +
        -filtered_test["crime"] +
        filtered_test["weather"] +
        filtered_test["house_size"]
    )
    filter_corr = stats.spearmanr(
        filtered_test["true"],
        filtered_test["filter_score"]
    ).correlation

    print("Filtering Spearman:", filter_corr)
else:
    print("Filtering baseline: Not enough data")

# filtered_top = filtered.sort_values("Suitability", ascending=False).head(5)
# print("Filtering baseline top 5:")
# print(filtered_top[["price", "weather", "crime", "Suitability"]])

# below is from ChatGPT for Add Top-K Recommendation Evaluation
df_test = test_x.copy()
df_test["true"] = test_y
df_test["pred"] = y_pred_lin

top_pred = df_test.sort_values("pred", ascending=False).head(5)
top_true = df_test.sort_values("true", ascending=False).head(5)

overlap = len(set(top_pred.index) & set(top_true.index))
print("Top-5 overlap:", overlap)

# from ChatGPT to output top 5 recommendations
def recommend(user_budget):
    subset = training_data[training_data["price"] <= user_budget]
    subset["pred"] = lin_reg.predict(subset[X.columns])
    return subset.sort_values("pred", ascending=False).head(5)


results = recommend(450000)

print("Top 5 recommendations:")
print(results[["price", "weather", "crime", "pred"]])
