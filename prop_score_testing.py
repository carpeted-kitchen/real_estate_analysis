import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta
import time
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score,mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats

lin_reg = LinearRegression()

training_data = pd.read_csv("Suitability_score_house.csv")
# to fix nan error
# Drop rows where ANY of your scoring metrics are missing
training_data.dropna(inplace=True)

# This original training set matches data too closely
# X = training_data.drop(["status", "city", "brokered_by", "zip_code", 'prev_sold_date'], axis='columns')

# Ensure 'street' is dropped from X (LinearRegression only eats numbers)
# and DROP the variables that were the 'strongest' drivers of the formula
# to see if the model can still rank houses accurately.
drop_cols = [
    "status", "city", "brokered_by", "zip_code", "prev_sold_date",
    "street", "Suitability",
    "Health Care", "Nitrogen Dioxde", "particulate_matter" # Hidden features
]

X = training_data.drop(columns=[c for c in drop_cols if c in training_data.columns])
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

# --- Improved Rule-Based Baseline ---
def comprehensive_rule_baseline(df):
    def norm_neg(col):  # Lower is better
        return (df[col].max() - df[col]) / (df[col].max() - df[col].min() + 1e-9)

    def norm_pos(col):  # Higher is better
        return (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-9)

    # Replicating the 11-factor sum from re_retirement_rank.py
    score = (
                    norm_neg("price") +
                    norm_neg("house_size") +
                    norm_neg("crime_risk") +
                    norm_pos("avg_temp") +
                    norm_neg("weather_risk") +
                    norm_neg("carbon_monoxide") +
                    norm_neg("Median Cash Rent") +
                    norm_pos("walkability") 
            ) / 11
    return score


# use the original data for test_df, matched to test_x indices
test_df = training_data.loc[test_x.index].copy()
test_df["comp_baseline"] = comprehensive_rule_baseline(test_df)
rule_corr, rule_p = stats.spearmanr(test_y, test_df["comp_baseline"])

print(f"Comprehensive Rule-Based Spearman: {rule_corr:.4f} (p-value: {rule_p:.4e})")

# --- Improved Filtering Baseline ---
def comprehensive_filtering_baseline(df):
    # Calculate the full suitability score for everyone
    full_scores = comprehensive_rule_baseline(df)

    # Define "Dealbreakers"
    # High Price or High Healthcare Index (since code treats high XCYHLT as bad)
    price_cutoff = df["price"].quantile(0.75)
    health_cutoff = df["Health Care"].quantile(0.75)

    # Apply Filter
    passes_filter = (df["price"] <= price_cutoff) & (df["Health Care"] <= health_cutoff)

    # Survivors keep their score, failures get 0
    df_result = pd.Series(0.0, index=df.index)
    df_result[passes_filter] = full_scores[passes_filter]

    return df_result


test_df["comp_filter"] = comprehensive_filtering_baseline(test_df)
filt_corr, filt_p = stats.spearmanr(test_y, test_df["comp_filter"])

print(f"Comprehensive Filtering Spearman: {filt_corr:.4f} (p-value: {filt_p:.4e})")

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
