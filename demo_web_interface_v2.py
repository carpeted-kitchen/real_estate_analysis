import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("/Users/katherineskarda/PycharmProjects/IntroToAI/real_estate_analysis/Suitability_score_house.csv")

# Prepare model
X = df.drop(["status", "city", "brokered_by", "zip_code", "prev_sold_date"], axis=1)
y = df["Suitability"]

model = LinearRegression()
model.fit(X, y)

# UI
st.title("🏡 Retirement Home Finder")

budget = st.slider("Max Budget ($)", 50000, 1000000, 450000)
zipcode = st.text_input("Preferred Zipcode (optional)")


# only works with budget change, not for zipcode at the moment
def recommend(df, budget, zipcode):
    subset = df[df["price"] <= budget].copy()

    if zipcode:
        subset = subset[subset["zip_code"] == zipcode]

    if subset.empty:
        return None

    subset["pred"] = model.predict(subset[X.columns])
    return subset.sort_values("pred", ascending=False).head(5)


# Run recommendation, results not visually appeally yet
if st.button("Find Homes"):
    results = recommend(df, budget, zipcode)

    if results is None:
        st.write("No matching homes found.")
    else:
        st.write("### Top Recommendations")
        st.dataframe(results[["price", "weather", "pred"]])
