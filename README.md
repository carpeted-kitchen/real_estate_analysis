## Prerequisites

## Make sure git LFS (large file storage) is installed:
on Mac:
brew install git-lfs

## Python Packages required:
1. scikit-learn
2. scipy.stats
3. Pandas
4. Streamlit
4. Numpy

---

## Datasets
USA Real Estate
https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset

Included in the Datasets folder, along with the AGS dataset for use by VT

---

## Running re_retirement_rank.py
This Python script is used for preprocessing the data and compute senior suitability score for the properties in our datasets.
It outputs the transformed dataset to Suitability_score_house.csv (this file is already populated in the repository)

Note that this script may take a few hours to run.

### To Run:
python re_retirement_rank.py

## Running Demo App

streamlit run Path\To\Repository\real_estate_analysis\demo_app.py

## Running recommender_knn.py
python Path\To\Repo\Real_Estate_Analysis\recommender_knn.py
