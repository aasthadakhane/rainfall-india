import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans, DBSCAN

st.set_page_config(page_title="Rainfall Analysis India", layout="wide")

# Title
st.title("Rainfall Analysis and Prediction in India")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("rainfaLLIndia.csv")
    df['Avg_Jun_Sep'] = df[['JUN', 'JUL', 'AUG', 'SEP']].mean(axis=1)
    df.sort_values(by=['subdivision', 'YEAR'], inplace=True)
    df['YoY_Change'] = df.groupby('subdivision')['Avg_Jun_Sep'].diff()
    df['Lag1_Avg'] = df.groupby('subdivision')['Avg_Jun_Sep'].shift(1)
    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)
    return df

df = load_data()

# Sidebar selection
st.sidebar.title("Select Options")
sub = st.sidebar.selectbox("Select Subdivision", df['subdivision'].unique())
a = df[df['subdivision'] == sub]

# EDA
st.header("Exploratory Data Analysis")

yearly_avg = df.groupby('YEAR')['Avg_Jun_Sep'].mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(yearly_avg.index, yearly_avg.values, marker='o')
ax.set_title("Average Rainfall in India (JUN–SEP) by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Avg Rainfall (mm)")
ax.grid(True)
st.pyplot(fig)

# Subdivision Yearly Rainfall
st.subheader("Monthly Rainfall for Selected Year")
yr = st.sidebar.selectbox("Select Year", sorted(a['YEAR'].unique()))
b = a[a['YEAR'] == yr]
if not b.empty:
    months = ['JUN', 'JUL', 'AUG', 'SEP']
    rainfall_values = b[months].values.flatten()
    st.write(f"Rainfall in **{sub}** during **{yr}**:")
    fig, ax = plt.subplots()
    sb.barplot(x=months, y=rainfall_values, palette="Blues_d", ax=ax)
    ax.set_title(f"Rainfall in {sub} - {yr} (JUN–SEP)")
    ax.set_ylabel("Rainfall (mm)")
    st.pyplot(fig)

# Boxplot
st.subheader("Rainfall Distribution by Subdivision")
fig, ax = plt.subplots(figsize=(16, 8))
sb.boxplot(data=df, x='subdivision', y='Avg_Jun_Sep', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

# Histogram
st.subheader("Histogram of Average Rainfall (JUN–SEP)")
fig, ax = plt.subplots(figsize=(8, 5))
sb.histplot(df['Avg_Jun_Sep'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Heatmap
st.subheader("Rainfall Heatmap")
start_year = st.slider("Start Year", int(df['YEAR'].min()), int(df['YEAR'].max()), 2000)
end_year = st.slider("End Year", start_year, int(df['YEAR'].max()), 2010)
b_range = a[(a['YEAR'] >= start_year) & (a['YEAR'] <= end_year)]
if not b_range.empty:
    b_heat = b_range[['YEAR', 'JUN', 'JUL', 'AUG', 'SEP']].set_index('YEAR')
    fig, ax = plt.subplots(figsize=(8, 5))
    sb.heatmap(b_heat, annot=True, cmap='YlGnBu', fmt=".1f", cbar_kws={'label': 'Rainfall (mm)'}, ax=ax)
    ax.set_title(f"Rainfall Heatmap (JUN–SEP) - {sub} [{start_year}–{end_year}]")
    st.pyplot(fig)

# Machine Learning
st.header("Rainfall Prediction using Machine Learning")
df_ml = df.copy()
label = LabelEncoder()
df_ml['subdivision'] = label.fit_transform(df_ml['subdivision'])

X = df_ml.drop(['YoY_Change', 'subdivision', 'YEAR'], axis=1)
y = df_ml['YoY_Change']

xtrain, xtest, ytrain, ytest = train_test_split(X,
