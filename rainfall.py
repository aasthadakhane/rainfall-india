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

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(xtrain, ytrain)
lr_pred = lr_model.predict(xtest)
lr_r2 = r2_score(ytest, lr_pred)
st.write(f"Linear Regression R² Score: {lr_r2 * 100:.2f}%")

# Random Forest Model
rf_model = RandomForestRegressor()
rf_model.fit(xtrain, ytrain)
rf_pred = rf_model.predict(xtest)
rf_r2 = r2_score(ytest, rf_pred)
st.write(f"Random Forest R² Score: {rf_r2 * 100:.2f}%")

# Cross-validation for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'bootstrap': [True]
}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='r2')
grid_search.fit(xtrain, ytrain)
st.write(f"Best RandomForest Parameters: {grid_search.best_params_}")
st.write(f"Best RandomForest R² Score: {grid_search.best_score_ * 100:.2f}%")

# AdaBoost Model
kfold = KFold(n_splits=20, shuffle=True, random_state=7)
adb_model = AdaBoostRegressor(n_estimators=30, random_state=7)
adb_results = cross_val_score(adb_model, X, y, cv=kfold, scoring='r2')
st.write(f"AdaBoost R² Score: {adb_results.mean() * 100:.2f}%")

# Evaluation metrics
rmse = mean_squared_error(ytest, rf_pred, squared=False)
mae = mean_absolute_error(ytest, rf_pred)
st.subheader("Model Evaluation Metrics")
st.metric("RMSE", f"{rmse:.4f}")
st.metric("MAE", f"{mae:.4f}")

# Clustering
st.header("Clustering Analysis")

# KMeans
X_cluster = df[['Avg_Jun_Sep', 'YoY_Change']]
wss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_cluster)
    wss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), wss, marker='o')
ax.set_title('Elbow Method for KMeans Clustering')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('WSS')
st.pyplot(fig)

# Fit KMeans with optimal clusters
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X_cluster)
X_cluster_with_labels = X_cluster.copy()
X_cluster_with_labels['Cluster'] = y_kmeans

# Plot KMeans
centers = kmeans.cluster_centers_
fig, ax = plt.subplots()
scatter = ax.scatter(X_cluster_with_labels['Avg_Jun_Sep'], X_cluster_with_labels['YoY_Change'], c=X_cluster_with_labels['Cluster'], cmap='viridis', alpha=0.7)
ax.scatter(centers[:, 0], centers[:, 1], c='red', s=300, marker='X', label='Centroids')
ax.set_title('KMeans Clustering')
ax.set_xlabel('Avg Rainfall (Jun-Sep)')
ax.set_ylabel('YoY Change')
ax.legend()
st.pyplot(fig)

# DBSCAN
dbscan = DBSCAN(eps=1.0, min_samples=5)
y_dbscan =_
