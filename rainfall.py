import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans, DBSCAN

st.set_page_config(layout="wide")

st.title("Rainfall Analysis and Prediction in India")

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

# EDA
st.header("Exploratory Data Analysis")

yearly_avg = df.groupby('YEAR')['Avg_Jun_Sep'].mean()
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(yearly_avg.index, yearly_avg.values, color='teal', marker='o', linestyle='-', linewidth=2)
ax.set_title('Average Rainfall in India (JUN-SEP) by Year')
ax.set_xlabel('Year')
ax.set_ylabel('Avg Rainfall (mm)')
ax.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig)

st.subheader("Rainfall per Subdivision and Year")
sub = st.selectbox("Select a subdivision:", df['subdivision'].unique())
a = df[df['subdivision'] == sub]

yr = st.selectbox("Select a year:", sorted(a['YEAR'].unique()))
b = a[a['YEAR'] == yr]

if not b.empty:
    months = ['JUN', 'JUL', 'AUG', 'SEP']
    rainfall_values = b[months].values.flatten()
    st.write(f"Rainfall in {sub} during {yr}:")
    for m, val in zip(months, rainfall_values):
        st.write(f"{m}: {val} mm")
    fig, ax = plt.subplots()
    sns.barplot(x=months, y=rainfall_values, palette="Set2", ax=ax)
    ax.set_title(f'Rainfall in {sub} - {yr} (JUN–SEP)')
    ax.set_ylabel('Rainfall (mm)')
    ax.set_xlabel('Month')
    st.pyplot(fig)

st.subheader("Distribution of Rainfall")
fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(data=df, x='subdivision', y='Avg_Jun_Sep', palette="pastel", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Distribution of Avg Rainfall by Subdivision')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df['Avg_Jun_Sep'], bins=30, kde=True, color='skyblue', ax=ax)
ax.set_title('Histogram of Average Rainfall (JUN-SEP)')
ax.set_xlabel('Avg Rainfall')
st.pyplot(fig)

st.subheader("Rainfall Heatmap")
start_year = st.number_input("Enter the start year:", min_value=int(df['YEAR'].min()), max_value=int(df['YEAR'].max()), value=2000)
end_year = st.number_input("Enter the end year:", min_value=int(df['YEAR'].min()), max_value=int(df['YEAR'].max()), value=2010)
b = a[(a['YEAR'] >= start_year) & (a['YEAR'] <= end_year)]
if not b.empty:
    b = b[['YEAR', 'JUN', 'JUL', 'AUG', 'SEP']].set_index('YEAR')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(b, annot=True, cmap='coolwarm', fmt=".1f", cbar_kws={'label': 'Rainfall (mm)'}, ax=ax)
    ax.set_title(f"Rainfall Heatmap (JUN–SEP)\n{sub} [{start_year}–{end_year}]")
    st.pyplot(fig)

# Machine Learning
st.header("Rainfall Prediction")
x = df[['JUN', 'JUL', 'AUG', 'SEP', 'Avg_Jun_Sep', 'Lag1_Avg']].values
y = df['YoY_Change'].values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

rmse = mean_squared_error(ytest, ypred, squared=False)
mae = mean_absolute_error(ytest, ypred)

st.subheader("Model Evaluation")
st.metric("RMSE", f"{rmse:.2f}")
st.metric("MAE", f"{mae:.2f}")

# Clustering
st.header("Clustering Analysis")
X_cluster = df[['Avg_Jun_Sep', 'YoY_Change']]

wss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_cluster)
    wss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), wss, marker='o', linestyle='-', color='crimson')
ax.set_title('Elbow Method')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('WSS')
st.pyplot(fig)

kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X_cluster)
X_cluster['Cluster'] = y_kmeans
centers = kmeans.cluster_centers_

fig, ax = plt.subplots()
scatter = ax.scatter(X_cluster['Avg_Jun_Sep'], X_cluster['YoY_Change'], c=X_cluster['Cluster'], cmap='tab10', alpha=0.7)
ax.scatter(centers[:, 0], centers[:, 1], c='black', s=300, marker='X', label='Centroids')
ax.set_title('KMeans Clustering')
ax.set_xlabel('Avg Rainfall (Jun-Sep)')
ax.set_ylabel('YoY Change')
ax.legend()
st.pyplot(fig)

# DBSCAN
st.subheader("DBSCAN Clustering")
dbscan = DBSCAN(eps=1.0, min_samples=5)
y_dbscan = dbscan.fit_predict(X_cluster[['Avg_Jun_Sep', 'YoY_Change']])

fig, ax = plt.subplots()
colors = sns.color_palette("husl", len(set(y_dbscan)))
for i, cluster in enumerate(set(y_dbscan)):
    mask = (y_dbscan == cluster)
    ax.scatter(X_cluster['Avg_Jun_Sep'][mask], X_cluster['YoY_Change'][mask], 
               label=f"Cluster {cluster}" if cluster != -1 else "Noise",
               s=50, alpha=0.7, color=colors[i])
ax.set_title('DBSCAN Clustering')
ax.set_xlabel('Average Rainfall (Jun-Sep)')
ax.set_ylabel('Year-over-Year Change')
ax.legend()
st.pyplot(fig)

# Trend Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['YEAR'], df['Avg_Jun_Sep'], label='Avg Rainfall (Jun-Sep)', color='navy', marker='o')
ax.set_title('Trend of Average Rainfall (Jun-Sep) Over Years')
ax.set_xlabel('Year')
ax.set_ylabel('Avg Rainfall (mm)')
ax.grid(True)
ax.legend()
st.pyplot(fig)
