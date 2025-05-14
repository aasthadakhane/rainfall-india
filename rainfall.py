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

st.set_page_config(layout="wide")
st.title("🌧️ Rainfall Analysis and Prediction in India")

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

# ---------------- EDA ----------------
st.header("📊 Exploratory Data Analysis")
df['subdivision_encoded'] = df['subdivision'].astype('category').cat.codes

# Line plot: Yearly trend
yearly_avg = df.groupby('YEAR')['Avg_Jun_Sep'].mean()
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(yearly_avg.index, yearly_avg.values)
ax.set_title('Average Rainfall in India (JUN-SEP) by Year')
ax.set_xlabel('Year')
ax.set_ylabel('Avg Rainfall (mm)')
ax.grid(True)
st.pyplot(fig)

# Subdivision-based rainfall
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
    sb.barplot(x=months, y=rainfall_values, palette="Blues_d", ax=ax)
    ax.set_title(f'Rainfall in {sub} - {yr} (JUN–SEP)')
    ax.set_ylabel('Rainfall (mm)')
    ax.set_xlabel('Month')
    st.pyplot(fig)

# Boxplot: Distribution by subdivision
st.subheader("Distribution of Rainfall")
fig, ax = plt.subplots(figsize=(16,8))
sb.boxplot(data=df, x='subdivision', y='Avg_Jun_Sep', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Distribution of Avg Rainfall by Subdivision')
st.pyplot(fig)

# Histogram
fig, ax = plt.subplots(figsize=(8,5))
sb.histplot(df['Avg_Jun_Sep'], bins=30, kde=True, ax=ax)
ax.set_title('Histogram of Average Rainfall (JUN-SEP)')
ax.set_xlabel('Avg Rainfall')
st.pyplot(fig)

# Heatmap
st.subheader("Rainfall Heatmap")
start_year = st.number_input("Start Year", min_value=int(df['YEAR'].min()), max_value=int(df['YEAR'].max()), value=2000)
end_year = st.number_input("End Year", min_value=int(df['YEAR'].min()), max_value=int(df['YEAR'].max()), value=2010)
b = a[(a['YEAR'] >= start_year) & (a['YEAR'] <= end_year)]
if not b.empty:
    b = b[['YEAR', 'JUN', 'JUL', 'AUG', 'SEP']].set_index('YEAR')
    fig, ax = plt.subplots(figsize=(8, 5))
    sb.heatmap(b, annot=True, cmap='YlGnBu', fmt=".1f", cbar_kws={'label': 'Rainfall (mm)'}, ax=ax)
    ax.set_title(f"Rainfall Heatmap (JUN–SEP)\n{sub} [{start_year}–{end_year}]")
    st.pyplot(fig)

# ---------------- ML ----------------
st.header("🤖 Rainfall Prediction")

# Encode categorical
label = LabelEncoder()
df['subdivision'] = label.fit_transform(df['subdivision'])

# Prepare data
features = df.drop(['YoY_Change', 'subdivision', 'YEAR'], axis=1)
x = features.values
y = df['YoY_Change'].values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Model selection
model_name = st.selectbox("Choose a regression model", ["Linear Regression", "Random Forest", "AdaBoost"])

if model_name == "Linear Regression":
    model = LinearRegression()
elif model_name == "Random Forest":
    model = RandomForestRegressor()
else:
    model = AdaBoostRegressor(n_estimators=30, random_state=7)

model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

# Evaluation
r2 = r2_score(ytest, ypred)
rmse = mean_squared_error(ytest, ypred, squared=False)
mae = mean_absolute_error(ytest, ypred)

st.subheader("📈 Model Evaluation Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.3f}")
col2.metric("RMSE", f"{rmse:.3f}")
col3.metric("MAE", f"{mae:.3f}")

# Cross-validation
cv_score = cross_val_score(model, x, y, cv=5, scoring='r2').mean()
st.write(f"Cross-validated R² Score: {cv_score:.3f}")

# Feature importance
if hasattr(model, 'feature_importances_'):
    st.subheader("Feature Importances")
    importances = pd.Series(model.feature_importances_, index=features.columns)
    fig, ax = plt.subplots()
    importances.sort_values().plot(kind='barh', ax=ax)
    ax.set_title('Feature Importance')
    st.pyplot(fig)

# Download predictions
if st.button("Download Predictions as CSV"):
    result_df = pd.DataFrame({'Actual': ytest, 'Predicted': ypred})
    st.download_button("Download", result_df.to_csv(index=False), file_name='predictions.csv')

# ---------------- Clustering ----------------
st.header("📍 Clustering Analysis")
X_cluster = df[['Avg_Jun_Sep', 'YoY_Change']]

# Elbow method
wss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_cluster)
    wss.append(kmeans.inertia_)
fig, ax = plt.subplots()
ax.plot(range(1, 11), wss, marker='o')
ax.set_title('Elbow Method for Optimal Clusters')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('WSS')
st.pyplot(fig)

# KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X_cluster)
X_cluster_with_labels = X_cluster.copy()
X_cluster_with_labels['Cluster'] = y_kmeans
centers = kmeans.cluster_centers_

fig, ax = plt.subplots()
scatter = ax.scatter(
    X_cluster_with_labels['Avg_Jun_Sep'], 
    X_cluster_with_labels['YoY_Change'], 
    c=X_cluster_with_labels['Cluster'], cmap='viridis', alpha=0.7
)
ax.scatter(centers[:, 0], centers[:, 1], c='red', s=300, marker='X', label='Centroids')
ax.set_title('KMeans Clustering')
ax.set_xlabel('Avg Rainfall (Jun-Sep)')
ax.set_ylabel('YoY Change')
ax.legend()
st.pyplot(fig)

# DBSCAN
dbscan = DBSCAN(eps=1.0, min_samples=5)
y_dbscan = dbscan.fit_predict(X_cluster)
fig, ax = plt.subplots()
ax.scatter(X_cluster['Avg_Jun_Sep'], X_cluster['YoY_Change'], c=y_dbscan, cmap='viridis')
ax.scatter(X_cluster['Avg_Jun_Sep'][y_dbscan == -1], X_cluster['YoY_Change'][y_dbscan == -1],
           c='red', s=100, label='Noise', marker='x')
ax.set_title('DBSCAN Clustering')
ax.set_xlabel('Average Rainfall (Jun-Sep)')
ax.set_ylabel('Year-over-Year Change')
ax.legend()
st.pyplot(fig)

# Trend over years
st.header("📉 Rainfall Trend")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['YEAR'], df['Avg_Jun_Sep'], label='Avg Rainfall (Jun-Sep)', color='b', marker='o')
ax.set_title('Trend of Average Rainfall (Jun-Sep) Over Years')
ax.set_xlabel('Year')
ax.set_ylabel('Avg Rainfall (mm)')
ax.grid(True)
ax.legend()
st.pyplot(fig)
