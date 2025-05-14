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

st.set_page_config(layout="wide", page_title="🌦️ Rainfall Analysis & Prediction App")
st.title("🌧️ Rainfall Analysis and Prediction in India")
st.markdown("""
<style>
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-1629p8f {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-10trblm {
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

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

# Ask user question about analysis preference
st.sidebar.header("🎛️ Customize Your Analysis")
section = st.sidebar.radio("Which section would you like to explore?", ["EDA", "ML Prediction", "Clustering", "Rainfall Trend"])

# ---------------- EDA ----------------
if section == "EDA":
    st.header("📊 Exploratory Data Analysis")
    df['subdivision_encoded'] = df['subdivision'].astype('category').cat.codes

    # Line plot: Yearly trend
    yearly_avg = df.groupby('YEAR')['Avg_Jun_Sep'].mean()
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(yearly_avg.index, yearly_avg.values, color='#1f77b4')
    ax.set_title('📈 Average Rainfall in India (JUN-SEP) by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Avg Rainfall (mm)')
    ax.grid(True)
    st.pyplot(fig)

    # Subdivision-based rainfall
    st.subheader("📍 Rainfall per Subdivision and Year")
    sub = st.selectbox("Select a subdivision:", df['subdivision'].unique())
    a = df[df['subdivision'] == sub]
    yr = st.selectbox("Select a year:", sorted(a['YEAR'].unique()))
    b = a[a['YEAR'] == yr]

    if not b.empty:
        months = ['JUN', 'JUL', 'AUG', 'SEP']
        rainfall_values = b[months].values.flatten()
        st.markdown(f"### ☔ Rainfall in `{sub}` during `{yr}`")
        fig, ax = plt.subplots()
        sb.barplot(x=months, y=rainfall_values, palette="coolwarm", ax=ax)
        ax.set_title(f'Rainfall in {sub} - {yr} (JUN–SEP)')
        ax.set_ylabel('Rainfall (mm)')
        ax.set_xlabel('Month')
        st.pyplot(fig)

    # Boxplot: Distribution by subdivision
    st.subheader("📦 Distribution of Rainfall")
    fig, ax = plt.subplots(figsize=(16,8))
    sb.boxplot(data=df, x='subdivision', y='Avg_Jun_Sep', palette="Spectral", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title('Distribution of Avg Rainfall by Subdivision')
    st.pyplot(fig)

    # Histogram
    fig, ax = plt.subplots(figsize=(8,5))
    sb.histplot(df['Avg_Jun_Sep'], bins=30, kde=True, color='#2ca02c', ax=ax)
    ax.set_title('Histogram of Average Rainfall (JUN-SEP)')
    ax.set_xlabel('Avg Rainfall')
    st.pyplot(fig)

    # Heatmap
    st.subheader("🌡️ Rainfall Heatmap")
    start_year = st.slider("Select Start Year", int(df['YEAR'].min()), int(df['YEAR'].max()), value=2000)
    end_year = st.slider("Select End Year", int(df['YEAR'].min()), int(df['YEAR'].max()), value=2010)
    b = a[(a['YEAR'] >= start_year) & (a['YEAR'] <= end_year)]
    if not b.empty:
        b = b[['YEAR', 'JUN', 'JUL', 'AUG', 'SEP']].set_index('YEAR')
        fig, ax = plt.subplots(figsize=(8, 5))
        sb.heatmap(b, annot=True, cmap='YlOrBr', fmt=".1f", cbar_kws={'label': 'Rainfall (mm)'}, ax=ax)
        ax.set_title(f"Rainfall Heatmap (JUN–SEP)\n{sub} [{start_year}–{end_year}]")
        st.pyplot(fig)

# ---------------- ML ----------------
elif section == "ML Prediction":
    st.header("🤖 Rainfall Prediction")
    label = LabelEncoder()
    df['subdivision'] = label.fit_transform(df['subdivision'])
    features = df.drop(['YoY_Change', 'subdivision', 'YEAR'], axis=1)
    x = features.values
    y = df['YoY_Change'].values
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    model_name = st.selectbox("Choose a regression model", ["Linear Regression", "Random Forest", "AdaBoost"])
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor()
    else:
        model = AdaBoostRegressor(n_estimators=30, random_state=7)

    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)

    r2 = r2_score(ytest, ypred)
    rmse = mean_squared_error(ytest, ypred, squared=False)
    mae = mean_absolute_error(ytest, ypred)

    st.subheader("📈 Model Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("RMSE", f"{rmse:.3f}")
    col3.metric("MAE", f"{mae:.3f}")

    cv_score = cross_val_score(model, x, y, cv=5, scoring='r2').mean()
    st.success(f"✅ Cross-validated R² Score: {cv_score:.3f}")

    if hasattr(model, 'feature_importances_'):
        st.subheader("🔍 Feature Importances")
        importances = pd.Series(model.feature_importances_, index=features.columns)
        fig, ax = plt.subplots()
        importances.sort_values().plot(kind='barh', ax=ax, color='#ff7f0e')
        ax.set_title('Feature Importance')
        st.pyplot(fig)

# ---------------- Clustering ----------------
elif section == "Clustering":
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
    ax.scatter(X_cluster_with_labels['Avg_Jun_Sep'], X_cluster_with_labels['YoY_Change'], c=X_cluster_with_labels['Cluster'], cmap='viridis', alpha=0.7)
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
    ax.scatter(X_cluster['Avg_Jun_Sep'][y_dbscan == -1], X_cluster['YoY_Change'][y_dbscan == -1], c='red', s=100, label='Noise', marker='x')
    ax.set_title('DBSCAN Clustering')
    ax.set_xlabel('Average Rainfall (Jun-Sep)')
    ax.set_ylabel('Year-over-Year Change')
    ax.legend()
    st.pyplot(fig)

# ---------------- Trend ----------------
elif section == "Rainfall Trend":
    st.header("📉 Rainfal
