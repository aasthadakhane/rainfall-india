import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
df = pd.read_csv("../DataSets/rainfaLLIndia.csv")
df['Avg_Jun_Sep'] = df[['JUN', 'JUL', 'AUG', 'SEP']].mean(axis=1)

df.sort_values(by=['subdivision', 'YEAR'], inplace=True)
df['YoY_Change'] = df.groupby('subdivision')['Avg_Jun_Sep'].diff()

df['Lag1_Avg'] = df.groupby('subdivision')['Avg_Jun_Sep'].shift(1)
df.isnull().sum()
df_pivot = df.pivot(index='YEAR', columns='subdivision', values='Avg_Jun_Sep')
df_pivot
df.dtypes
df.drop_duplicates(inplace=True)
Phase 2: Exploratory Data Analysis (EDA)
df['subdivision_encoded'] = df['subdivision'].astype('category').cat.codes
yearly_avg = df.groupby('YEAR')['Avg_Jun_Sep'].mean()
plt.figure(figsize=(12,6))
plt.plot(yearly_avg.index, yearly_avg.values)
plt.title('Average Rainfall in India (JUN-SEP) by Year')
plt.xlabel('Year')
plt.ylabel('Avg Rainfall (mm)')
plt.grid(True)
plt.show()

print("\nAvailable Subdivisions:")
print(df['subdivision'].unique())
sub = input("\nEnter the subdivision: ").strip()
a = df[df['subdivision'] == sub]


print("\nAvailable Years:")
print(sorted(a['YEAR'].unique()))
yr = int(input("Enter the year: "))
b = a[a['YEAR'] == yr]

if b.empty:
    print("\nNo data found for that subdivision and year.")
else:
    months = ['JUN', 'JUL', 'AUG', 'SEP']
    rainfall_values = b[months].values.flatten()

    print(f"\nRainfall in {sub} during {yr}:")
    for m, val in zip(months, rainfall_values):
        print(f"{m}: {val} mm")

    sb.barplot(x=months, y=rainfall_values, palette="Blues_d")
    plt.title(f'Rainfall in {sub} - {yr} (JUN–SEP)')
    plt.ylabel('Rainfall (mm)')
    plt.xlabel('Month')
    plt.tight_layout()
    plt.show()
plt.figure(figsize=(16,8))
sb.boxplot(data=df, x='subdivision', y='Avg_Jun_Sep')
plt.xticks(rotation=90)
plt.title('Distribution of Avg Rainfall by Subdivision')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8,5))
sb.histplot(df['Avg_Jun_Sep'], bins=30, kde=True)
plt.title('Histogram of Average Rainfall (JUN-SEP)')
plt.xlabel('Avg Rainfall')
plt.show()
print("\nAvailable Subdivisions:")
print(df['subdivision'].unique())
sub = input("\nEnter the subdivision: ").strip()
a = df[df['subdivision'] == sub]

print("\nAvailable Years:")
print(sorted(a['YEAR'].unique()))
start_year = int(input("\nEnter the start year: "))
end_year = int(input("Enter the end year: "))

b = a[(a['YEAR'] >= start_year) & (a['YEAR'] <= end_year)]

if b.empty:
    print("\nNo data available for that range.")
else:
    b = b[['YEAR', 'JUN', 'JUL', 'AUG', 'SEP']].set_index('YEAR')
    plt.figure(figsize=(8, 5))
    sb.heatmap(b, annot=True, cmap='YlGnBu', fmt=".1f", cbar_kws={'label': 'Rainfall (mm)'})
    plt.title(f"Rainfall Heatmap (JUN–SEP)\n{sub} [{start_year}–{end_year}]")
    plt.ylabel("Year")
    plt.xlabel("Month")
    plt.tight_layout()
    plt.show()
corr_cols = ['JUN', 'JUL', 'AUG', 'SEP', 'JUN-SEP']
corr_matrix = df[corr_cols].corr()
corr_matrix
Phase 3: Machine Learning - Rainfall Prediction
df=df.fillna(0)
df.dtypes
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

df.subdivision= label.fit_transform(df.subdivision)

x = df.drop(['YoY_Change', 'subdivision', 'YEAR'], axis=1).values
y= df['YoY_Change'].values
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

xtrain.shape

xtest.shape

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(xtrain,ytrain)

ypred = model.predict(xtest)


from sklearn.metrics import r2_score
r2 = r2_score(ytest,ypred)
r2*100
from sklearn.model_selection import cross_val_score

modelcv = LinearRegression()

score = cross_val_score(modelcv,x,y,cv=5,scoring='r2')


score.mean()*100

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(xtrain,ytrain)

from sklearn.model_selection import GridSearchCV


param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}


model_gscv=RandomForestRegressor()

grid_search = GridSearchCV(model_gscv,param_grid, cv=5, scoring='r2')
grid_search.fit(xtrain,ytrain)

grid_search.best_params_
grid_search.best_score_

from sklearn import model_selection
from sklearn.ensemble import AdaBoostRegressor

kfold = model_selection.KFold(n_splits=20, shuffle=True, random_state=7)

seed=7
num_trees=30

model = AdaBoostRegressor(n_estimators=num_trees, random_state=seed)
results= model_selection.cross_val_score(model,x,y,cv=kfold,scoring='r2')

adb=round(results.mean(),2)*100

from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = mean_squared_error(ytest, ypred, squared=False)
mae = mean_absolute_error(ytest, ypred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
Phase 4: Insights and Advanced Analysis

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

# EDA
st.header("Exploratory Data Analysis")
df['subdivision_encoded'] = df['subdivision'].astype('category').cat.codes
yearly_avg = df.groupby('YEAR')['Avg_Jun_Sep'].mean()
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(yearly_avg.index, yearly_avg.values)
ax.set_title('Average Rainfall in India (JUN-SEP) by Year')
ax.set_xlabel('Year')
ax.set_ylabel('Avg Rainfall (mm)')
ax.grid(True)
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
    sb.barplot(x=months, y=rainfall_values, palette="Blues_d", ax=ax)
    ax.set_title(f'Rainfall in {sub} - {yr} (JUN–SEP)')
    ax.set_ylabel('Rainfall (mm)')
    ax.set_xlabel('Month')
    st.pyplot(fig)

st.subheader("Distribution of Rainfall")
fig, ax = plt.subplots(figsize=(16,8))
sb.boxplot(data=df, x='subdivision', y='Avg_Jun_Sep', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Distribution of Avg Rainfall by Subdivision')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8,5))
sb.histplot(df['Avg_Jun_Sep'], bins=30, kde=True, ax=ax)
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
    sb.heatmap(b, annot=True, cmap='YlGnBu', fmt=".1f", cbar_kws={'label': 'Rainfall (mm)'}, ax=ax)
    ax.set_title(f"Rainfall Heatmap (JUN–SEP)\n{sub} [{start_year}–{end_year}]")
    st.pyplot(fig)

# Machine Learning
st.header("Rainfall Prediction")
label = LabelEncoder()
df['subdivision'] = label.fit_transform(df['subdivision'])

x = df.drop(['YoY_Change', 'subdivision', 'YEAR'], axis=1).values
y = df['YoY_Change'].values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

model = LinearRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

r2 = r2_score(ytest, ypred)
st.write(f"Linear Regression R2 Score: {r2*100:.2f}%")

modelcv = LinearRegression()
score = cross_val_score(modelcv, x, y, cv=5, scoring='r2')
st.write(f"Cross-validated R2 Score: {score.mean()*100:.2f}%")

model_rf = RandomForestRegressor()
model_rf.fit(xtrain, ytrain)

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
st.write(f"Best RandomForest R2 Score: {grid_search.best_score_ * 100:.2f}%")

kfold = KFold(n_splits=20, shuffle=True, random_state=7)
model_adb = AdaBoostRegressor(n_estimators=30, random_state=7)
results = cross_val_score(model_adb, x, y, cv=kfold, scoring='r2')
st.write(f"AdaBoost R2 Score: {round(results.mean(),2)*100:.2f}%")

try:
    if len(ytest) != len(ypred):
        st.error("Prediction and test set length mismatch.")
    else:
        rmse = mean_squared_error(ytest, ypred, squared=False)
        mae = mean_absolute_error(ytest, ypred)
        st.subheader("📊 Model Evaluation Metrics")
        st.metric("RMSE", f"{rmse:.4f}")
        st.metric("MAE", f"{mae:.4f}")
except Exception as e:
    st.error(f"Error computing metrics: {e}")


# Clustering
st.header("Clustering Analysis")
X_cluster = df[['Avg_Jun_Sep', 'YoY_Change']]

wss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_cluster)
    wss.append(kmeans.inertia_)
fig, ax = plt.subplots()
ax.plot(range(1, 11), wss, marker='o')
ax.set_title('Elbow Method')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('WSS')
st.pyplot(fig)

# KMeans Clustering Plot (Fixed)
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X_cluster)

# Assign cluster labels for plotting
X_cluster_with_labels = X_cluster.copy()
X_cluster_with_labels['Cluster'] = y_kmeans

# Extract centroids
centers = kmeans.cluster_centers_

# Plot
fig, ax = plt.subplots()
scatter = ax.scatter(
    X_cluster_with_labels['Avg_Jun_Sep'], 
    X_cluster_with_labels['YoY_Change'], 
    c=X_cluster_with_labels['Cluster'], 
    cmap='viridis', 
    alpha=0.7
)
ax.scatter(
    centers[:, 0], centers[:, 1], 
    c='red', s=300, marker='X', label='Centroids'
)
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

# Trend Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['YEAR'], df['Avg_Jun_Sep'], label='Avg Rainfall (Jun-Sep)', color='b', marker='o')
ax.set_title('Trend of Average Rainfall (Jun-Sep) Over Years')
ax.set_xlabel('Year')
ax.set_ylabel('Avg Rainfall (mm)')
ax.grid(True)
ax.legend()
st.pyplot(fig)
