Customer Segmentation using K-Means Clustering

Project Overview

This project performs Customer Segmentation using the K-Means Clustering algorithm.

The project uses customer information such as Annual Income and Spending Score to group customers into different clusters based on similar characteristics.

Dataset

The project uses the Mall_Customers.csv dataset.

The main columns used in the analysis are:

Age

Annual Income (k$)

Spending Score (1-100)

Technologies and Libraries

Python

NumPy

Pandas

Matplotlib

Scikit-learn

Project Workflow

1. Import Libraries

The following libraries are imported:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

2. Load Dataset

The dataset is loaded using Pandas:

df = pd.read_csv('Mall_Customers.csv')
df.head()

3. Understand the Dataset

The shape, information and statistical summary of the dataset are checked using:

df.shape
df.info()
df.describe()

4. Check Missing Values and Duplicates

Missing values are checked using:

df.isnull().sum()

Duplicate records are checked using:

df.duplicated().sum()

Duplicate records are removed using:

df = df.drop_duplicates()

5. Exploratory Data Analysis

Histograms are created to understand the distribution of:

Age

Annual Income

Spending Score

Example:

plt.hist(df['Age'], bins=10)
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.title("Age Distribution")
plt.show()

Similar visualizations are created for Annual Income and Spending Score.

6. Feature Selection

The following two features are selected for customer segmentation:

x = df[["Annual Income (k$)", "Spending Score (1-100)"]]

7. Feature Scaling

StandardScaler is used to scale the selected features:

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

Scaling helps put the selected features on a comparable scale before applying K-Means.

8. Elbow Method

The Elbow Method is used to determine an appropriate number of clusters.

WCSS (Within-Cluster Sum of Squares) is calculated for different values of K:

WCSS = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(x_scaled)
    WCSS.append(model.inertia_)

The WCSS values are then plotted:

plt.plot(range(1, 11), WCSS, marker="o")
plt.xlabel("Number of Clusters(k)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

Based on the notebook, 5 clusters are used for the final K-Means model.

9. Train K-Means Model

The K-Means model is created with 5 clusters:

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(x_scaled)

10. Add Cluster Labels

The generated cluster labels are added to the original DataFrame:

df['Cluster'] = clusters

11. Visualize Customer Clusters

The customer clusters are visualized using a scatter plot:

plt.scatter(x_scaled[:,0], x_scaled[:,1], c=clusters)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation Using K-Means")
plt.show()

12. Analyze Clusters

The average Age, Annual Income and Spending Score are calculated for each cluster:

cluster_summary = df.groupby('Cluster')[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]].mean

cluster_summary

This helps in understanding the characteristics of each customer cluster.

13. New Customer Segmentation

A new customer's Annual Income and Spending Score are provided:

new_customer = np.array([[80, 75]])

new_customer_scaled = scaler.transform(new_customer)

cluster = kmeans.predict(new_customer_scaled)

print("Customer belongs to cluster", cluster[0])

The trained K-Means model predicts which cluster the new customer belongs to.

Project Structure

Customer-Segmentation/
│
├── Customer_segmenation.ipynb
├── Mall_Customers.csv
└── README.md

Key Concepts Used

Data Loading

Data Cleaning

Duplicate Handling

Exploratory Data Analysis (EDA)

Feature Selection

Feature Scaling

K-Means Clustering

Elbow Method

WCSS

Cluster Visualization

New Customer Prediction

Conclusion

This project demonstrates how K-Means Clustering can be used to segment customers based on their Annual Income and Spending Score. The notebook performs data cleaning, exploratory analysis, feature scaling, cluster selection using the Elbow Method, customer clustering and prediction for a new customer.

How to Run the Project

Clone the repository.

Make sure Mall_Customers.csv and Customer_segmenation.ipynb are in the required location.

Install the required libraries:

pip install numpy pandas matplotlib scikit-learn

Open Customer_segmenation.ipynb in Jupyter Notebook or VS Code.

Run the cells step by step.
