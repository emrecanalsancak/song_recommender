import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/processed/02_no_outliers.csv")
df.columns


# * ------------ Scaling ------------ #

numerical_cols = [
    col
    for col in df.columns
    if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 2
]


X = df[numerical_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ^ ------------ Clustering ------------ #
sse = []
k_range = range(1, 31)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)


# Plot the elbow plot
plt.figure(figsize=(8, 6))
plt.plot(k_range, sse, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Distances (Inertia)")
plt.title("Elbow Method for Determining the Number of Clusters")
plt.grid(False)
plt.savefig("../reports/clusters/elbow_k.png")
plt.show()


kmeans = KMeans(n_clusters=4, random_state=42)
df["cluster_kmeans"] = kmeans.fit_predict(X_scaled)
df.to_csv("../data/processed/03_data_clustered.csv", index=False)

df["cluster_kmeans"].value_counts()

# * Counting songs in each cluster
cluster_counts = df["cluster_kmeans"].value_counts().sort_values(ascending=False)

# percentage of each cluster
total_songs = cluster_counts.sum()
cluster_percentages = (cluster_counts / total_songs) * 100


# Plot the distribution of clusters with percentages
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="deep")
plt.xlabel("Cluster")
plt.ylabel("Number of Songs")
plt.title("Distribution of Songs by Cluster")

for index, value in enumerate(cluster_counts):
    plt.text(
        index,
        value + 100,
        f"{value} ({cluster_percentages[index]:.1f}%)",
        ha="center",
        fontsize=12,
    )

plt.grid(False)
plt.savefig("../reports/clusters/song_dist_by_cluster.png")
plt.show()


# * ------------ PCA ------------ #
pca = PCA(n_components=2)
x_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(x_pca, columns=["PC1", "PC2"])
df_pca["cluster_kmeans"] = df["cluster_kmeans"]
df_pca.to_csv("../data/processed/04_data_pca_clustered.csv", index=False)

# Cluster visualization

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=x_pca[:, 0],
    y=x_pca[:, 1],
    hue=df["cluster_kmeans"],
    palette="viridis",
    alpha=0.6,
    edgecolor="k",
)

centers = kmeans.cluster_centers_
centers_pca = pca.transform(centers)
plt.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    c="red",
    s=200,
    marker="X",
    label="Cluster Centers",
)


plt.title("Clusters after Applying K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(False)
plt.savefig("../reports/clusters/clusters.png")
plt.show()


# Clusters after pca
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=x_pca[:, 0],
    y=x_pca[:, 1],
    hue=df["cluster_kmeans"],
    palette="viridis",
    s=70,
    alpha=0.7,
)

# Plot the cluster centers
plt.scatter(
    centers[:, 0], centers[:, 1], c="red", s=200, marker="X", label="Centroides"
)

plt.title("Visualization of Clusters after PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Clusters")
plt.grid(False)
plt.savefig("../reports/clusters/clusters_after_pca.png")
plt.show()


# * ---------- Mean of features for clusters ---------- #
cluster_1_features = df[df["cluster_kmeans"] == 0][numerical_cols].mean()
print(f"Mean of Features for Cluster 0: {cluster_1_features}", end="\n-----------")

cluster_2_features = df[df["cluster_kmeans"] == 1][numerical_cols].mean()
print(f"Mean of Features for Cluster 1: {cluster_2_features}", end="\n-----------")

cluster_3_features = df[df["cluster_kmeans"] == 2][numerical_cols].mean()
print(f"Mean of Features for Cluster 2: {cluster_3_features}", end="\n-----------")

cluster_4_features = df[df["cluster_kmeans"] == 3][numerical_cols].mean()
print(f"Mean of Features for Cluster 3: {cluster_4_features}", end="\n-----------")

cluster_5_features = df[df["cluster_kmeans"] == 4][numerical_cols].mean()
print(f"Mean of Features for Cluster 4: {cluster_5_features}", end="\n-----------")
