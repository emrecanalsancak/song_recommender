import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score

df = pd.read_csv("../data/processed/02_no_outliers.csv")

train_data, test_data = train_test_split(df, test_size=0.25, random_state=42)

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")


# * Matrix interaction
interaction_matrix = df.pivot_table(
    index="playlist_id", columns="track_id", values="track_popularity"
).fillna(0)

# Finding the value of K
k_values = list(range(1, 31))
mean_distances = []

for k in k_values:
    model = NearestNeighbors(n_neighbors=k, metric="cosine")
    model.fit(interaction_matrix)

    distances, indices = model.kneighbors(interaction_matrix)

    mean_distance = np.mean(distances)
    mean_distances.append(mean_distance)


plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_distances, marker="o")
plt.title("Average Distance for Different Values of k")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Average Distance")
plt.grid(False)
plt.show()

best_k = k_values[np.argmin(mean_distances)]
print(f"Best value of k: {best_k}")


# ? Collaborative Filtering-Based Recommendation Model
model = NearestNeighbors(n_neighbors=best_k, metric="cosine")
model.fit(interaction_matrix)


# Making recommendations


def recommend_song(data, model, interaction_matrix, song_name, k=10):
    song_playlist = data[
        data["track_name"].str.contains(song_name, case=False, na=False)
    ]["playlist_id"].unique()

    if len(song_playlist) == 0:
        print("Song not found")
        return

    playlist_id = song_playlist[0]

    unique_playlists = df["playlist_id"].unique()

    if playlist_id not in unique_playlists:
        print("Playlist ID not found.")
        return

    playlist_index = np.where(unique_playlists == playlist_id)[0][0]

    try:
        if isinstance(interaction_matrix, pd.DataFrame):
            distances, indices = model.kneighbors(
                interaction_matrix.iloc[playlist_index].values.reshape(1, -1),
                n_neighbors=k + 1,
            )

        else:
            distances, indices = model.kneighbors(
                interaction_matrix[playlist_index].reshape(1, -1), n_neighbors=k + 1
            )

    except IndexError:
        print("Playlist index is out of bounds")
        return

    similar_playlists = indices.flatten()[1:]

    original_playlist_tracks = set(df[df["playlist_id"] == playlist_id]["track_id"])
    recommended_tracks = set()

    for idx in similar_playlists:
        similar_playlist_id = unique_playlists[idx]
        similar_playlist_tracks = set(
            df[df["playlist_id"] == similar_playlist_id]["track_id"]
        )
        recommended_tracks.update(similar_playlist_tracks - original_playlist_tracks)

    if recommended_tracks:
        recommended_tracks_info = df[df["track_id"].isin(recommended_tracks)][
            ["track_name", "track_artist"]
        ].drop_duplicates()
        print("Recommended songs")
        print(recommended_tracks_info)
    else:
        print("No new songs to recommend.")


default_song_name = "Discombobulated"


print("Similar songs to ", default_song_name)
recommend_song(df, model, interaction_matrix, "those kinda nights")
df[df["track_artist"] == "Eminem"]
