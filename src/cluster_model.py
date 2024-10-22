import pandas as pd
import numpy as np

df = pd.read_csv("../data/processed/03_data_clustered.csv")


def recommend_songs_with_variety(song_name, data, num_clusters=4, sample_size=30):
    selected_song = data[
        data["track_name"].str.contains(song_name, case=False, na=False)
    ]

    if selected_song.empty:
        print("Song not found.")
        return None

    cluster = selected_song["cluster_kmeans"].values[0]

    # Get clusters that are close to the selected song's cluster
    clusters_to_consider = data["cluster_kmeans"].unique()
    clusters_to_consider = np.random.choice(
        clusters_to_consider,
        size=min(num_clusters, len(clusters_to_consider)),
        replace=False,
    )

    recommended_songs = data[data["cluster_kmeans"].isin(clusters_to_consider)]

    # Exclude the selected song
    recommended_songs = recommended_songs[
        recommended_songs["track_name"] != selected_song["track_name"].values[0]
    ]

    # Drop duplicates based on track_name and shuffle
    recommended_songs = recommended_songs.drop_duplicates(subset="track_name")
    recommended_songs = recommended_songs.sample(frac=1).reset_index(
        drop=True
    )  # Shuffle the songs

    return recommended_songs[["track_name", "track_artist"]].head(sample_size)


song_name = "Monster"
top_recommendations_kmeans = recommend_songs_with_variety(song_name, df, 15, 10)

top_recommendations_kmeans
df[df["track_name"] == "Blinding Lights"]
