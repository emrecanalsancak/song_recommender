import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/processed/02_no_outliers.csv")

numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

# scaler = MinMaxScaler()
# df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

features = csr_matrix(df[numerical_cols])

similarity_matrix = cosine_similarity(features, features, dense_output=False)


def get_recommendations_by_name(track_name, similarity_matrix, df, top_n=5):
    track_name_lower = track_name.lower()
    df["track_name_lower"] = df["track_name"].str.lower()

    song_index = df[df["track_name_lower"] == track_name_lower].index

    if len(song_index) == 0:
        print(f"Song '{track_name}' not found in the dataset.")
        return None

    song_index = song_index[0]  # Take the first match if there are duplicates

    # Get the similarity scores for the target song
    song_similarities = similarity_matrix[song_index].toarray().flatten()

    # Exclude the song itself by setting its similarity to -1
    song_similarities[song_index] = -1

    # Get the indices of the top N most similar songs
    top_n_indices = np.argsort(song_similarities)[-top_n:][::-1]

    # Return the recommended songs and their similarity scores
    recommended_songs = df.iloc[top_n_indices]
    similarity_scores = song_similarities[top_n_indices]

    return (
        recommended_songs[["track_name", "track_artist", "playlist_subgenre"]],
        similarity_scores,
    )


# Example usage: Get recommendations for a song by its track name
recommended_songs = get_recommendations_by_name(
    track_name="godzilla",
    similarity_matrix=similarity_matrix,
    df=df,
    top_n=10,
)
print(recommended_songs)
