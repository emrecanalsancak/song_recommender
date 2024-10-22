import torch
import torch.nn as nn
import pandas as pd
import streamlit as st


class SongRecommender(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(SongRecommender, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.embedding = nn.Linear(64, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        embedding = self.embedding(x)
        return embedding


df = pd.read_csv("../data/processed/03_data_clustered.csv")
numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

numerical_data = df[numerical_cols].values
input_size = numerical_data.shape[1]
embedding_size = 32
model = SongRecommender(input_size, embedding_size)
model.load_state_dict(torch.load("../torch_model/song_recommender_model.pth"))

with torch.no_grad():
    song_embeddings = model(torch.tensor(numerical_data, dtype=torch.float32))


song_name_to_index = {track.lower(): idx for idx, track in enumerate(df["track_name"])}


st.title("Song Recommender System")

# Input field for song name
song_name = st.text_input("Enter a song name:", "Shape of You")


def recommend_songs_by_name(song_name, song_embeddings, song_data, top_k=5):
    # Convert the input song name to lowercase
    song_name_lower = song_name.lower()

    # Get the index of the song by its (lowercased) name
    song_idx = song_name_to_index.get(song_name_lower, None)

    if song_idx is None:
        print(f"Song '{song_name}' not found in the dataset.")
        return pd.DataFrame(
            columns=["artist_name", "track_name", "genre"]
        )  # Return an empty DataFrame

    # Get the embedding of the selected song
    song_embedding = song_embeddings[song_idx]

    # Compute cosine similarity between the selected song and all others
    similarity_scores = torch.nn.functional.cosine_similarity(
        song_embedding.unsqueeze(0), song_embeddings
    )

    # Get top-k most similar songs (excluding the song itself)
    top_songs = torch.topk(similarity_scores, top_k + 1).indices.tolist()

    # Remove the song itself from the recommendations
    if song_idx in top_songs:
        top_songs.remove(song_idx)

    # Retrieve details for the top_k recommended songs
    recommended_songs = song_data.iloc[top_songs[:top_k]]

    # Return only artist_name, track_name, and genre columns as a DataFrame
    return recommended_songs[["track_artist", "track_name", "playlist_subgenre"]]


def main():
    if st.button("Recommend"):
        if song_name:
            recommendations = recommend_songs_by_name(song_name, song_embeddings, df)

            st.write("Top reccommended")
            st.dataframe(recommendations)


if __name__ == "__main__":
    main()
