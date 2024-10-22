import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class SongDataset(Dataset):
    def __init__(self, features, similarity_labels=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.similarity_labels = similarity_labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if self.similarity_labels is not None:
            return self.features[idx], self.similarity_labels[idx]

        else:
            return self.features[idx]


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

data = SongDataset(numerical_data)
dataloader = DataLoader(data, batch_size=32, shuffle=True)

input_size = numerical_data.shape[1]
embedding_size = 32
model = SongRecommender(input_size, embedding_size)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 100
for epoch in range(num_epochs):
    for batch_features in dataloader:
        optimizer.zero_grad()
        embeddings = model(batch_features)

        loss = criterion(embeddings, embeddings)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


torch.save(model.state_dict(), "../torch_model/song_recommender_model.pth")


with torch.no_grad():
    song_embeddings = model(torch.tensor(numerical_data, dtype=torch.float32))


# Create a mapping of lowercase song names to indices
song_name_to_index = {track.lower(): idx for idx, track in enumerate(df["track_name"])}


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


recommended = recommend_songs_by_name("godzilla", song_embeddings, df)
recommended_songs = df.iloc[recommended]["track_name"].tolist()
print(f"Recommended songs: {recommended_songs}")

df[df["track_name"].str.lower() == "middle child"]
