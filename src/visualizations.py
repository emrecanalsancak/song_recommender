import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/processed/01_songs_no_missing.csv")
df.head()


def avg_popularity_by_genre(genre="playlist_genre"):
    name = "Genre" if genre == "playlist_genre" else "Subgenre"
    popularity = (
        df.groupby(genre)["track_popularity"].mean().sort_values(ascending=False)
    )

    # top 30 most popular genres
    top_n = 30
    top_popularity = popularity.nlargest(top_n)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_popularity.values, y=top_popularity.index, palette="autumn")

    # Adding data labels
    for index, value in enumerate(top_popularity.values):
        plt.text(value, index, f"{value:.1f}", va="center", ha="left", color="black")

    # Making it more readable
    plt.title(f"Average popularity by {name.lower()}", fontsize=16)
    plt.xlabel("Average popularity", fontsize=12)
    plt.ylabel(name, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Gridlines for comparision
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.savefig(f"../reports/EDA_plots/avg_popularity_by_{name.lower()}.png")


def genre_key_counts(genre="playlist_genre"):
    name = "Genre" if genre == "playlist_genre" else "Subgenre"
    genre_counts = df[genre].value_counts()
    plt.figure(figsize=(15, 6))
    sns.barplot(x=genre_counts.index, y=genre_counts.values, palette="autumn")
    plt.xlabel(name)
    plt.ylabel("Count")
    plt.title(f"{name} Counts")
    plt.xticks(rotation=90)
    plt.savefig(f"../reports/EDA_plots/{name.lower()}_counts.png")

    key_counts = df["key"].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=key_counts.index, y=key_counts.values, palette="autumn")
    plt.xlabel("Key")
    plt.ylabel("Count")
    plt.title("Key Counts")
    plt.savefig("../reports/EDA_plots/key_counts.png")


def danceability_plots(genre="playlist_genre"):
    name = "Genre" if genre == "playlist_genre" else "Subgenre"
    # Danceability by genre
    genre_danceability = (
        df.groupby(genre)["danceability"].mean().sort_values(ascending=False)
    )

    plt.figure(figsize=(15, 6))
    sns.barplot(
        x=genre_danceability.index, y=genre_danceability.values, palette="autumn"
    )
    plt.xlabel(name)
    plt.ylabel("Average Danceability")
    plt.title(f"Top Danceability by {name}")
    plt.xticks(rotation=90)
    plt.savefig(f"../reports/EDA_plots/top_danceability_by_{name}.png")

    # Danceability distribution
    plt.figure(figsize=(20.5, 10))
    sns.kdeplot(
        df["danceability"],
        fill=True,
        label="danceability",
        color="blue",
        alpha=0.5,
        bw_adjust=0.7,
    )
    sns.kdeplot(
        df["energy"], fill=True, label="energy", color="red", alpha=0.5, bw_adjust=0.7
    )
    plt.axvline(
        df["danceability"].mean(),
        color="blue",
        linestyle="--",
        label="Danceability Mean",
    )
    plt.axvline(df["energy"].mean(), color="red", linestyle="--", label="Energy Mean")
    plt.title("Danceability Distribution", fontsize=16)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(False)
    plt.savefig("../reports/EDA_plots/danceability_distribution.png")


def numerical_dist():
    numerical_cols = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
    ]
    df[numerical_cols].hist(bins=15, figsize=(15, 10), color="orangered")
    plt.suptitle("Distribution of Numerical Variables")
    plt.savefig("../reports/EDA_plots/numerical_distribution.png")


if __name__ == "__main__":
    for genre in ["playlist_genre", "playlist_subgenre"]:
        avg_popularity_by_genre(genre)
        genre_key_counts(genre)
        danceability_plots(genre)
    numerical_dist()
