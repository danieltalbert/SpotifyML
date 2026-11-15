import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# The nine audio features that describe a song's sound.
# Defined once at the top of the file so every function can use the same list.
feature_col = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

#Just load in the data and have it usable upon request
def load_data(path):
    df = pd.read_csv(path)
    return df

#Get a sneak peak at the data, not really needed but it's good practive for a Junior Dev like me
def inspect_data(df):
    print(df.shape, df.columns, df.head(10))

#Now to actually manipulate the data. We will 'collapse' it into one row per track_id
def dedupe_tracks(df):
    #The parameters like subset and keep are just two things I put in there, but you could put any other parameters that exist in there depending on your use case
    df = df.drop_duplicates(subset="track_id", keep="first")
    return df

#name is self explanatory
def clean_values(df):
    df = df.dropna()
    df = df[df["tempo"] > 0]
    df = df[df["time_signature"] > 0]
    return df

# Keep only the nine feature columns and scale them, so no single feature
# dominates the distance math later just because its raw numbers are bigger.
def build_features(df):
    features = df[feature_col]
    scaler = StandardScaler()
    # fit_transform learns each column's average and spread, then returns the rescaled numbers.
    scaled = scaler.fit_transform(features)
    # The scaler hands back a bare grid with no labels, so rebuild a DataFrame
    # with the same column names and the same row index as the original table.
    scaled_df = pd.DataFrame(scaled, columns=feature_col, index=df.index)
    return scaled_df

# Print the shape of the popularity column, including how much of it is zeros.
def popularity_summary(df):
    print(df["popularity"].describe())
    zero_share = (df["popularity"] == 0).mean() * 100
    print("Share of tracks with popularity 0:", zero_share, "%")

# Score how each feature moves together with popularity, strongest first.
def feature_correlations(df):
    corr = df[feature_col + ["popularity"]].corr()["popularity"].drop("popularity")
    print(corr.sort_values(ascending=False))
    return corr

# Average popularity and track count for every genre, highest average first.
def genre_summary(df):
    summary = df.groupby("track_genre")["popularity"].agg(["mean", "count"])
    summary = summary.sort_values("mean", ascending=False)
    print(summary)
    return summary

# Bar charts of the ten highest and ten lowest genres by average popularity.
# The PNGs are saved so they can go into the report.
def plot_genre_popularity(summary):
    top10 = summary.head(10)
    plt.figure(figsize=(10, 5))
    plt.bar(top10.index, top10["mean"])
    plt.title("Top 10 genres by average popularity")
    plt.ylabel("Average popularity")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("top_genres.png")
    plt.show()

    bottom10 = summary.tail(10)
    plt.figure(figsize=(10, 5))
    plt.bar(bottom10.index, bottom10["mean"])
    plt.title("Bottom 10 genres by average popularity")
    plt.ylabel("Average popularity")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("bottom_genres.png")
    plt.show()

# Scatter of one feature against popularity, one dot per song.
# Tiny dots with low alpha, or 90,000 songs would just be a solid blob.
def plot_feature_scatter(df, feature="instrumentalness"):
    plt.figure(figsize=(8, 5))
    plt.scatter(df[feature], df["popularity"], s=1, alpha=0.1)
    plt.title(feature + " vs popularity")
    plt.xlabel(feature)
    plt.ylabel("popularity")
    plt.tight_layout()
    plt.savefig("scatter_" + feature + ".png")
    plt.show()

# Elbow method: fit K-Means once for each k from 2 to 10 and record the inertia.
# Inertia always drops as k grows, so the k to use is the bend in the curve.
def choose_k(scaled_df):
    inertias = []
    for k in range(2, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(scaled_df)
        inertias.append(model.inertia_)
        print("k =", k, "inertia =", model.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 11), inertias, marker="o")
    plt.title("Elbow method: inertia by number of clusters")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig("elbow.png")
    plt.show()
    return inertias

# Fit K-Means once at the chosen k and store each song's cluster number
# as a new column on the original table.
def fit_clusters(df, scaled_df, k=5):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(scaled_df)
    print(df["cluster"].value_counts().sort_index())
    return df, model

# Average raw feature values inside each cluster.
# This is the table the vibe names come from, so it uses raw values a human can read.
def describe_clusters(df):
    profile = df.groupby("cluster")[feature_col].mean()
    print(profile)
    return profile

# Plot every song on two scaled features, colored by which cluster it landed in.
def plot_clusters(df, scaled_df):
    plt.figure(figsize=(8, 5))
    plt.scatter(scaled_df["energy"], scaled_df["danceability"], c=df["cluster"], s=1, alpha=0.3, cmap="tab10")
    plt.title("Clusters: energy vs danceability (scaled)")
    plt.xlabel("energy (scaled)")
    plt.ylabel("danceability (scaled)")
    plt.colorbar(label="cluster")
    plt.tight_layout()
    plt.savefig("clusters.png")
    plt.show()

# Content-based recommender: find the n songs closest to the seed song
# in scaled feature space and return them with their real names and genres.
def recommend(df, scaled_df, track_name, n=5):
    matches = df[df["track_name"] == track_name]
    if matches.empty:
        print("Track not found:", track_name)
        return None
    seed_index = matches.index[0]
    seed = scaled_df.loc[seed_index]
    # Straight-line distance from the seed to every song across the nine features.
    distances = ((scaled_df - seed) ** 2).sum(axis=1) ** 0.5
    # Sorted smallest first; position 0 is the seed itself, so skip it.
    closest = distances.sort_values().iloc[1:n + 1]
    result = df.loc[closest.index, ["track_name", "artists", "track_genre", "popularity"]].copy()
    result["distance"] = closest.values
    print(result)
    return result

# Distance test: the five recommendations should sit much closer to the seed
# than five random songs do. If the averages came out similar, the recommender
# would be no better than chance.
def test_recommender(df, scaled_df, track_name):
    recs = recommend(df, scaled_df, track_name)
    seed_index = df[df["track_name"] == track_name].index[0]
    seed = scaled_df.loc[seed_index]
    distances = ((scaled_df - seed) ** 2).sum(axis=1) ** 0.5
    random_rows = df.sample(5, random_state=42)
    random_distances = distances.loc[random_rows.index]
    print("Average distance, 5 recommendations:", recs["distance"].mean())
    print("Average distance, 5 random tracks:", random_distances.mean())

# Genre test, two halves: how many of the recommendations share the seed's
# genre, and how pure each cluster is (its biggest genre's share of the cluster).
def test_cluster_genres(df, scaled_df, track_name):
    seed_genre = df[df["track_name"] == track_name]["track_genre"].iloc[0]
    recs = recommend(df, scaled_df, track_name)
    matches = (recs["track_genre"] == seed_genre).sum()
    print("Seed genre:", seed_genre)
    print("Recommendations sharing the seed genre:", matches, "out of 5")

    table = pd.crosstab(df["cluster"], df["track_genre"], normalize="index")
    purity = table.max(axis=1)
    print("Biggest genre share inside each cluster:")
    print(purity)
    print("Average biggest-genre share:", purity.mean())

# Run the whole pipeline in order: load, clean, analyze, cluster, recommend, test.
def main():
    df = load_data("dataset.csv")
    inspect_data(df)
    df = dedupe_tracks(df)
    df = clean_values(df)
    scaled_df = build_features(df)

    popularity_summary(df)
    feature_correlations(df)
    summary = genre_summary(df)
    plot_genre_popularity(summary)
    plot_feature_scatter(df)

    choose_k(scaled_df)
    df, model = fit_clusters(df, scaled_df, k=5)
    describe_clusters(df)
    plot_clusters(df, scaled_df)

    # The seed song picks itself: the most popular track in the cleaned data,
    # so the name is guaranteed to exist.
    seed = df.sort_values("popularity", ascending=False)["track_name"].iloc[0]
    print("Seed track:", seed)
    recommend(df, scaled_df, seed)
    test_recommender(df, scaled_df, seed)
    test_cluster_genres(df, scaled_df, seed)


if __name__ == "__main__":
    main()
