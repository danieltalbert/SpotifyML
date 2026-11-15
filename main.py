import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

def clean_values(df):
    df = df.dropna()
    df = df[df["tempo"] > 0]
    return df

