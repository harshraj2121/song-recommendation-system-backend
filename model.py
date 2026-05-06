import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

df = pd.read_csv("songs.csv")

df = df[['track_name', 'artists', 'track_genre', 'album_name', 'duration_ms']].dropna()

df['combined'] = df['track_name'] + " " + df['artists'] + " " + df['track_genre'] + " " + df['album_name'] + " " + df['duration_ms'].astype(str)

tfidf = TfidfVectorizer(stop_words='english')
matrix = tfidf.fit_transform(df['combined'])

# Save only needed things
pickle.dump(df, open('songs.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(matrix, open('matrix.pkl', 'wb'))

print("Lightweight model ready 🚀")