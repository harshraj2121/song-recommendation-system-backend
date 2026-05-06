from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
CORS(app)

# Load model
df = pickle.load(open('songs.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
matrix = pickle.load(open('matrix.pkl', 'rb'))

# ---------------- SEARCH (iTunes API) ----------------
@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data.get("search", "").strip().lower()

        if not query:
            return jsonify({"searchedOutput": []})

        # ----------- ONLY DATASET SEARCH -----------
        matches = df[df['track_name'].str.lower().str.contains(query, regex=False, na=False)]

        songs = []

        for _, row in matches.iterrows():
            songs.append({
                "name": row['track_name'],
                "artist": row['artists'],
                "album": row.get('album_name', ''),  # agar column hai to use hoga
                "image": "",
                "song_link": ""
            })

        return jsonify({"searchedOutput": songs})

    except Exception as e:
        return jsonify({"error": str(e)})


# ---------------- RECOMMEND (DATASET BASED) ----------------
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        song_name = data.get("songName", "").lower()

        # find index
        matches = df[df['track_name'].str.lower().str.contains(song_name)]

        if matches.empty:
            return jsonify({"recommendation": []})

        idx = matches.index[0]

        # 🔥 compute similarity ONLY for 1 song

        sim_scores = cosine_similarity(matrix[idx], matrix).flatten()
        top_indices = sim_scores.argsort()[-6:-1][::-1]

        top_indices = sim_scores.argsort()[-6:-1][::-1]

        rec_songs = df.iloc[top_indices]

        print(rec_songs.columns.tolist())

        results = []
        for _, row in rec_songs.iterrows():
            results.append({
                "name": row['track_name'],
                "artist": row['artists'],
                "duration_ms": row['duration_ms'],
                "album": row['album_name'],
            })
        
        print(results)

        return jsonify({"recommendation": results})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)