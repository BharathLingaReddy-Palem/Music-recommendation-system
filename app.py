import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="wide")


@st.cache_data
def load_data(path: str = "dataset_cleaned.csv") -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def build_recommender(df: pd.DataFrame):
    feature_cols = [
        "danceability",
        "energy",
        "tempo",
        "valence",
        "acousticness",
        "instrumentalness",
        "liveness",
        "speechiness",
        "loudness",
        "duration_ms",
        "popularity",
        "explicit",
        "key",
        "mode",
        "time_signature",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(x_scaled)

    search_df = df.copy()
    search_df["track_name_lower"] = search_df["track_name"].astype(str).str.lower().str.strip()
    search_df["artists_lower"] = search_df["artists"].astype(str).str.lower().str.strip()

    return model, x_scaled, search_df


def find_candidates(search_df: pd.DataFrame, song_name: str, max_candidates: int = 10) -> pd.DataFrame:
    query = str(song_name).lower().strip()

    exact = search_df[search_df["track_name_lower"] == query]
    if not exact.empty:
        return exact[["track_name", "artists", "track_genre"]].head(max_candidates)

    partial = search_df[search_df["track_name_lower"].str.contains(query, na=False)]
    return partial[["track_name", "artists", "track_genre"]].drop_duplicates().head(max_candidates)


def recommend_songs(
    df: pd.DataFrame,
    search_df: pd.DataFrame,
    model: NearestNeighbors,
    x_scaled,
    song_name: str,
    top_n: int = 10,
    artist_name: str | None = None,
):
    query = str(song_name).lower().strip()
    candidates = search_df[search_df["track_name_lower"] == query]

    if candidates.empty:
        return {
            "status": "not_found",
            "message": f"Song not found: {song_name}",
            "suggestions": find_candidates(search_df, song_name),
        }

    if artist_name:
        artist_query = str(artist_name).lower().strip()
        filtered = candidates[candidates["artists_lower"].str.contains(artist_query, na=False)]
        if not filtered.empty:
            candidates = filtered

    query_index = candidates.index[0]

    distances, indices = model.kneighbors(x_scaled[query_index].reshape(1, -1), n_neighbors=top_n + 1)

    rec_indices = []
    rec_distances = []
    for idx, dist in zip(indices.flatten(), distances.flatten()):
        if idx == query_index:
            continue
        rec_indices.append(idx)
        rec_distances.append(dist)
        if len(rec_indices) == top_n:
            break

    rec_df = df.iloc[rec_indices][["track_name", "artists", "track_genre", "popularity"]].copy()
    rec_df["cosine_distance"] = rec_distances
    rec_df["similarity_score"] = 1 - rec_df["cosine_distance"]
    rec_df = rec_df.sort_values("similarity_score", ascending=False).reset_index(drop=True)

    return {
        "status": "ok",
        "query_song": df.loc[query_index, ["track_name", "artists", "track_genre"]].to_dict(),
        "recommendations": rec_df,
    }


st.title("Music Recommendation System")
st.caption("Content-based filtering with cosine similarity on audio features")

try:
    df_clean = load_data("dataset_cleaned.csv")
except FileNotFoundError:
    st.error("dataset_cleaned.csv not found. Place it in the same folder as app.py.")
    st.stop()

model, x_scaled, search_df = build_recommender(df_clean)

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    song_name = st.text_input("Song name", value="I'm Yours")
with col2:
    artist_name = st.text_input("Artist name (optional)", value="Jason Mraz")
with col3:
    top_n = st.slider("Top N", min_value=5, max_value=20, value=10, step=1)

if st.button("Get Recommendations", type="primary"):
    result = recommend_songs(
        df=df_clean,
        search_df=search_df,
        model=model,
        x_scaled=x_scaled,
        song_name=song_name,
        top_n=top_n,
        artist_name=artist_name if artist_name.strip() else None,
    )

    if result["status"] == "ok":
        st.success("Recommendations generated")
        st.write("Query song:")
        st.json(result["query_song"])
        st.dataframe(result["recommendations"], use_container_width=True)
    else:
        st.warning(result["message"])
        st.write("Suggestions:")
        st.dataframe(result["suggestions"], use_container_width=True)

with st.expander("How to Search (Examples)"):
    st.markdown("""
    ### Try these queries
    1. Song: **I'm Yours** | Artist: **Jason Mraz**
    2. Song: **Say Something** | Artist: *(leave blank)*
    3. Song: **Brave** | Artist: **Sara Bareilles**
    4. Song: **Boston** | Artist: **Augustana**
    5. Song: **Hold On** | Artist: **Chord Overstreet**
    6. Song: **93 Million Miles** | Artist: **Jason Mraz**
    7. Song: **Kaleidoscope** | Artist: **A Great Big World**

    ### Tips
    - Start with the exact song title.
    - If the title is common, add the artist name.
    - If not found, try a shorter phrase.
    - Increase **Top N** to 15 or 20 for broader results.
    """)
