import install_fix
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('netflix_titles.csv')
df = df.dropna(subset=['title', 'listed_in', 'description'])

# Combine text features
df['combined'] = df['listed_in'] + " " + df['description']

# TF-IDF Matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation Function
def recommend(title):
    if title not in df['title'].values:
        return []
    idx = df[df['title'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    recs = [df.iloc[i[0]]['title'] for i in scores]
    return recs

# --- Streamlit UI ---
st.title("🎬 Netflix Movie Recommender")
st.write("Find similar movies instantly!")

movie = st.selectbox("Select a movie:", sorted(df['title'].unique()))
if st.button("Recommend"):
    recs = recommend(movie)
    if recs:
        st.subheader("You might also like:")
        for r in recs:
            st.write(f"✅ {r}")
    else:
        st.write("Sorry, no recommendations found.")
