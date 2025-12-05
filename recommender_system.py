import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from scipy.sparse.linalg import svds
import random
import urllib.parse # Required for generating links

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================
st.set_page_config(page_title="Streamline - AI Recommender", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    h1 { color: #1e3799; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #4a69bd; color: white; }
    .stButton>button:hover { background-color: #1e3799; color: white; }
    .metric-card { background-color: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. DATA GENERATION (MASSIVE DATASET)
# ==========================================

def generate_ratings(items_df, num_users=100, density=0.2):
    """
    Generates synthetic realistic ratings to make the system feel 'alive'.
    """
    data = []
    item_ids = items_df['item_id'].tolist()
    
    # Create specific personas for demo purposes (Users 1-5)
    for i in range(1, 6):
        num_ratings = random.randint(10, 20)
        chosen_items = random.sample(item_ids, num_ratings)
        for item in chosen_items:
            data.append({'user_id': i, 'item_id': item, 'rating': random.randint(3, 5)})
            
    # Generate background noise users (Users 6-100)
    for u in range(6, num_users + 1):
        num_ratings = random.randint(5, 25) 
        chosen_items = random.sample(item_ids, num_ratings)
        for item in chosen_items:
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.35, 0.3])
            data.append({'user_id': u, 'item_id': item, 'rating': rating})
            
    return pd.DataFrame(data)

def get_smart_link(title, type='movie'):
    """Generates a search link for Spotify (Music) or YouTube (Movies)"""
    safe_title = urllib.parse.quote(title)
    if type == 'music':
        return f"https://open.spotify.com/search/{safe_title}"
    else:
        # For movies, we link to a YouTube trailer search
        return f"https://www.youtube.com/results?search_query={safe_title}+trailer"

@st.cache_data
def load_movie_data():
    """Expanded Database for Movies"""
    titles = [
        # Western (1-20)
        'The Matrix', 'Inception', 'Toy Story', 'Finding Nemo', 'The Godfather', 
        'Pulp Fiction', 'Interstellar', 'John Wick', 'Up', 'Fight Club',
        'The Dark Knight', 'Spirited Away', 'Parasite', 'Avengers: Endgame', 'Joker',
        'Titanic', 'Avatar', 'The Lion King', 'Gladiator', 'Forrest Gump',
        # Indian (21-50)
        '3 Idiots', 'RRR', 'Dangal', 'Sholay', 'Lagaan',
        'Baahubali', 'Kantara', 'Zindagi Na Milegi Dobara', 'Gully Boy', 'Drishyam',
        'Dilwale Dulhania Le Jayenge', 'Swades', 'Chak De India', 'Queen', 'Andhadhun',
        'Gangs of Wasseypur', 'PK', 'Bajrangi Bhaijaan', 'K.G.F', 'Pushpa',
        'Taare Zameen Par', 'Munna Bhai MBBS', 'Hera Pheri', 'Drishyam 2', 'Animal',
        'Jawan', 'Pathaan', 'Stree', 'Tumbbad', '12th Fail'
    ]
    
    items = pd.DataFrame({
        'item_id': range(1, 51),
        'title': titles,
        'features': [
            # Western
            'Sci-Fi Action', 'Sci-Fi Thriller', 'Animation Children', 'Animation Children', 'Crime Drama', 
            'Crime Drama', 'Sci-Fi Drama', 'Action Thriller', 'Animation Drama', 'Drama Thriller',
            'Action Crime', 'Animation Fantasy', 'Thriller Drama', 'Action Sci-Fi', 'Crime Drama',
            'Romance Drama Epic', 'Sci-Fi Adventure', 'Animation Musical', 'Action Historical', 'Drama Romance',
            # Indian
            'Comedy Drama Education', 'Action Historical Epic', 'Sports Drama Biopic', 'Action Adventure Classic', 'Sports Drama Historical',
            'Action Epic Fantasy', 'Thriller Folklore Mystical', 'Adventure Comedy Road', 'Music Drama Rap', 'Thriller Mystery Crime',
            'Romance Drama Classic', 'Drama Social Patriotic', 'Sports Drama Patriotic', 'Comedy Drama Travel', 'Thriller Crime Dark',
            'Crime Action Grit', 'Comedy Sci-Fi Satire', 'Drama Comedy Heartfelt', 'Action Crime Gold', 'Action Crime Smuggling',
            'Drama Education Emotional', 'Comedy Drama Medical', 'Comedy Classic Cult', 'Thriller Mystery Sequel', 'Action Drama Violent',
            'Action Thriller Mass', 'Action Spy Blockbuster', 'Horror Comedy', 'Horror Fantasy Atmospheric', 'Drama Inspiration Biopic'
        ],
        'region': ['Western']*20 + ['Indian']*30,
        'link': [get_smart_link(t, 'movie') for t in titles] # Generate YouTube Links
    })
    
    ratings = generate_ratings(items)
    return items, ratings

@st.cache_data
def load_music_data():
    """Expanded Database for Music"""
    titles = [
        # Western (101-120)
        'Bohemian Rhapsody', 'Shape of You', 'Smells Like Teen Spirit', 'Hotel California', 'Blinding Lights',
        'Rolling in the Deep', 'Billie Jean', 'Sicko Mode', 'Imagine', 'Lose Yourself',
        'Uptown Funk', 'Someone Like You', 'Despacito', 'Believer', 'Starboy',
        'Hips Don\'t Lie', 'Viva La Vida', 'Thinking Out Loud', 'Bad Guy', 'Levitating',
        # Indian (121-150)
        'Kesariya', 'Jai Ho', 'Naatu Naatu', 'Tum Hi Ho', 'Kun Faya Kun',
        'Apna Time Aayega', 'Chaiyya Chaiyya', 'Kabira', 'Kal Ho Naa Ho', 'Pasoori',
        'Dil Diyan Gallan', 'Raabta', 'Ilahi', 'Agar Tum Saath Ho', 'Channa Mereya',
        'Khalibali', 'Brown Munde', 'Lover', 'Excuses', 'Tera Ghata',
        'Maan Meri Jaan', 'King of Kotha', 'Hukum', 'Chaleya', 'Jhoome Jo Pathaan',
        'Saami Saami', 'Srivalli', 'Oo Antava', 'Apna Bana Le', 'Pehle Bhi Main'
    ]

    items = pd.DataFrame({
        'item_id': range(101, 151),
        'title': titles,
        'features': [
            # Western
            'Classic Rock Opera', 'Pop Dance Happy', 'Grunge Rock Energetic', 'Classic Rock Chill', 'Synthwave Pop Night',
            'Soul Pop Sad', 'Pop Funk Dance', 'Hip-Hop Trap Energetic', 'Piano Pop Calm', 'Hip-Hop Rap Motivational',
            'Funk Pop Party', 'Soul Ballad Sad', 'Reggaeton Latin Dance', 'Rock Pop Power', 'R&B Pop Dark',
            'Latin Pop Dance', 'Alt Rock Anthem', 'Acoustic Pop Romance', 'Pop Trap Dark', 'Disco Pop Dance',
            # Indian
            'Romantic Bollywood Acoustic', 'Anthem Bollywood Energetic', 'Tollywood Dance High-Energy', 'Romantic Bollywood Sad', 'Sufi Spiritual Calm',
            'Hip-Hop Rap Gully', 'Folk Bollywood Dance', 'Sufi Folk Soulful', 'Ballad Bollywood Sad', 'Coke Studio Pop Fusion',
            'Romantic Bollywood Soft', 'Pop Romance Happy', 'Travel Folk Indie', 'Drama Soulful Sad', 'Rock Sufi Heartbreak',
            'Energy Historic Intense', 'Punjabi Hip-Hop Chill', 'Punjabi Pop Romance', 'Punjabi Pop Party', 'Indie Pop Sad',
            'Pop Rap Romantic', 'Hip-Hop South Mass', 'Rock Anthem Mass', 'Romantic Bollywood Breeze', 'Dance Bollywood Party',
            'Folk Dance Mass', 'Melody South Love', 'Item Dance Beats', 'Romantic Arijit Soul', 'Rock Animal Emotional'
        ],
        'region': ['Western']*20 + ['Indian']*30,
        'link': [get_smart_link(t, 'music') for t in titles] # Generate Spotify Links
    })
    
    ratings = generate_ratings(items)
    return items, ratings

# ==========================================
# 2. THE RECOMMENDATION ENGINE (LOGIC)
# ==========================================
class RecommenderEngine:
    def __init__(self, items_df, ratings_df):
        self.items_df = items_df
        self.ratings_df = ratings_df
        self.item_map = dict(zip(items_df['item_id'], items_df['title']))
        self.link_map = dict(zip(items_df['item_id'], items_df['link'])) # NEW: Map IDs to Links
        self.id_map = {mid: i for i, mid in enumerate(items_df['item_id'])}
        
        # Calculate Average Rating for every item (for filtering)
        avg_ratings = ratings_df.groupby('item_id')['rating'].mean()
        self.items_df['avg_rating'] = self.items_df['item_id'].map(avg_ratings).fillna(0)
        
        # Train models
        self.content_sim = self._calculate_content_similarity()
        self.preds_df, self.sigma = self._calculate_collaborative_filtering()

    def _calculate_content_similarity(self):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.items_df['features'])
        return cosine_similarity(tfidf_matrix, tfidf_matrix)

    def _calculate_collaborative_filtering(self):
        if self.ratings_df.empty:
            return pd.DataFrame(), np.array([])

        R_df = self.ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        R = R_df.values
        
        user_ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)

        k = min(R_demeaned.shape) - 1
        if k < 1: k = 1
        if k > 5: k = 5 
        
        U, sigma, Vt = svds(R_demeaned, k=k)
        
        sigma_diag = np.diag(sigma)
        predicted_ratings = np.dot(np.dot(U, sigma_diag), Vt) + user_ratings_mean.reshape(-1, 1)
        
        return pd.DataFrame(predicted_ratings, columns=R_df.columns, index=R_df.index), sigma

    def extract_all_genres(self):
        all_features = " ".join(self.items_df['features'].tolist())
        unique_genres = set(all_features.split())
        return sorted(list(unique_genres))

    def recommend_by_user(self, user_id, top_n=5, alpha=0.5, min_rating=0.0):
        if self.preds_df.empty or user_id not in self.preds_df.index:
            return []

        user_preds = self.preds_df.loc[user_id].sort_values(ascending=False)
        already_rated = self.ratings_df[self.ratings_df['user_id'] == user_id]['item_id'].tolist()
        candidates = user_preds[~user_preds.index.isin(already_rated)]
        
        user_hist = self.ratings_df[self.ratings_df['user_id'] == user_id]
        if user_hist.empty:
             fav_idx = None
        else:
            fav_item_id = user_hist.sort_values('rating', ascending=False).iloc[0]['item_id']
            fav_idx = self.id_map.get(fav_item_id)
        
        results = []
        for item_id, svd_score in candidates.items():
            if item_id not in self.id_map: continue
            
            avg_r = self.items_df[self.items_df['item_id'] == item_id]['avg_rating'].values[0]
            if avg_r < min_rating:
                continue

            idx = self.id_map[item_id]
            content_score = self.content_sim[fav_idx][idx] if fav_idx is not None else 0
            svd_norm = (np.clip(svd_score, 1, 5) - 1) / 4 
            hybrid_score = (alpha * svd_norm) + ((1 - alpha) * content_score)
            
            results.append({
                'id': item_id,
                'title': self.item_map[item_id],
                'link': self.link_map[item_id],
                'score': hybrid_score,
                'avg_rating': avg_r,
                'why': f"Similar to your taste ({self.items_df[self.items_df['item_id']==item_id]['features'].values[0]})"
            })
            
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]

    def recommend_by_genre(self, selected_genre, top_n=5, min_rating=0.0):
        if selected_genre == "All":
            matches = self.items_df
        else:
            matches = self.items_df[self.items_df['features'].str.contains(selected_genre, case=False)]
        
        if matches.empty:
            return []

        results = []
        for item_id in matches['item_id']:
            avg_rating = self.items_df[self.items_df['item_id'] == item_id]['avg_rating'].values[0]
            
            if avg_rating < min_rating:
                continue
            
            count = self.ratings_df[self.ratings_df['item_id'] == item_id]['rating'].count()
            score = avg_rating 
            
            if selected_genre == "All":
                 features = self.items_df[self.items_df['item_id'] == item_id]['features'].values[0]
                 why_text = f"{features} | {count} votes"
            else:
                 why_text = f"Genre match | {count} votes"

            results.append({
                'id': item_id,
                'title': self.item_map[item_id],
                'link': self.link_map[item_id],
                'score': score,
                'avg_rating': avg_rating,
                'why': why_text
            })
            
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]

# ==========================================
# 3. STREAMLIT UI - MAIN APP
# ==========================================

st.sidebar.title("🚀 Streamline Engine")
app_mode = st.sidebar.selectbox("Select Domain:", ["🎬 Movies", "🎵 Music"])

st.sidebar.subheader("🌍 Filters")
region_filter = st.sidebar.radio("Content Region:", ["All", "🇮🇳 Indian", "🌎 Western"])
min_rating_filter = st.sidebar.slider("⭐ Minimum Rating Filter", 0.0, 5.0, 3.5, 0.5)

st.sidebar.subheader("🧠 Strategy")
rec_strategy = st.sidebar.radio("Discovery Mode:", ["🎭 Genre / Mood", "👤 User Profile"])

if app_mode == "🎬 Movies":
    full_items_df, full_ratings_df = load_movie_data()
    st.title("🎬 Movie Discovery Platform")
    item_label = "Movie"
    btn_label = "Watch Trailer 🎬"
else:
    full_items_df, full_ratings_df = load_music_data()
    st.title("🎵 Music Discovery Platform")
    item_label = "Song"
    btn_label = "Play on Spotify 🟢"

# --- APPLY FILTERING ---
if region_filter == "All":
    items_df = full_items_df
    ratings_df = full_ratings_df
elif region_filter == "🇮🇳 Indian":
    items_df = full_items_df[full_items_df['region'] == 'Indian']
    valid_ids = items_df['item_id'].tolist()
    ratings_df = full_ratings_df[full_ratings_df['item_id'].isin(valid_ids)]
else: # Western
    items_df = full_items_df[full_items_df['region'] == 'Western']
    valid_ids = items_df['item_id'].tolist()
    ratings_df = full_ratings_df[full_ratings_df['item_id'].isin(valid_ids)]

engine = RecommenderEngine(items_df, ratings_df)

# --- USER INPUT LOGIC ---
selected_genre = None
selected_user = None

# Using columns to make the selector smaller (less wide)
col_input, col_space = st.columns([1, 3])

with col_input:
    if rec_strategy == "🎭 Genre / Mood":
        available_genres = engine.extract_all_genres()
        if available_genres:
            genre_options = ["All"] + available_genres
            selected_genre = st.selectbox(f"Select a {item_label} Category:", genre_options)
        else:
            st.warning("No items found for this region selection.")

    else: # By User Profile
        available_users = ratings_df['user_id'].unique()
        if len(available_users) > 0:
            selected_user = st.selectbox("Select User ID (Simulated):", sorted(available_users)[:20])
            alpha = st.slider("Algorithm Balance (Hybrid Weight)", 0.0, 1.0, 0.6)
        else:
            st.warning("No users found.")

# --- MAIN DISPLAY ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎯 Recommendations")
    
    recs = []
    
    if rec_strategy == "🎭 Genre / Mood" and selected_genre:
        recs = engine.recommend_by_genre(selected_genre, top_n=10, min_rating=min_rating_filter)
        if not recs:
            st.warning(f"No {item_label.lower()}s found in '{selected_genre}' with > {min_rating_filter} stars.")
            
    elif rec_strategy == "👤 User Profile" and selected_user:
        recs = engine.recommend_by_user(selected_user, top_n=10, alpha=alpha, min_rating=min_rating_filter)
        if not recs:
            st.warning("No recommendations found. Try lowering the Rating Filter.")

    # Render Cards
    if recs:
        for r in recs:
            with st.container():
                # Columns: [Icon] [Info] [Score/Link]
                c_icon, c_info, c_action = st.columns([0.1, 0.65, 0.25])
                
                c_icon.markdown("## 📀")
                
                c_info.markdown(f"**{r['title']}**")
                c_info.caption(r['why'])
                
                # Action Column: Rating + Link Button
                c_action.metric("Avg Rating", f"⭐ {r['avg_rating']:.1f}")
                c_action.link_button(btn_label, r['link'])
                
                st.divider()

with col2:
    st.markdown("### 📊 Database Stats")
    st.metric("Total Items", len(items_df))
    st.metric("Total Ratings", len(ratings_df))
    
    st.divider()
    
    st.markdown("### 🔥 Trending Now")
    if not ratings_df.empty:
        # 1. Calculate Top 5 Highest Rated Items
        top_rated = ratings_df.groupby('item_id')['rating'].mean().sort_values(ascending=False).head(5)
        top_titles = [engine.item_map.get(i, "Unknown") for i in top_rated.index]
        
        # Plot: Horizontal Bar Chart
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(x=top_rated.values, y=top_titles, palette="viridis", ax=ax, hue=top_titles, legend=False)
        ax.set_xlabel("Average Rating (Stars)")
        ax.set_xlim(0, 5.5) # Give space for text
        
        # Add values to bars
        for i, v in enumerate(top_rated.values):
            ax.text(v + 0.1, i, f"{v:.1f}", color='black', va='center')
            
        sns.despine()
        st.pyplot(fig)
        
        st.divider()

        # 2. Rating Distribution Chart
        st.markdown("### 📈 Voting Trends")
        rating_counts = ratings_df['rating'].value_counts().sort_index()
        
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="rocket", ax=ax2, hue=rating_counts.index, legend=False)
        ax2.set_xlabel("Star Rating Given")
        ax2.set_ylabel("Number of Votes")
        sns.despine()
        st.pyplot(fig2)
        
    else:
        st.write("No data available.")

st.markdown("---")
st.caption(f"Showing results for {region_filter} Region • {rec_strategy} Mode")