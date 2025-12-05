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

# --- SIGNATURE: CROMIN ---
# UPDATED: Moved to TOP RIGHT corner (below header to avoid overlap)
st.markdown("""
    <div style='position: fixed; top: 90px; right: 20px; z-index: 10000; 
    font-family: "Segoe UI", sans-serif; font-weight: 900; font-size: 1.2rem; 
    color: #00d2ff; text-shadow: 0px 0px 5px rgba(0, 210, 255, 0.6); 
    background-color: rgba(15, 23, 42, 0.5); padding: 5px 15px; 
    border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1); 
    pointer-events: none;'>
        CROMIN
    </div>
""", unsafe_allow_html=True)

# Custom CSS for a professional look with Background Image
st.markdown("""
    <style>
    /* --- BACKGROUND SETTINGS --- */
    
    /* 1. Default (PC/Desktop/Tablet) Background */
    .stApp {
        /* Wide Landscape Image */
        background-image: url("https://wallpapercat.com/w/full/0/a/8/319915-3840x2160-desktop-4k-iron-man-background.jpg");
        background-attachment: fixed;
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
    }
    
    /* 2. Mobile Specific Background (Screens smaller than 768px) */
    @media only screen and (max-width: 768px) {
        .stApp {
            /* Vertical Portrait Image - Darker & Better for Phones */
            background-image: url("https://i.pinimg.com/474x/9d/96/84/9d968452bdb775ec7edea0c1bb7af701.jpg?nii=t") !important;
            background-attachment: scroll; /* Better scroll performance on mobile */
            background-position: center center;
            background-size: cover;
        }
    }
    
    /* Main Content Card - Dark Glassmorphism */
    div.block-container {
        background-color: rgba(15, 23, 42, 0.85); /* Dark Navy with opacity */
        border-radius: 15px;
        padding: 2rem !important;
        margin-top: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }

    /* Headings (Cyan for contrast) */
    h1, h2, h3 {
        color: #00d2ff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }
    
    /* Standard Text (White) */
    p, label, .stMarkdown, div {
        color: #e2e8f0 !important;
    }
    
    /* Metrics/Stats Cards styling */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
    }
    
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important; /* Muted text for labels */
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #00d2ff !important; /* Bright value */
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
    }
    
    /* Sidebar Styling (Dark) */
    section[data-testid="stSidebar"] {
        background-color: #0f172a; /* Match main theme */
        border-right: 1px solid #1e293b;
    }

    /* Force White Text for all Sidebar elements */
    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    
    /* Fix for Selectbox inside sidebar */
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div {
        color: white !important;
        background-color: #1e293b;
    }
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
    n_items = len(item_ids) # Total available items
    
    # Create specific personas for demo purposes (Users 1-5)
    for i in range(1, 6):
        target_count = random.randint(10, 20)
        num_ratings = min(target_count, n_items)
        
        if num_ratings > 0:
            chosen_items = random.sample(item_ids, num_ratings)
            for item in chosen_items:
                data.append({'user_id': i, 'item_id': item, 'rating': random.randint(3, 5)})
            
    # Generate background noise users (Users 6-100)
    for u in range(6, num_users + 1):
        target_count = random.randint(5, 25) 
        num_ratings = min(target_count, n_items)
        
        if num_ratings > 0:
            chosen_items = random.sample(item_ids, num_ratings)
            for item in chosen_items:
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.35, 0.3])
                data.append({'user_id': u, 'item_id': item, 'rating': rating})
            
    return pd.DataFrame(data)

def get_smart_link(title, type='movie'):
    """Generates a search link for Spotify (Music) or YouTube (Movies/Games)"""
    safe_title = urllib.parse.quote(title)
    if type == 'music':
        return f"https://open.spotify.com/search/{safe_title}"
    elif type == 'game':
        return f"https://www.youtube.com/results?search_query={safe_title}+gameplay+trailer"
    else:
        # For movies
        return f"https://www.youtube.com/results?search_query={safe_title}+trailer"

@st.cache_data
def load_movie_data():
    """Expanded Database for Movies (Western, Indian, Anime) - 150 Items"""
    
    # 50 Western Movies
    western_titles = [
        'The Matrix', 'Inception', 'Toy Story', 'Finding Nemo', 'The Godfather', 'Pulp Fiction', 'Interstellar', 'John Wick', 'Up', 'Fight Club',
        'The Dark Knight', 'Spirited Away', 'Parasite', 'Avengers: Endgame', 'Joker', 'Titanic', 'Avatar', 'The Lion King', 'Gladiator', 'Forrest Gump',
        'Dune: Part Two', 'Oppenheimer', 'Barbie', 'Spider-Man: Across the Spider-Verse', 'The Batman', 'Top Gun: Maverick', 'Everything Everywhere All At Once', 'Knives Out', 'Mad Max: Fury Road', 'Logan',
        'Deadpool', 'Guardians of the Galaxy', 'Black Panther', 'Iron Man', 'Thor: Ragnarok', 'Captain America: Civil War', 'Doctor Strange', 'Ant-Man', 'The Avengers', 'Wonder Woman',
        'Harry Potter and the Sorcerer\'s Stone', 'The Lord of the Rings: The Fellowship of the Ring', 'Star Wars: A New Hope', 'Jurassic Park', 'Back to the Future', 'The Shawshank Redemption', 'Schindler\'s List', 'Goodfellas', 'Saving Private Ryan', 'The Silence of the Lambs'
    ]
    
    # 50 Indian Movies
    indian_titles = [
        '3 Idiots', 'RRR', 'Dangal', 'Sholay', 'Lagaan', 'Baahubali: The Beginning', 'Baahubali 2: The Conclusion', 'Kantara', 'Zindagi Na Milegi Dobara', 'Gully Boy',
        'Drishyam', 'Dilwale Dulhania Le Jayenge', 'Swades', 'Chak De! India', 'Queen', 'Andhadhun', 'Gangs of Wasseypur', 'PK', 'Bajrangi Bhaijaan', 'K.G.F: Chapter 1',
        'K.G.F: Chapter 2', 'Pushpa: The Rise', 'Taare Zameen Par', 'Munna Bhai M.B.B.S.', 'Hera Pheri', 'Drishyam 2', 'Animal', 'Jawan', 'Pathaan', 'Stree',
        'Tumbbad', '12th Fail', 'Vikram', 'Kaithi', 'Master', 'Leo', 'Jailer', 'Salaar', 'Sita Ramam', 'Hi Nanna',
        'Sanju', 'Uri: The Surgical Strike', 'Article 15', 'Badhaai Ho', 'Chhichhore', 'Super 30', 'Kabir Singh', 'War', 'Tanhaji', 'Bhool Bhulaiyaa 2'
    ]
    
    # 50 Anime Movies/Series (Treated as titles)
    anime_titles = [
        'Your Name', 'Demon Slayer: Mugen Train', 'Jujutsu Kaisen 0', 'Weathering with You', 'Suzume', 'A Silent Voice', 'My Neighbor Totoro', 'Attack on Titan: Chronicle', 'One Piece Film: Red', 'Naruto: The Last',
        'Spirited Away', 'Howl\'s Moving Castle', 'Princess Mononoke', 'Akira', 'Ghost in the Shell', 'Cowboy Bebop: The Movie', 'Evangelion: 3.0+1.0 Thrice Upon a Time', 'The Girl Who Leapt Through Time', 'Wolf Children', '5 Centimeters Per Second',
        'Dragon Ball Super: Broly', 'Dragon Ball Super: Super Hero', 'One Piece: Stampede', 'My Hero Academia: Heroes Rising', 'Hunter x Hunter: The Last Mission', 'Bleach: Hell Verse', 'Fullmetal Alchemist: The Conqueror of Shamballa', 'Violet Evergarden: The Movie', 'I Want to Eat Your Pancreas', 'Perfect Blue',
        'Paprika', 'Tokyo Godfathers', 'The Boy and the Heron', 'Grave of the Fireflies', 'Ponyo', 'Kiki\'s Delivery Service', 'Castle in the Sky', 'Nausicaä of the Valley of the Wind', 'Summer Wars', 'Redline',
        'Sword Art Online: Ordinal Scale', 'No Game No Life: Zero', 'Rascal Does Not Dream of a Dreaming Girl', 'Josee, the Tiger and the Fish', 'Words Bubble Up Like Soda Pop', 'Belle', 'Promare', 'Ninja Scroll', 'Vampire Hunter D: Bloodlust', 'Steins;Gate: The Movie'
    ]
    
    all_titles = western_titles + indian_titles + anime_titles
    
    # Generate Features roughly
    western_features = ['Sci-Fi Action Adventure'] * 10 + ['Drama Crime Thriller'] * 10 + ['Action Superhero Sci-Fi'] * 20 + ['Classic Drama Adventure'] * 10
    indian_features = ['Bollywood Drama Comedy'] * 15 + ['Action Thriller Crime'] * 15 + ['Romance Drama Musical'] * 10 + ['Historical Epic Action'] * 10
    anime_features = ['Anime Romance Fantasy'] * 15 + ['Anime Action Supernatural'] * 20 + ['Anime Sci-Fi Mecha'] * 5 + ['Anime Drama Slice-of-Life'] * 10
    
    all_features = western_features + indian_features + anime_features
    all_regions = ['Western'] * 50 + ['Indian'] * 50 + ['Anime'] * 50
    
    items = pd.DataFrame({
        'item_id': range(1, 151),
        'title': all_titles,
        'features': all_features, 
        'region': all_regions,
        'link': [get_smart_link(t, 'movie') for t in all_titles]
    })
    
    items['features'] = items.apply(lambda x: x['features'] + " " + x['region'], axis=1)
    
    ratings = generate_ratings(items)
    return items, ratings

@st.cache_data
def load_music_data():
    """Expanded Database for Music (Western, Indian + Punjabi) - 100 Items"""
    
    # 50 Western Songs
    western_titles = [
        'Bohemian Rhapsody', 'Shape of You', 'Smells Like Teen Spirit', 'Hotel California', 'Blinding Lights', 'Rolling in the Deep', 'Billie Jean', 'Sicko Mode', 'Imagine', 'Lose Yourself',
        'Uptown Funk', 'Someone Like You', 'Despacito', 'Believer', 'Starboy', 'Hips Don\'t Lie', 'Viva La Vida', 'Thinking Out Loud', 'Bad Guy', 'Levitating',
        'As It Was', 'Stay', 'Heat Waves', 'Seven Nation Army', 'Mr. Brightside', 'Sweet Child O\' Mine', 'Wonderwall', 'In The End', 'Numb', 'Boulevard of Broken Dreams',
        'Thriller', 'Beat It', 'Smooth Criminal', 'Like a Prayer', 'Vogue', 'Toxic', 'Baby One More Time', 'I Want It That Way', 'Bye Bye Bye', 'Umbrella',
        'Single Ladies', 'Halo', 'Crazy in Love', 'Rolling', 'Empire State of Mind', 'Firework', 'Roar', 'Dark Horse', 'Shake It Off', 'Blank Space'
    ]
    
    # 50 Indian/Punjabi Songs
    indian_titles = [
        'Kesariya', 'Jai Ho', 'Naatu Naatu', 'Tum Hi Ho', 'Kun Faya Kun', 'Apna Time Aayega', 'Chaiyya Chaiyya', 'Kabira', 'Kal Ho Naa Ho', 'Pasoori',
        'Dil Diyan Gallan', 'Raabta', 'Ilahi', 'Agar Tum Saath Ho', 'Channa Mereya', 'Khalibali', 'Brown Munde', 'Lover', 'Excuses', 'Tera Ghata',
        'Maan Meri Jaan', 'King of Kotha', 'Hukum', 'Chaleya', 'Jhoome Jo Pathaan', 'Saami Saami', 'Srivalli', 'Oo Antava', 'Apna Bana Le', 'Pehle Bhi Main',
        'Softly - Karan Aujla', 'With You - AP Dhillon', '295 - Sidhu Moose Wala', 'Lemonade - Diljit Dosanjh', 'Check It Out - Parmish Verma',
        'Elevated - Shubh', 'We Rollin - Shubh', 'Summer High - AP Dhillon', 'G.O.A.T. - Diljit Dosanjh', 'So High - Sidhu Moose Wala',
        'The Last Ride', 'Levels', 'Same Beef', 'Old Skool', 'Daru Badnaam', 'Lahore', 'High Rated Gabru', 'Suit Suit', 'Patiala Peg', '3 Peg'
    ]

    all_titles = western_titles + indian_titles
    
    items = pd.DataFrame({
        'item_id': range(201, 301),
        'title': all_titles,
        'features': ['Western Pop Rock'] * 50 + ['Indian Bollywood Punjabi'] * 50,
        'region': ['Western'] * 50 + ['Indian'] * 50,
        'link': [get_smart_link(t, 'music') for t in all_titles]
    })
    
    items['features'] = items.apply(lambda x: x['features'] + " Hit", axis=1)
    
    ratings = generate_ratings(items)
    return items, ratings

@st.cache_data
def load_game_data():
    """Database for PC Games - 50 Items"""
    titles = [
        'Grand Theft Auto V', 'The Witcher 3: Wild Hunt', 'Elden Ring', 'Red Dead Redemption 2', 'Cyberpunk 2077',
        'Valorant', 'Counter-Strike 2', 'Call of Duty: Modern Warfare III', 'Apex Legends', 'Overwatch 2',
        'Minecraft', 'Stardew Valley', 'Terraria', 'Hades', 'Hollow Knight',
        'League of Legends', 'Dota 2', 'Baldur\'s Gate 3', 'Civilization VI', 'Age of Empires IV',
        'God of War', 'Spider-Man Remastered', 'Horizon Zero Dawn', 'Ghost of Tsushima', 'Uncharted 4',
        'The Last of Us Part I', 'Resident Evil 4 Remake', 'Dead Space', 'Doom Eternal', 'Half-Life 2',
        'Portal 2', 'Team Fortress 2', 'Left 4 Dead 2', 'Destiny 2', 'Warframe',
        'Fortnite', 'PUBG: Battlegrounds', 'Rocket League', 'FIFA 24', 'NBA 2K24',
        'Forza Horizon 5', 'Gran Turismo 7', 'Need for Speed Unbound', 'Assassin\'s Creed Mirage', 'Far Cry 6',
        'Starfield', 'Fallout 4', 'Skyrim', 'Mass Effect Legendary Edition', 'Bioshock Infinite'
    ]
    
    categories = [
        'Open World', 'RPG', 'RPG', 'Open World', 'RPG',
        'Shooter', 'Shooter', 'Shooter', 'Shooter', 'Shooter',
        'Indie/Sandbox', 'Indie/Sandbox', 'Indie/Sandbox', 'Indie/Sandbox', 'Indie/Sandbox',
        'Strategy', 'Strategy', 'RPG', 'Strategy', 'Strategy',
        'Action', 'Action', 'Action', 'Action', 'Action',
        'Action', 'Horror', 'Horror', 'Shooter', 'Shooter',
        'Puzzle', 'Shooter', 'Shooter', 'Shooter', 'Shooter',
        'Shooter', 'Shooter', 'Sports', 'Sports', 'Sports',
        'Racing', 'Racing', 'Racing', 'Open World', 'Open World',
        'RPG', 'RPG', 'RPG', 'RPG', 'Shooter'
    ]
    
    items = pd.DataFrame({
        'item_id': range(401, 451),
        'title': titles,
        'features': [c + " Game" for c in categories],
        'region': categories, 
        'link': [get_smart_link(t, 'game') for t in titles]
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
        self.link_map = dict(zip(items_df['item_id'], items_df['link']))
        self.id_map = {mid: i for i, mid in enumerate(items_df['item_id'])}
        self.reverse_id_map = {i: mid for mid, i in self.id_map.items()}
        
        if not ratings_df.empty:
            avg_ratings = ratings_df.groupby('item_id')['rating'].mean()
            self.items_df['avg_rating'] = self.items_df['item_id'].map(avg_ratings).fillna(0)
        else:
            self.items_df['avg_rating'] = 0
        
        if not self.items_df.empty:
            self.content_sim = self._calculate_content_similarity()
        else:
            self.content_sim = np.array([])
            
        self.preds_df, self.sigma = self._calculate_collaborative_filtering()

    def _calculate_content_similarity(self):
        if self.items_df.empty:
            return np.array([])
            
        tfidf = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = tfidf.fit_transform(self.items_df['features'])
            return cosine_similarity(tfidf_matrix, tfidf_matrix)
        except ValueError:
            return np.zeros((len(self.items_df), len(self.items_df)))

    def _calculate_collaborative_filtering(self):
        if self.ratings_df.empty:
            return pd.DataFrame(), np.array([])

        R_df = self.ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        R = R_df.values
        
        if R.shape[0] < 2 or R.shape[1] < 2:
             return pd.DataFrame(), np.array([])

        user_ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)

        k = min(R_demeaned.shape) - 1
        if k < 1: k = 1
        if k > 5: k = 5 
        
        try:
            U, sigma, Vt = svds(R_demeaned, k=k)
            sigma_diag = np.diag(sigma)
            predicted_ratings = np.dot(np.dot(U, sigma_diag), Vt) + user_ratings_mean.reshape(-1, 1)
            return pd.DataFrame(predicted_ratings, columns=R_df.columns, index=R_df.index), sigma
        except:
            return pd.DataFrame(), np.array([])

    def extract_all_genres(self):
        if self.items_df.empty:
            return []
        all_features = " ".join(self.items_df['features'].tolist())
        unique_genres = set(all_features.split())
        return sorted(list(unique_genres))

    def search_items(self, query):
        if not query or self.items_df.empty:
            return []
        matches = self.items_df[self.items_df['title'].str.contains(query, case=False, regex=False)]
        return matches['item_id'].tolist()

    def get_similar_items(self, item_id, top_n=5):
        if item_id not in self.id_map or self.content_sim.size == 0:
            return []
        
        idx = self.id_map[item_id]
        sim_scores = list(enumerate(self.content_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        sim_scores = sim_scores[1:top_n+1]
        
        results = []
        for i, score in sim_scores:
            similar_item_id = self.reverse_id_map[i]
            avg_r = self.items_df[self.items_df['item_id'] == similar_item_id]['avg_rating'].values[0]
            
            results.append({
                'id': similar_item_id,
                'title': self.item_map[similar_item_id],
                'link': self.link_map[similar_item_id],
                'score': score * 5, 
                'avg_rating': avg_r,
                'why': f"Similar content ({int(score*100)}% match)"
            })
        return results

    def recommend_by_user(self, user_id, top_n=5, alpha=0.5):
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

            idx = self.id_map[item_id]
            content_score = self.content_sim[fav_idx][idx] if fav_idx is not None and self.content_sim.size > 0 else 0
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

    def recommend_by_genre(self, selected_genre, top_n=5):
        if self.items_df.empty:
            return []
            
        if selected_genre == "All":
            matches = self.items_df
        else:
            matches = self.items_df[self.items_df['features'].str.contains(selected_genre, case=False)]
        
        if matches.empty:
            return []

        results = []
        for item_id in matches['item_id']:
            avg_rating = self.items_df[self.items_df['item_id'] == item_id]['avg_rating'].values[0]
            
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
app_mode = st.sidebar.selectbox("Select Domain:", ["🎬 Movies", "🎵 Music", "🎮 PC Games"])

st.sidebar.subheader("🌍 Filters")

# --- DYNAMIC REGION FILTER LOGIC ---
# UPDATED: Moved emojis BEFORE the text
if app_mode == "🎬 Movies":
    available_regions = ["All", "🇮🇳 Indian", "🌎 Western", "🇯🇵 Anime"]
    btn_label = "Watch Trailer 🎬"
elif app_mode == "🎵 Music":
    available_regions = ["All", "🇮🇳 Indian", "🌎 Western"]
    btn_label = "Play on Spotify 🟢" 
else: # Games
    available_regions = ["All", "Shooter", "Open World", "Strategy", "Indie/Sandbox", "Action", "RPG", "Sports", "Racing"]
    btn_label = "Watch Trailer 🎬"

# Create the Radio Button
region_filter = st.sidebar.radio("Content Region/Category:", available_regions)

st.sidebar.subheader("🧠 Strategy")
rec_strategy = st.sidebar.radio("Discovery Mode:", ["🔍 Search", "🎭 Genre / Mood", "👤 User Profile"])

# Load correct data based on selection
if app_mode == "🎬 Movies":
    full_items_df, full_ratings_df = load_movie_data()
    st.title("🎬 Movie Discovery Platform")
    item_label = "Movie"
elif app_mode == "🎵 Music":
    full_items_df, full_ratings_df = load_music_data()
    st.title("🎵 Music Discovery Platform")
    item_label = "Song"
else: # Games
    full_items_df, full_ratings_df = load_game_data()
    st.title("🎮 PC Game Recommendations")
    item_label = "Game"

# --- APPLY FILTERING (FIXED LOGIC) ---
items_df = full_items_df.copy()
ratings_df = full_ratings_df.copy()

# Map the UI Labels to Data Labels
# UPDATED: Keys updated to match the new emoji positions (Emoji First)
region_map = {
    "🇮🇳 Indian": "Indian",
    "🌎 Western": "Western",
    "🇯🇵 Anime": "Anime",
}

if region_filter != "All":
    data_label = region_map.get(region_filter, region_filter)
    items_df = full_items_df[full_items_df['region'] == data_label]

# Sync ratings with filtered items
valid_ids = items_df['item_id'].tolist()
ratings_df = full_ratings_df[full_ratings_df['item_id'].isin(valid_ids)]

# Initialize Engine with SAFE data
engine = RecommenderEngine(items_df, ratings_df)

# --- USER INPUT LOGIC ---
selected_genre = None
selected_user = None
search_query = None

col_input, col_space = st.columns([1, 3])

with col_input:
    if rec_strategy == "🔍 Search":
        all_titles = sorted(items_df['title'].tolist()) if not items_df.empty else []
        search_query = st.selectbox(
            f"Search {item_label}:",
            options=all_titles,
            index=None, 
            placeholder=f"Type {item_label} name...",
        )
    
    elif rec_strategy == "🎭 Genre / Mood":
        available_genres = engine.extract_all_genres()
        if available_genres:
            genre_options = ["All"] + available_genres
            selected_genre = st.selectbox(f"Select a {item_label} Category:", genre_options)
        else:
            st.warning(f"No {item_label.lower()}s found for {region_filter}.")

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
    
    # CASE: SEARCH
    if rec_strategy == "🔍 Search":
        if search_query:
            found_ids = engine.search_items(search_query)
            
            if found_ids:
                item_id = found_ids[0]
                match_row = items_df[items_df['item_id'] == item_id].iloc[0]
                st.success(f"Found Match: **{match_row['title']}**")
                
                with st.container():
                     c1, c2, c3 = st.columns([0.1, 0.65, 0.25])
                     c1.markdown("## 🎯")
                     with c2:
                         st.markdown(f"**{match_row['title']}**")
                         st.caption(f"Genre: {match_row['features']}")
                     c3.link_button(btn_label, match_row['link'])
                     st.divider()

                st.markdown("### ✨ More Like This:")
                recs = engine.get_similar_items(item_id, top_n=5)
                
            else:
                st.warning("No exact matches found. Try a different keyword.")
        else:
            st.info(f"👆 Type a {item_label.lower()} name above to find similar content!")
            if not items_df.empty:
                st.markdown("### 🌟 Or check out these Top Rated items:")
                recs = engine.recommend_by_genre("All", top_n=5)

    elif rec_strategy == "🎭 Genre / Mood" and selected_genre:
        recs = engine.recommend_by_genre(selected_genre, top_n=10)
        if not recs:
            st.warning(f"No {item_label.lower()}s found in '{selected_genre}'.")
            
    elif rec_strategy == "👤 User Profile" and selected_user:
        recs = engine.recommend_by_user(selected_user, top_n=10, alpha=alpha)
        if not recs:
            st.warning("No recommendations found.")

    # Render Cards
    if recs:
        for r in recs:
            with st.container():
                c_icon, c_info, c_action = st.columns([0.1, 0.65, 0.25])
                
                c_icon.markdown("## 📀")
                
                with c_info:
                    st.markdown(f"**{r['title']}**")
                    st.caption(r['why'])
                
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
        top_rated = ratings_df.groupby('item_id')['rating'].mean().sort_values(ascending=False).head(5)
        top_titles = [engine.item_map.get(i, "Unknown") for i in top_rated.index]
        
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none") 
        
        sns.barplot(x=top_rated.values, y=top_titles, palette="viridis", ax=ax, hue=top_titles, legend=False)
        
        ax.set_xlabel("Average Rating (Stars)", color="white")
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_xlim(0, 5.5) 
        
        for i, v in enumerate(top_rated.values):
            ax.text(v + 0.1, i, f"{v:.1f}", color='white', va='center')
            
        sns.despine()
        st.pyplot(fig)
        
        st.divider()

        st.markdown("### 📈 Voting Trends")
        rating_counts = ratings_df['rating'].value_counts().sort_index()
        
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        fig2.patch.set_alpha(0)
        ax2.set_facecolor("none") 
        
        sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="rocket", ax=ax2, hue=rating_counts.index, legend=False)
        
        ax2.set_xlabel("Star Rating Given", color="white")
        ax2.set_ylabel("Number of Votes", color="white")
        ax2.tick_params(colors='white')
        
        sns.despine()
        st.pyplot(fig2)
        
    else:
        st.write("No data available.")

st.markdown("---")
st.caption(f"Showing results for {region_filter} Region • {rec_strategy} Mode")