import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

# Set data paths
data_dir = os.path.join(os.path.dirname(__file__), '../../data/cleaned')
feature_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')
os.makedirs(feature_dir, exist_ok=True)

imdb_basics_path = os.path.join(data_dir, 'imdb_basics_cleaned.csv')
imdb_ratings_path = os.path.join(data_dir, 'imdb_ratings_cleaned.csv')
tmdb_data_path = os.path.join(data_dir, 'tmdb_data_cleaned.csv')

# Logging function
def log(message):
    print(f"[LOG]: {message}")

# Load data
log("Loading IMDb basics data...")
imdb_basics = pd.read_csv(imdb_basics_path)
log(f"IMDb basics data loaded with shape: {imdb_basics.shape}")

log("Loading IMDb ratings data...")
imdb_ratings = pd.read_csv(imdb_ratings_path)
log(f"IMDb ratings data loaded with shape: {imdb_ratings.shape}")

log("Loading TMDb data...")
tmdb_data = pd.read_csv(tmdb_data_path)
log(f"TMDb data loaded with shape: {tmdb_data.shape}")

# Merge IMDb basics and ratings
log("Merging IMDb basics and ratings data...")
imdb_data = imdb_basics.merge(imdb_ratings, on='tconst')
log(f"Merged IMDb data shape: {imdb_data.shape}")

# One-hot encode genres
log("One-hot encoding genres for IMDb data...")
genres_onehot = imdb_data['genres'].str.get_dummies(sep=',')
imdb_data = pd.concat([imdb_data, genres_onehot], axis=1)
log(f"One-hot encoded genres added. IMDb data shape: {imdb_data.shape}")

# One-hot encode TMDb genres
log("One-hot encoding genres for TMDb data...")
tmdb_data['genres'] = tmdb_data['genres'].apply(lambda x: x.split(','))
tmdb_genres_onehot = tmdb_data['genres'].explode().str.get_dummies().groupby(level=0).sum()
tmdb_data = pd.concat([tmdb_data, tmdb_genres_onehot], axis=1)
log(f"One-hot encoded genres added. TMDb data shape: {tmdb_data.shape}")

# Feature extraction: Year of release
log("Extracting year of release and decade from IMDb data...")
imdb_data['startYear'] = pd.to_numeric(imdb_data['startYear'], errors='coerce')
imdb_data['decade'] = (imdb_data['startYear'] // 10) * 10
log("Year and decade features extracted.")

# Save processed features
log("Saving processed IMDb features...")
imdb_features_path = os.path.join(feature_dir, 'imdb_features.csv')
imdb_data.to_csv(imdb_features_path, index=False)
log(f"Processed IMDb features saved to: {imdb_features_path}")

log("Saving processed TMDb features...")
tmdb_features_path = os.path.join(feature_dir, 'tmdb_features.csv')
tmdb_data.to_csv(tmdb_features_path, index=False)
log(f"Processed TMDb features saved to: {tmdb_features_path}")
