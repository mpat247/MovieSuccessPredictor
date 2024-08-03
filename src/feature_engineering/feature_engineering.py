import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import re
import datetime
from textstat import textstat

# Enhanced logging function with timestamps
def log(message):
    print(f"[{datetime.datetime.now()}] [LOG]: {message}")

# Set data paths
data_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')
feature_dir = os.path.join(os.path.dirname(__file__), '../../data/features')
os.makedirs(feature_dir, exist_ok=True)

# Load processed data
log("Loading processed IMDb basics data...")
imdb_basics = pd.read_csv(os.path.join(data_dir, 'imdb_basics_processed.csv'))
log("First 15 rows of IMDb basics data loaded:")
print(imdb_basics.head(15))

log("Loading processed IMDb ratings data...")
imdb_ratings = pd.read_csv(os.path.join(data_dir, 'imdb_ratings_processed.csv'))
log("First 15 rows of IMDb ratings data loaded:")
print(imdb_ratings.head(15))

log("Loading processed TMDb data...")
tmdb_data = pd.read_csv(os.path.join(data_dir, 'tmdb_data_processed.csv'))
log("First 15 rows of TMDb data loaded:")
print(tmdb_data.head(15))

log("Loading movie scripts...")
scripts_dir = os.path.join(os.path.dirname(__file__), '../../data/scripts')
script_files = [f for f in os.listdir(scripts_dir) if f.endswith('.txt')]

# Feature Engineering

# 1. Merging IMDb basics and ratings data
log("Merging IMDb basics and ratings data...")
imdb_data = pd.merge(imdb_basics, imdb_ratings, on='tconst')
log("First 15 rows of merged IMDb data:")
print(imdb_data.head(15))

# 2. IMDb Features
log("Processing IMDb features...")

# One-hot encoding titleType
imdb_data = pd.concat([imdb_data, pd.get_dummies(imdb_data['titleType'], prefix='titleType')], axis=1)

# Normalizing runtimeMinutes
scaler = StandardScaler()
imdb_data['runtimeMinutes_normalized'] = scaler.fit_transform(imdb_data[['runtimeMinutes']].fillna(0))

# One-hot encoding genres
genres = imdb_data['genres'].str.get_dummies(sep=',')
imdb_data = pd.concat([imdb_data, genres], axis=1)

# Extracting and one-hot encoding release season
def extract_season(start_year):
    try:
        year = int(start_year)
        month = datetime.date(year, 1, 1).month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
    except ValueError:
        return 'Unknown'

imdb_data['releaseSeason'] = imdb_data['startYear'].apply(lambda x: extract_season(x))
imdb_data = pd.concat([imdb_data, pd.get_dummies(imdb_data['releaseSeason'], prefix='season')], axis=1)

log("First 15 rows of IMDb data after feature processing:")
print(imdb_data.head(15))

# 3. TMDb Features
log("Processing TMDb features...")

# Extracting cast and crew features
def extract_name_list(data):
    try:
        return [person['name'] for person in eval(data)]
    except:
        return []

tmdb_data['cast_names'] = tmdb_data['cast'].apply(extract_name_list)
tmdb_data['crew_names'] = tmdb_data['crew'].apply(extract_name_list)

log("First 15 rows of TMDb data after extracting cast and crew names:")
print(tmdb_data.head(15))

# Extracting Director Popularity (example logic, customize based on data availability)
# This is a placeholder as actual director popularity may require external data sources
tmdb_data['director_popularity'] = tmdb_data['crew_names'].apply(lambda x: len(x))  # Example: number of crew members

log("First 15 rows of TMDb data after processing director popularity:")
print(tmdb_data.head(15))

# 4. Script Features
log("Processing script features...")

def get_script_features(script_path):
    with open(script_path, 'r', encoding='utf-8') as file:
        text = file.read()
        
        # Word Count
        word_count = len(re.findall(r'\w+', text))
        
        # Sentiment Analysis Scores
        sentiment = TextBlob(text).sentiment
        
        # Readability Score using textstat
        try:
            flesch_kincaid = textstat.flesch_kincaid_grade(text)
        except:
            flesch_kincaid = np.nan
        
        # Placeholder for genre indicators or other textual features
        # Additional feature extraction can be done here
        
        return {
            'word_count': word_count,
            'sentiment_polarity': sentiment.polarity,
            'sentiment_subjectivity': sentiment.subjectivity,
            'flesch_kincaid': flesch_kincaid
        }

# Process all script files
script_features = []

for script_file in script_files:  # Removed limit to process all scripts
    script_path = os.path.join(scripts_dir, script_file)
    features = get_script_features(script_path)
    features['script_name'] = script_file
    script_features.append(features)

script_features_df = pd.DataFrame(script_features)
log("First 15 rows of script features:")
print(script_features_df.head(15))

# Saving final features
log("Saving final feature set...")

imdb_features_path = os.path.join(feature_dir, 'imdb_features.csv')
imdb_data.to_csv(imdb_features_path, index=False)
log(f"IMDb features saved to '{imdb_features_path}'")

tmdb_features_path = os.path.join(feature_dir, 'tmdb_features.csv')
tmdb_data.to_csv(tmdb_features_path, index=False)
log(f"TMDb features saved to '{tmdb_features_path}'")

script_features_path = os.path.join(feature_dir, 'final_script_features.csv')
script_features_df.to_csv(script_features_path, index=False)
log(f"Script features saved to '{script_features_path}'")

log("Feature engineering completed successfully.")
