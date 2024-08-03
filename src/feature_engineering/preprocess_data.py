import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Logging function
def log(message):
    print(f"[{pd.Timestamp.now()}] [LOG]: {message}")

# Define data paths
base_dir = 'D:\\manav\\Documents\\Engineering\\Masters\\EE8206\\Project\\MovieSuccessPredictor'
data_dir = os.path.join(base_dir, 'data', 'processed')
imdb_data_path = os.path.join(data_dir, 'imdb_basics_processed.csv')
tmdb_data_path = os.path.join(data_dir, 'tmdb_data_processed.csv')

# Load and limit data
log("Loading IMDb and TMDb data...")
imdb_data = pd.read_csv(imdb_data_path).head(3000)
tmdb_data = pd.read_csv(tmdb_data_path).head(150)  # Adjust to 150 for consistency

log(f"IMDb data loaded with shape: {imdb_data.shape}")
log(f"TMDb data loaded with shape: {tmdb_data.shape}")

# Combine and select features
log("Combining IMDb and TMDb data...")
combined_data = pd.concat([imdb_data, tmdb_data], axis=0, ignore_index=True)
log(f"Combined data shape: {combined_data.shape}")

# Check columns
log(f"Columns in combined data: {list(combined_data.columns)}")

# Define numerical and categorical features
numerical_columns = ['averageRating', 'numVotes', 'runtimeMinutes_normalized']
# Make sure these columns exist in combined_data
missing_numerical_columns = [col for col in numerical_columns if col not in combined_data.columns]

if missing_numerical_columns:
    log(f"Missing numerical columns: {missing_numerical_columns}")
    raise KeyError(f"Missing expected numerical columns: {missing_numerical_columns}")

categorical_columns = [
    'titleType_movie', 'titleType_short', 'titleType_tvEpisode', 'titleType_tvMiniSeries', 'titleType_tvMovie',
    'titleType_tvSeries', 'titleType_tvShort', 'titleType_tvSpecial', 'titleType_video', 'titleType_videoGame',
    'season_Winter'
]

# Normalizing numerical features
log("Normalizing numerical features...")
scaler = StandardScaler()
combined_data[numerical_columns] = scaler.fit_transform(combined_data[numerical_columns])
log("Normalization completed.")
print(combined_data[numerical_columns].head(15))

# One-hot encoding categorical features
log("One-hot encoding categorical features...")
combined_data = pd.get_dummies(combined_data, columns=categorical_columns, drop_first=True)
log("One-hot encoding completed.")
print(combined_data.head(15))

# Save preprocessed IMDb and TMDb data
preprocessed_imdb_tmdb_path = os.path.join(data_dir, 'preprocessed_imdb_tmdb_data.csv')
log("Saving preprocessed IMDb and TMDb data...")
combined_data.to_csv(preprocessed_imdb_tmdb_path, index=False)
log(f"Preprocessed IMDb and TMDb data saved to '{preprocessed_imdb_tmdb_path}'")
