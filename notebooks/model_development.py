import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences

# Logging function
def log(message):
    print(f"[{pd.Timestamp.now()}] [LOG]: {message}")

# Define data paths
base_dir = 'D:\\manav\\Documents\\Engineering\\Masters\\EE8206\\Project\\MovieSuccessPredictor'
data_dir = os.path.join(base_dir, 'data', 'processed')
final_features_path = os.path.join(data_dir, 'final_features.csv')

# Ensure the processed data directory exists
os.makedirs(data_dir, exist_ok=True)

# Load final features dataset
log("Loading final features dataset...")
try:
    final_features = pd.read_csv(final_features_path, low_memory=False)
    log(f"Final features dataset loaded with shape: {final_features.shape}")
    print(final_features.head(15))
except FileNotFoundError:
    log(f"File not found: {final_features_path}")
    raise

# Define columns for specific preprocessing
numerical_columns = ['averageRating', 'numVotes', 'runtimeMinutes_normalized']
categorical_columns = [
    'titleType_movie', 'titleType_short', 'titleType_tvEpisode', 'titleType_tvMiniSeries', 'titleType_tvMovie',
    'titleType_tvSeries', 'titleType_tvShort', 'titleType_tvSpecial', 'titleType_video', 'titleType_videoGame',
    'season_Winter'
]

# Tokenizer setup for BERT
log("Setting up BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Note: Skipping script text processing because the 'script_text' column is not present

# Normalizing numerical features
log("Normalizing numerical features...")
scaler = StandardScaler()
final_features[numerical_columns] = scaler.fit_transform(final_features[numerical_columns])
log("Normalization completed.")
print(final_features[numerical_columns].head(15))

# One-hot encoding categorical features
log("One-hot encoding categorical features...")
final_features = pd.get_dummies(final_features, columns=categorical_columns, drop_first=True)
log("One-hot encoding completed.")
print(final_features.head(15))

# Save the preprocessed data for model training
preprocessed_data_path = os.path.join(data_dir, 'preprocessed_data.csv')
log("Saving preprocessed data...")
final_features.to_csv(preprocessed_data_path, index=False)
log(f"Preprocessed data saved to '{preprocessed_data_path}'")

log("Data loading and preprocessing completed.")
