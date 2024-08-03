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
data_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')
final_features_path = os.path.join('D:\\', 'manav', 'Documents', 'Engineering', 'Masters', 'EE8206', 'Project', 'MovieSuccessPredictor', 'data', 'processed', 'final_features.csv')

# Load final features dataset
log("Loading final features dataset...")
final_features = pd.read_csv(final_features_path)
log(f"Final features dataset loaded with shape: {final_features.shape}")
print(final_features.head(15))

# Define columns for specific preprocessing
numerical_columns = ['averageRating', 'numVotes', 'runtimeMinutes_normalized']
categorical_columns = ['titleType', 'releaseSeason']  # Genres are also one-hot encoded in the dataset
text_column = 'script_text'

# Tokenizer setup for BERT
log("Setting up BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocessing script text (tokenization and padding)
if text_column in final_features.columns:
    log("Tokenizing script text data...")
    max_length = 512  # BERT maximum input length

    def tokenize_script(text):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return tokens

    final_features['tokenized_script'] = final_features[text_column].apply(tokenize_script)
    log("Tokenization completed.")
    print(final_features[['tokenized_script']].head(15))

    log("Padding tokenized sequences...")
    final_features['tokenized_script_padded'] = list(pad_sequences(final_features['tokenized_script'], maxlen=max_length, padding='post', truncating='post'))
    log("Padding completed.")
    print(final_features[['tokenized_script_padded']].head(15))
else:
    log(f"Column '{text_column}' not found in dataset. Skipping script text processing.")

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
