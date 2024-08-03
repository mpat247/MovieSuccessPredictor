import pandas as pd
import os
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob
import textstat

# Logging function
def log(message):
    print(f"[{pd.Timestamp.now()}] [LOG]: {message}")

# Define base directory and data paths
base_dir = 'D:\\manav\\Documents\\Engineering\\Masters\\EE8206\\Project\\MovieSuccessPredictor'
scripts_dir = os.path.join(base_dir, 'data', 'scripts')
data_dir = os.path.join(base_dir, 'data', 'processed')

# Ensure the processed data directory exists
os.makedirs(data_dir, exist_ok=True)

# Tokenizer setup for BERT
log("Setting up BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Initialize lists to store extracted features
script_features = []

# Function to process each script
def process_script(script_text):
    log(f"Processing script...")

    # Tokenization and padding
    log("Tokenizing script text data...")
    tokens = tokenizer.encode(script_text, add_special_tokens=True)
    log(f"Tokens: {tokens[:10]}...")  # Print first 10 tokens for brevity

    log("Padding tokenized sequences...")
    max_length = 512
    tokenized_script_padded = pad_sequences([tokens], maxlen=max_length, padding='post', truncating='post')[0]
    log(f"Padded tokens: {tokenized_script_padded[:10]}...")  # Print first 10 padded tokens for brevity

    # Sentiment analysis
    log("Performing sentiment analysis...")
    sentiment = TextBlob(script_text).sentiment.polarity
    log(f"Sentiment: {sentiment}")

    # Readability score
    log("Calculating readability score...")
    readability_score = textstat.flesch_reading_ease(script_text)
    log(f"Readability score: {readability_score}")

    return {
        'tokenized_script_padded': tokenized_script_padded,
        'sentiment': sentiment,
        'readabilityScore': readability_score
    }

# Process each script file in the directory
log("Loading scripts from directory...")
for i, script_file in enumerate(os.listdir(scripts_dir)):
    script_path = os.path.join(scripts_dir, script_file)
    with open(script_path, 'r', encoding='utf-8') as file:
        script_text = file.read()

    log(f"Loaded script: {script_file}")

    # Process the script and extract features
    features = process_script(script_text)
    script_features.append(features)

    # Print detailed logs for the first 15 scripts
    if i < 15:
        log(f"Detailed processing for script {i + 1}:")
        print(f"File: {script_file}")
        print(f"Tokens (first 10): {features['tokenized_script_padded'][:10]}")
        print(f"Sentiment: {features['sentiment']}")
        print(f"Readability Score: {features['readabilityScore']}")
        print('-' * 40)

# Convert the list of features to a DataFrame
log("Converting extracted features to DataFrame...")
script_features_df = pd.DataFrame(script_features)

# Save preprocessed script data
final_script_features_path = os.path.join(data_dir, 'final_script_features.csv')
log("Saving final script features data...")
script_features_df.to_csv(final_script_features_path, index=False)
log(f"Final script features data saved to '{final_script_features_path}'")

log("Script data processing completed.")
