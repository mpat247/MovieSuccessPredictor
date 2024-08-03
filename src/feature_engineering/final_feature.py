import pandas as pd
import os

# Logging function
def log(message):
    print(f"[{pd.Timestamp.now()}] [LOG]: {message}")

# Define data paths
base_dir = 'D:\\manav\\Documents\\Engineering\\Masters\\EE8206\\Project\\MovieSuccessPredictor'
processed_dir = os.path.join(base_dir, 'data', 'processed')
final_features_path = os.path.join(processed_dir, 'final_features.csv')
final_script_features_path = os.path.join(processed_dir, 'final_script_features.csv')

# Ensure the processed data directory exists
os.makedirs(processed_dir, exist_ok=True)

# Load datasets
log("Loading final features dataset...")
final_features = pd.read_csv(final_features_path)
log(f"Final features dataset loaded with shape: {final_features.shape}")

log("Loading final script features dataset...")
final_script_features = pd.read_csv(final_script_features_path)
log(f"Final script features dataset loaded with shape: {final_script_features.shape}")

# Check if the number of rows match (optional, based on your integration approach)
if len(final_features) != len(final_script_features):
    log("The number of rows in final features and script features datasets do not match.")
    log("Proceeding with separate treatment of the datasets.")
    # Do not raise an error, but log the message

# Save the datasets as they are for separate modeling or further processing
log("Saving IMDb and TMDb features data as 'final_features_processed.csv'...")
final_features_processed_path = os.path.join(processed_dir, 'final_features_processed.csv')
final_features.to_csv(final_features_processed_path, index=False)
log(f"IMDb and TMDb features data saved to '{final_features_processed_path}'")

log("Saving script features data as 'final_script_features_processed.csv'...")
final_script_features_processed_path = os.path.join(processed_dir, 'final_script_features_processed.csv')
final_script_features.to_csv(final_script_features_processed_path, index=False)
log(f"Script features data saved to '{final_script_features_processed_path}'")

log("Final feature processing completed.")
