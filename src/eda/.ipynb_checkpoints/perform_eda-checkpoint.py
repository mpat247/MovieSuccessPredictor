import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from ast import literal_eval

# Enhanced logging function with timestamps
def log(message):
    print(f"[{datetime.datetime.now()}] [LOG]: {message}")

# Set data paths
data_dir = os.path.join(os.path.dirname(__file__), '../../data/cleaned')
processed_data_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')
imdb_basics_path = os.path.join(data_dir, 'imdb_basics_cleaned.csv')
imdb_ratings_path = os.path.join(data_dir, 'imdb_ratings_cleaned.csv')
tmdb_data_path = os.path.join(data_dir, 'tmdb_data_cleaned.csv')

# Ensure processed data directory exists
os.makedirs(processed_data_dir, exist_ok=True)

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

# Basic statistics
log("Calculating basic statistics for IMDb basics data...")
imdb_basics_stats = imdb_basics.describe(include='all')
print(imdb_basics_stats)
log("Basic statistics for IMDb basics data calculated.")

log("Calculating basic statistics for IMDb ratings data...")
imdb_ratings_stats = imdb_ratings.describe(include='all')
print(imdb_ratings_stats)
log("Basic statistics for IMDb ratings data calculated.")

log("Calculating basic statistics for TMDb data...")
tmdb_data_stats = tmdb_data.describe(include='all')
print(tmdb_data_stats)
log("Basic statistics for TMDb data calculated.")

# Save processed datasets
log("Saving processed IMDb basics data...")
imdb_basics_processed_path = os.path.join(processed_data_dir, 'imdb_basics_processed.csv')
imdb_basics.to_csv(imdb_basics_processed_path, index=False)
log(f"IMDb basics data saved to '{imdb_basics_processed_path}'")

log("Saving processed IMDb ratings data...")
imdb_ratings_processed_path = os.path.join(processed_data_dir, 'imdb_ratings_processed.csv')
imdb_ratings.to_csv(imdb_ratings_processed_path, index=False)
log(f"IMDb ratings data saved to '{imdb_ratings_processed_path}'")

log("Saving processed TMDb data...")
tmdb_data_processed_path = os.path.join(processed_data_dir, 'tmdb_data_processed.csv')
tmdb_data.to_csv(tmdb_data_processed_path, index=False)
log(f"TMDb data saved to '{tmdb_data_processed_path}'")

# Visualizations
# Histogram for IMDb average ratings
log("Creating histogram for IMDb average ratings...")
plt.figure(figsize=(10, 6))
sns.histplot(imdb_ratings['averageRating'], kde=True)
plt.title('Distribution of IMDb Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
histogram_path = os.path.join(processed_data_dir, 'imdb_average_ratings_distribution.png')
plt.savefig(histogram_path)
plt.show()
log(f"Histogram saved as '{histogram_path}'")

# Process text columns
def process_text_column(column):
    return column.apply(lambda x: literal_eval(x) if isinstance(x, str) else [])

log("Processing 'cast' and 'crew' columns...")
tmdb_data['cast'] = process_text_column(tmdb_data['cast'])
tmdb_data['crew'] = process_text_column(tmdb_data['crew'])
log("'cast' and 'crew' columns processed.")

# Analyzing 'cast' column
log("Analyzing 'cast' column...")
cast_list = tmdb_data['cast'].apply(pd.Series).stack().reset_index(drop=True)
cast_counts = cast_list.value_counts()
log(f"Top 5 actors:\n{cast_counts.head()}")

# Plotting top actors
log("Plotting top actors...")
top_actors = cast_counts.head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_actors.values, y=top_actors.index)
plt.title('Top 10 Actors in TMDb Dataset')
plt.xlabel('Number of Movies')
plt.ylabel('Actor')
actors_plot_path = os.path.join(processed_data_dir, 'top_actors.png')
plt.savefig(actors_plot_path)
plt.show()
log(f"Top actors plot saved as '{actors_plot_path}'")

# Analyzing 'crew' column (example for directors)
log("Analyzing 'crew' column...")
crew_list = tmdb_data['crew'].apply(pd.Series).stack().reset_index(drop=True)
crew_counts = crew_list.value_counts()
log(f"Top 5 crew members:\n{crew_counts.head()}")

# Plotting top crew members (example)
log("Plotting top crew members...")
top_crew = crew_counts.head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_crew.values, y=top_crew.index)
plt.title('Top 10 Crew Members in TMDb Dataset')
plt.xlabel('Number of Movies')
plt.ylabel('Crew Member')
crew_plot_path = os.path.join(processed_data_dir, 'top_crew.png')
plt.savefig(crew_plot_path)
plt.show()
log(f"Top crew members plot saved as '{crew_plot_path}'")

log("EDA completed.")
