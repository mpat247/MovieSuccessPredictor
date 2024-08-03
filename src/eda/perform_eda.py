import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import numpy as np
import ast

# Enhanced logging function with timestamps
def log(message):
    print(f"[{datetime.datetime.now()}] [LOG]: {message}")

# Set data paths
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/cleaned'))
processed_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
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
tmdb_data = pd.read_csv(tmdb_data_path, nrows=150)  # Limiting to 150 rows for efficiency
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

# Save processed datasets with error handling
try:
    log("Saving processed IMDb basics data...")
    imdb_basics_processed_path = os.path.join(processed_data_dir, 'imdb_basics_processed.csv')
    imdb_basics.to_csv(imdb_basics_processed_path, index=False)
    log(f"IMDb basics data saved to '{imdb_basics_processed_path}'")
except OSError as e:
    log(f"Error saving IMDb basics data: {e}")

try:
    log("Saving processed IMDb ratings data...")
    imdb_ratings_processed_path = os.path.join(processed_data_dir, 'imdb_ratings_processed.csv')
    imdb_ratings.to_csv(imdb_ratings_processed_path, index=False)
    log(f"IMDb ratings data saved to '{imdb_ratings_processed_path}'")
except OSError as e:
    log(f"Error saving IMDb ratings data: {e}")

try:
    log("Saving processed TMDb data...")
    tmdb_data_processed_path = os.path.join(processed_data_dir, 'tmdb_data_processed.csv')
    tmdb_data.to_csv(tmdb_data_processed_path, index=False)
    log(f"TMDb data saved to '{tmdb_data_processed_path}'")
except OSError as e:
    log(f"Error saving TMDb data: {e}")

# Visualizations
# 1. Histogram for IMDb average ratings
log("Creating histogram for IMDb average ratings...")
plt.figure(figsize=(10, 6))
sns.histplot(imdb_ratings['averageRating'], kde=True)
plt.title('Distribution of IMDb Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
histogram_path = os.path.join(processed_data_dir, 'imdb_average_ratings_distribution.png')
plt.savefig(histogram_path)
plt.show()
plt.close()  # Close the plot
log(f"Histogram saved as '{histogram_path}'")

# 2. Scatter plot for IMDb ratings vs. number of votes
log("Creating scatter plot for IMDb ratings vs. number of votes...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=imdb_ratings['numVotes'], y=imdb_ratings['averageRating'])
plt.title('IMDb Ratings vs. Number of Votes')
plt.xlabel('Number of Votes')
plt.ylabel('Average Rating')
scatter_plot_path = os.path.join(processed_data_dir, 'imdb_ratings_vs_votes.png')
plt.savefig(scatter_plot_path)
plt.show()
plt.close()  # Close the plot
log(f"Scatter plot saved as '{scatter_plot_path}'")

# 3. Bar plot of top genres
log("Creating bar plot for top genres...")
genres = imdb_basics['genres'].str.get_dummies(sep=',')
top_genres = genres.sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_genres.values, y=top_genres.index)
plt.title('Top 10 Genres')
plt.xlabel('Number of Titles')
plt.ylabel('Genre')
genres_plot_path = os.path.join(processed_data_dir, 'top_genres.png')
plt.savefig(genres_plot_path)
plt.show()
plt.close()  # Close the plot
log(f"Bar plot saved as '{genres_plot_path}'")

# 4. Heatmap of IMDb data correlations
log("Creating correlation heatmap for IMDb data...")
imdb_numeric_data = imdb_ratings.join(imdb_basics[['runtimeMinutes']].apply(pd.to_numeric, errors='coerce'), how='inner')
imdb_corr = imdb_numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(imdb_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of IMDb Data')
imdb_heatmap_path = os.path.join(processed_data_dir, 'imdb_correlation_heatmap.png')
plt.savefig(imdb_heatmap_path)
plt.show()
plt.close()  # Close the plot
log(f"Heatmap saved as '{imdb_heatmap_path}'")

# 5. Processing 'cast' and 'crew' columns
log("Processing 'cast' and 'crew' columns...")
tmdb_data['cast'] = tmdb_data['cast'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
tmdb_data['crew'] = tmdb_data['crew'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
log("'cast' and 'crew' columns processed.")

# 6. Top 5 actors by appearance
log("Analyzing 'cast' column...")
actor_count = {}
for cast_list in tmdb_data['cast']:
    for cast in cast_list:
        name = cast.get('name', 'Unknown')
        if name in actor_count:
            actor_count[name] += 1
        else:
            actor_count[name] = 1

top_actors = pd.Series(actor_count).sort_values(ascending=False).head(5)
log(f"Top 5 actors:\n{top_actors}")

log("Plotting top actors...")
plt.figure(figsize=(10, 6))
sns.barplot(x=top_actors.values, y=top_actors.index)
plt.title('Top 5 Actors by Appearance')
plt.xlabel('Number of Appearances')
plt.ylabel('Actor')
actors_plot_path = os.path.join(processed_data_dir, 'top_actors.png')
plt.savefig(actors_plot_path)
plt.show()
plt.close()  # Close the plot
log(f"Bar plot of top actors saved as '{actors_plot_path}'")

log("EDA completed successfully.")
