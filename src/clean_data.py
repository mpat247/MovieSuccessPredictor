"""
clean_data.py
Module to clean raw IMDB and TMDB datasets.
"""
import pandas as pd
from pathlib import Path
from config import config


def clean_imdb():
    """Clean IMDB basics and ratings datasets."""
    base = Path(__file__).parent
    raw_dir = base / config['raw_data_path']
    clean_dir = base / config['clean_data_path']
    clean_dir.mkdir(parents=True, exist_ok=True)

    basics = pd.read_csv(raw_dir / 'imdb_basics.csv')
    ratings = pd.read_csv(raw_dir / 'imdb_ratings.csv')

    # Drop rows with missing critical values
    basics_cleaned = basics.dropna(subset=['primaryTitle', 'startYear', 'genres'])
    basics_cleaned['primaryTitle'] = basics_cleaned['primaryTitle'].str.lower()
    basics_cleaned['genres'] = basics_cleaned['genres'].str.lower()

    # Fill missing ratings
    ratings['averageRating'].fillna(ratings['averageRating'].mean(), inplace=True)

    # Save cleaned data
    basics_cleaned.to_csv(clean_dir / 'imdb_basics_cleaned.csv', index=False)
    ratings.to_csv(clean_dir / 'imdb_ratings_cleaned.csv', index=False)

    return basics_cleaned, ratings


def clean_tmdb():
    """Clean TMDB dataset."""
    base = Path(__file__).parent
    raw_dir = base / config['raw_data_path']
    clean_dir = base / config['clean_data_path']

    tmdb = pd.read_csv(raw_dir / 'tmdb_data.csv')
    tmdb_cleaned = tmdb.dropna()
    tmdb_cleaned.to_csv(clean_dir / 'tmdb_data_cleaned.csv', index=False)

    return tmdb_cleaned


def clean_all():
    """Run all data cleaning steps."""
    print("Cleaning IMDB data...")
    clean_imdb()
    print("Cleaning TMDB data...")
    clean_tmdb()


if __name__ == '__main__':
    clean_all()
