#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def clean_imdb_data(imdb_basics, imdb_crew, imdb_principals, imdb_ratings):
    # Dropping rows with critical missing values in IMDb basics
    imdb_basics_cleaned = imdb_basics.dropna(subset=['primaryTitle', 'startYear', 'genres'])
    print(f"Dropped rows with missing critical values. Remaining rows: {imdb_basics_cleaned.shape[0]}")

    # Filling missing values in IMDb ratings
    imdb_ratings['averageRating'].fillna(imdb_ratings['averageRating'].mean(), inplace=True)
    print("IMDb Ratings - Missing values after filling:")
    print(imdb_ratings.isnull().sum())

    # Standardize text fields
    imdb_basics_cleaned['primaryTitle'] = imdb_basics_cleaned['primaryTitle'].str.lower()
    imdb_basics_cleaned['genres'] = imdb_basics_cleaned['genres'].str.lower()

    print("\nSample data after cleaning and standardization:")
    print(imdb_basics_cleaned.head())

    return imdb_basics_cleaned, imdb_ratings

