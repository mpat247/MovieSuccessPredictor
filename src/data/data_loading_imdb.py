#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

def load_imdb_data(basics_path, crew_path, principals_path, ratings_path):
    def read_tsv(file_path):
        return pd.read_csv(file_path, sep='\t', na_values='\\N', low_memory=False)

    print("Loading IMDb datasets...")
    imdb_basics = read_tsv(basics_path)
    imdb_crew = read_tsv(crew_path)
    imdb_principals = read_tsv(principals_path)
    imdb_ratings = read_tsv(ratings_path)

    print("\nIMDb Basics Head:")
    print(imdb_basics.head())
    print("\nIMDb Crew Head:")
    print(imdb_crew.head())
    print("\nIMDb Principals Head:")
    print(imdb_principals.head())
    print("\nIMDb Ratings Head:")
    print(imdb_ratings.head())

    # Checking for missing values
    print("\nMissing values in IMDb Basics:")
    print(imdb_basics.isnull().sum())
    print("\nMissing values in IMDb Crew:")
    print(imdb_crew.isnull().sum())
    print("\nMissing values in IMDb Principals:")
    print(imdb_principals.isnull().sum())
    print("\nMissing values in IMDb Ratings:")
    print(imdb_ratings.isnull().sum())

    return imdb_basics, imdb_crew, imdb_principals, imdb_ratings

