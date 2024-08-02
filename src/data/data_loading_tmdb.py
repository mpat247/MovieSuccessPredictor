#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

def load_tmdb_data(tmdb_data_path):
    print("Loading TMDb dataset...")
    tmdb_data = pd.read_csv(tmdb_data_path)
    print("\nTMDb Data Head:")
    print(tmdb_data.head())

    # Checking for missing values
    print("\nMissing values in TMDb Data:")
    print(tmdb_data.isnull().sum())

    return tmdb_data

