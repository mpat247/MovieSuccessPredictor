#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def clean_tmdb_data(tmdb_data):
    # Fill missing numerical values with the mean
    tmdb_data_filled = tmdb_data.fillna(tmdb_data.mean())
    # Standardize string fields if necessary, example with 'title' field
    if 'title' in tmdb_data.columns:
        tmdb_data_filled['title'] = tmdb_data_filled['title'].str.lower()
    
    print("\nSample data after cleaning:")
    print(tmdb_data_filled.head())

    return tmdb_data_filled

