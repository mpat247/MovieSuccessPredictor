# Data Pipeline

This document details the data processing pipeline, from raw data ingestion to feature-ready datasets.

## Overview

The data pipeline transforms raw movie data from multiple sources into a unified, feature-rich dataset suitable for machine learning. The pipeline handles three primary data sources and produces engineered features for both metadata-based and NLP-based modeling.

## Data Sources

### 1. IMDb Dataset

**Source**: Internet Movie Database
**Components**:

- `imdb_basics_cleaned.csv`: Core movie metadata
- `imdb_ratings_cleaned.csv`: User ratings and vote counts

**Schema**:

```
imdb_basics_cleaned.csv:
- tconst: Unique movie identifier (tt1234567)
- primaryTitle: Movie title
- startYear: Release year
- runtimeMinutes: Duration in minutes
- genres: Pipe-separated genre list
- titleType: Content type (movie, tvSeries, etc.)

imdb_ratings_cleaned.csv:
- tconst: Movie identifier (matches basics)
- averageRating: Average user rating (1-10)
- numVotes: Number of user votes
```

### 2. TMDb Dataset

**Source**: The Movie Database
**File**: `tmdb_data_cleaned.csv`

**Schema**:

```
tmdb_data_cleaned.csv:
- tconst: Movie identifier (matches IMDb)
- budget: Production budget (USD)
- popularity: TMDb popularity score
- cast_info: Cast member details
- crew_info: Crew member details
```

### 3. IMSDb Scripts

**Source**: Internet Movie Script Database
**Location**: `data/scripts/`
**Format**: Individual text files named by movie ID

**Structure**:

```
data/scripts/
├── tt0111161.txt  # The Shawshank Redemption
├── tt0068646.txt  # The Godfather
└── ...
```

## Data Processing Stages

### Stage 1: Data Loading

**Module**: `data_loader.py`
**Input**: Cleaned CSV files
**Output**: Merged DataFrame

**Process**:

1. Load individual CSV files using pandas
2. Validate schema and data types
3. Merge datasets on `tconst` identifier using inner joins
4. Handle missing values and data inconsistencies

**Code Example**:

```python
def load_data() -> pd.DataFrame:
    """Load and merge IMDb and TMDb datasets."""
    basics = pd.read_csv('data/cleaned/imdb_basics_cleaned.csv')
    ratings = pd.read_csv('data/cleaned/imdb_ratings_cleaned.csv')
    tmdb = pd.read_csv('data/cleaned/tmdb_data_cleaned.csv')

    df = basics.merge(ratings, on='tconst', how='inner')
             .merge(tmdb, on='tconst', how='inner')
    return df
```

### Stage 2: Feature Engineering

**Module**: `feature_engineer.py`
**Input**: Merged raw DataFrame
**Output**: Feature matrix with target variable

**Feature Types**:

#### Numeric Features

- `budget`: Production budget (normalized)
- `popularity`: TMDb popularity score
- `runtimeMinutes`: Movie duration
- `numVotes`: Number of user ratings

#### Categorical Features

- `genres`: One-hot encoded using pandas.get_dummies()
- Creates binary columns for each genre (Action, Comedy, Drama, etc.)

#### Target Variable

- `success`: Binary classification target
- Based on median rating threshold
- 1 = Above median rating, 0 = Below median rating

**Implementation**:

```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate modeling features from raw data."""
    # Numeric features
    numeric_cols = ['budget', 'popularity', 'runtimeMinutes', 'numVotes']
    features = df[['tconst'] + numeric_cols].copy()

    # Genre encoding
    if 'genres' in df:
        genres_dummies = df['genres'].str.get_dummies(sep='|')
        features = pd.concat([features, genres_dummies], axis=1)

    # Target variable
    median = df['averageRating'].median()
    features['success'] = (df['averageRating'] >= median).astype(int)

    return features
```

### Stage 3: Script Processing

**Module**: `script_features.py`
**Input**: Movie script text files
**Output**: NLP feature matrix

**NLP Features**:

#### BERT Tokenization

- Tokenizer: `bert-base-uncased`
- Max sequence length: 512 tokens
- Padding: Post-padding with zeros
- Truncation: Truncate longer sequences

#### Sentiment Analysis

- Library: TextBlob
- Output: Polarity score (-1 to 1)
- -1 = Negative, 0 = Neutral, 1 = Positive

#### Readability Metrics

- Library: textstat
- Metric: Flesch Reading Ease score
- Range: 0-100 (higher = more readable)

**Processing Pipeline**:

```python
def process_scripts() -> pd.DataFrame:
    """Extract NLP features from movie scripts."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    features_list = []

    for script_file in script_files:
        tconst = script_file.stem
        text = script_file.read_text(encoding='utf-8')

        # Tokenization
        tokens = tokenizer.encode(text, add_special_tokens=True)
        padded = pad_sequences([tokens], maxlen=512,
                              padding='post', truncating='post')[0]

        # Sentiment and readability
        sentiment = TextBlob(text).sentiment.polarity
        readability = textstat.flesch_reading_ease(text)

        features_list.append({
            'tconst': tconst,
            'sentiment': sentiment,
            'readabilityScore': readability,
            **{f'token_{i}': token for i, token in enumerate(padded)}
        })

    return pd.DataFrame(features_list)
```

## Data Quality Measures

### Missing Value Handling

**Strategy**: Context-dependent imputation

- **Numeric features**: Median imputation for outlier robustness
- **Categorical features**: Mode imputation or separate "Unknown" category
- **Scripts**: Optional processing (NLP features set to 0 if missing)

### Data Validation

**Checks Performed**:

1. Schema validation (required columns present)
2. Data type consistency
3. Value range validation (ratings 1-10, positive budgets)
4. Duplicate detection on `tconst`

### Outlier Detection

**Methods**:

- IQR-based outlier detection for numeric features
- Z-score analysis for extreme values
- Domain knowledge validation (reasonable budget ranges)

## Pipeline Output

### Final Feature Matrix

**Location**: `data/processed/features.csv`
**Schema**:

```
- tconst: Movie identifier
- budget: Normalized production budget
- popularity: TMDb popularity score
- runtimeMinutes: Movie duration
- numVotes: Number of ratings
- Action, Comedy, Drama, ...: Genre binary features
- success: Target variable (binary)
```

### NLP Feature Matrix

**Location**: `data/processed/script_features.csv`
**Schema**:

```
- tconst: Movie identifier
- sentiment: TextBlob polarity score
- readabilityScore: Flesch reading ease
- token_0, token_1, ..., token_511: BERT token embeddings
```

## Performance Considerations

### Memory Optimization

- Streaming processing for large script files
- Chunk-based processing for memory-limited environments
- Efficient data types (int8 for binary features)

### Processing Speed

- Vectorized operations using pandas
- Batch processing for BERT tokenization
- Parallel processing for independent scripts

### Storage Efficiency

- Compressed CSV storage
- Sparse matrix representation for one-hot encoded features
- Parquet format for large datasets

## Monitoring and Logging

### Data Quality Metrics

- Missing value percentages
- Feature distribution statistics
- Correlation analysis between features

### Processing Metrics

- Processing time per stage
- Memory usage tracking
- Error rates and failure modes

### Validation Reports

- Schema compliance checks
- Data drift detection
- Feature importance analysis

## Troubleshooting

### Common Issues

**Missing Data Files**:

- Verify file paths in configuration
- Check file permissions and accessibility
- Validate CSV structure and encoding

**Memory Errors**:

- Reduce batch size for script processing
- Use data sampling for development
- Consider incremental processing

**Encoding Issues**:

- Ensure UTF-8 encoding for script files
- Handle special characters in movie titles
- Validate text preprocessing steps

**Feature Engineering Errors**:

- Check for empty genre strings
- Validate numeric feature ranges
- Handle edge cases in target variable creation
