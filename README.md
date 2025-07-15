# Movie Success Predictor

## Overview

A comprehensive machine learning pipeline that predicts movie success using a dual-model approach combining metadata analysis with script sentiment processing. This project demonstrates advanced data science techniques by integrating **Random Forest** regression for metadata-based predictions and **Neural Networks** with BERT embeddings for script sentiment analysis, leveraging comprehensive datasets from IMDb, TMDb, and IMSDb.

**🎯 Key Achievement:** Successfully implemented an end-to-end machine learning pipeline that processes 5,000+ movies from multiple data sources, achieving robust prediction capabilities through feature engineering, hyperparameter optimization, and multi-modal data integration.

## 🚀 Key Features

- **🤖 Dual-Model Architecture:** Combines Random Forest (metadata regression) + Neural Network (script sentiment)
- **🔄 End-to-End Pipeline:** Automated data loading, cleaning, feature engineering, and model training
- **📊 Multi-Source Integration:** IMDb ratings/metadata, TMDb cast/crew data, IMSDb movie scripts
- **🧠 Advanced NLP:** BERT tokenization, TextBlob sentiment analysis, Flesch-Kincaid readability scoring
- **📈 Comprehensive Evaluation:** RMSE, R² score, MSE, feature importance analysis with GridSearchCV
- **⚙️ Production Ready:** Modular architecture with comprehensive logging and error handling

## 🏗️ Project Architecture

```text
src/
├── 📁 feature_engineering/     # Core feature processing modules
│   ├── feature_engineering.py  # Main feature extraction pipeline
│   ├── final_feature.py       # Feature integration and finalization
│   ├── preprocess_data.py     # Data preprocessing utilities
│   └── preprocess_script.py   # Script-specific preprocessing
├── 📁 eda/                    # Exploratory Data Analysis
│   └── perform_eda.py         # Statistical analysis and visualization
├── 📁 web_scraping/           # Data collection modules
│   ├── web_scraping_imsdb.py  # IMSDb script scraping
│   └── scripts.py             # Additional scraping utilities
├── 📁 notebooks/              # Development notebooks
├── model_development.py       # Model training and evaluation
├── data_loading_and_cleaning.ipynb # Data pipeline notebook
└── config.json               # Configuration settings
```

## 💻 Technologies Used

- **Programming:** Python 3.8+, Jupyter Lab
- **ML/DL:** scikit-learn, TensorFlow/Keras, Hugging Face Transformers
- **Data Processing:** pandas, numpy, textblob, textstat
- **Visualization:** matplotlib, seaborn
- **Development:** VS Code, Git, Docker-ready

## ⚡ Quick Start

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (16GB recommended for BERT processing)
- GPU support optional but recommended for BERT-LSTM training

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mpat247/MovieSuccessPredictor.git
   cd MovieSuccessPredictor
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### 🔄 Running the Pipeline

**Execute individual components for development:**

```bash
# Data processing and feature engineering
python src/feature_engineering/feature_engineering.py  # Extract features from all data sources
python src/feature_engineering/final_feature.py        # Combine and finalize features
python src/model_development.py                        # Data preprocessing and model training

# Exploratory data analysis
python src/eda/perform_eda.py                         # Generate data insights and visualizations

# Web scraping (data collection)
python src/web_scraping/web_scraping_imsdb.py         # Scrape movie scripts from IMSDb
```

**Or run the complete pipeline using Jupyter notebooks:**

```bash
# Open and execute the main development notebook
jupyter lab src/data_loading_and_cleaning.ipynb
```

## 📊 Data Pipeline

### 1. Data Sources

| Source    | Content                                  | Records | Features           |
| --------- | ---------------------------------------- | ------- | ------------------ |
| **IMDb**  | Movie metadata, ratings, genres, runtime | 10,000+ | Basic movie info   |
| **TMDb**  | Cast, crew, budget, popularity metrics   | 5,000+  | Production details |
| **IMSDb** | Movie scripts for sentiment analysis     | 1,000+  | Script text        |

### 2. Feature Engineering Pipeline

**Numeric Features:**

- Budget, popularity, runtime, vote counts
- Normalized using StandardScaler for consistent scaling
- Runtime minutes specifically normalized to handle varying movie lengths

**Categorical Features:**

- Genres (one-hot encoded via pandas.get_dummies with comma separation)
- Title types (movie, short, tvEpisode, tvMiniSeries, etc.)
- Release seasons (Winter, Spring, Summer, Fall) extracted from start year

**NLP Features:**

- BERT tokenization (512 max sequence length)
- Script sentiment polarity and subjectivity (TextBlob)
- Readability scores (Flesch-Kincaid grade level using textstat)
- Word count and textual complexity metrics
- BERT embeddings (768-dimensional) for semantic representation

**Feature Processing Details:**

- **IMDb Data:** Merged basics and ratings datasets on 'tconst' identifier
- **TMDb Data:** Cast and crew name extraction using eval() for JSON-like string parsing
- **Script Data:** Comprehensive NLP processing including sentiment analysis and readability scoring
- **Final Integration:** 49 engineered features for Random Forest, separate script features for Neural Network

**Target Variable:**

- Continuous rating prediction (averageRating from IMDb)
- Regression task for Random Forest model
- Sentiment score prediction for Neural Network model
