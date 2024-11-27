
# Movie Success Prediction Using BERT-LSTM and Random Forest

## Description
This project predicts movie success using a dual-model approach combining metadata from IMDb and TMDb datasets with script sentiment analysis from IMSDb scripts. The **Random Forest Model** predicts overall ratings, while the **BERT-LSTM Neural Network** analyzes script sentiment. The project leverages machine learning and natural language processing techniques to provide insights into key factors influencing a movie's success.

## Features
- **Random Forest Model:** Uses metadata (e.g., cast, crew, ratings) for prediction.
- **BERT-LSTM Model:** Analyzes scripts for sentiment classification and scoring.
- **Feature Engineering:** Includes normalization, one-hot encoding, and custom scoring.
- **Data Sources:** IMDb, TMDb, and IMSDb datasets.
- **Evaluation Metrics:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.

## Technologies Used
- **Programming Languages:** Python
- **Libraries/Frameworks:** Keras, Scikit-learn, Hugging Face Transformers
- **Tools:** Jupyter Lab, matplotlib, pandas, numpy
- **Hardware Requirements:** Intel i7 CPU, 32GB RAM, 1TB HDD

## Process Overview
### Data Collection
- **IMDb Dataset:** Movie ratings, genres, release years, and more.
- **TMDb Dataset:** Information on cast and crew.
- **IMSDb Dataset:** 1,000 movie scripts for sentiment analysis.

### Data Preprocessing
- Normalized numerical features using Min-Max Scaling.
- Applied one-hot encoding for categorical variables.
- Tokenized scripts using BERT tokenizer for embedding generation.

### Model Development
1. **Random Forest Model:**
   - Predicts overall ratings using metadata.
   - Hyperparameters optimized with GridSearchCV.
2. **BERT-LSTM Model:**
   - Extracts sentiment from scripts using BERT embeddings.
   - Sequential data modeled with LSTM layers.
   - Additional dense layers with ReLU activation.

### Evaluation Metrics
- **Random Forest Model:**
  - RMSE: 0.945
  - R² Score: 0.201
- **BERT-LSTM Model:**
  - MSE: 0.0022

## Results and Insights
- Metadata such as cast popularity, ratings, and release year significantly influence success.
- Scripts with clear positive sentiments tend to perform better with audiences.
- Combining metadata and script sentiment enhances prediction accuracy.

## Challenges and Limitations
- **Script Complexity:** Highly complex scripts posed challenges for BERT-LSTM.
- **Feature Limitation:** Incorporating more metadata could improve Random Forest predictions.
- **Social Media Sentiment:** Including social media trends could enhance the model.

## Future Work
- Expand datasets to include more features like production budgets and marketing data.
- Integrate social media sentiment analysis.
- Improve model architectures for better handling of complex script patterns.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/your-username/your-repo-name.git
   ```
2. Install the required Python libraries:
   ```
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook for training and testing:
   ```
   jupyter lab
   ```
