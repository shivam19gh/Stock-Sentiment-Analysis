               STOCK SENTIMENT ANALYSIS 

This project focuses on analyzing stock market sentiment based on textual data such as news headlines. By combining Natural Language Processing (NLP) techniques and Machine Learning models, it predicts the sentiment of stock-related news and evaluates its potential impact on stock prices.


PROJECT WORKFLOW 
1) Data Preprocessing: Load and clean stock-related news headlines. Remove stop words, punctuation, and perform tokenization.

2) Feature Extraction: Use TF-IDF vectorization to convert text into numerical features.

3) Sentiment Classification: Train a Random Forest Classifier with sentiment-labeled data. Apply 10-fold cross-validation to evaluate model performance.

4) Model Validation: Calculate accuracy scores using cross-validation. Visualize the performance metrics.

5) Prediction: Predict the sentiment of new, unseen stock-related news.



TECH STACK

1) Programming Language: Python

2) Libraries:
a) Data Processing: Pandas, NumPy
b) NLP: NLTK
c) Vectorization: Scikit-learn (TF-IDF)
d) Machine Learning: RandomForestClassifier
e) Visualization: Matplotlib, Seaborn


