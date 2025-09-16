# Fake News Detection

A machine learning project to classify news articles and text snippets as "Real" or "Fake" in real-time. This tool is designed to assist content moderators and media analysts in quickly identifying potentially false information.

***

## 📝 Project Overview

This project addresses the need for rapid identification of misinformation. We built a text classification model that predicts whether a given piece of content is fake or real, deploying it through a simple web interface for immediate use by analysts.

The workflow is divided into four main stages:

1.  **Data Preparation**: We sourced a public dataset (e.g., ISOT Fake News Dataset), then cleaned and preprocessed the text by removing noise, handling missing values, and normalizing the content through tokenization and stopword removal.

2.  **Exploratory Data Analysis (EDA) & Feature Engineering**: We analyzed patterns in the text to distinguish between fake and real news, looking at features like article length, punctuation, and sentiment. We then converted the text into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization. 

3.  **Model Training & Validation**: Several classification models, such as **Logistic Regression** and **Naive Bayes**, were trained on the processed data. The models were rigorously evaluated using key metrics:
    * Accuracy
    * Precision
    * Recall
    * F1-Score
    A confusion matrix was also used to analyze the model's performance on false positives versus false negatives.

4.  **Deployment via Web Interface**: The best-performing model was saved and wrapped in a lightweight web application using **Streamlit**. This interface allows analysts to paste a news snippet and instantly receive a "Fake" or "Real" classification.

***
