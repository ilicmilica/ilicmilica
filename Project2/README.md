# Project 2

## Hospital Review Sentiment Analysis with BI-LSTM (NLP)
### Overview
The goal of this project is to build a machine learning model that classifies hospital reviews as either positive or negative based on their content. This sentiment analysis project involves preprocessing textual data, training a Bi-LSTM (Bidirectional Long Short-Term Memory) deep learning model, and evaluating its performance. The resulting model provides an accurate classification of patient feedback, which can help hospitals improve their services by understanding patient experiences.
### Dataset Description
The dataset consists of hospital reviews labeled with corresponding sentiment indicators:

Positive (1) — Indicates a favorable experience
Negative (0) — Indicates a poor experience
Each review undergoes extensive preprocessing to remove noise and prepare it for training. The steps include tokenization, punctuation removal, stopword elimination, lowercasing, and stemming. The cleaned dataset is then used to train a deep learning model.
### Key Insights:
#### 1. Data Cleaning
Missing values and duplicate rows were identified and removed to ensure data quality.
#### 2. Text Preprocessing
Feedback text was tokenized, lowercased, stripped of punctuation, and stemmed to prepare it for modeling.
#### 3. Sentiment Prediction
A BI-LSTM model was trained to classify feedback as either positive (1) or negative (0).

#### Model Results:
- Accuracy: Achieved an accuracy of 0.85 on the test set.
- AUC Score: The model attained an AUC of 0.78, indicating reliable performance in distinguishing between positive and negative feedback.

#### Visual Results
The model’s performance is visualized below, showing the alignment between true and predicted values:
![aoc_curve](https://github.com/user-attachments/assets/46d643c0-8a54-4a59-bd2e-2871b9b1f561)

![conf_mat](https://github.com/user-attachments/assets/a8de1567-8393-458a-a44f-b7a8514f32ed)

### Conclusion
This project demonstrates the complete pipeline for hospital feedback sentiment analysis, starting from raw text preprocessing through to deep learning classification with BI-LSTM. With an achieved accuracy of 85 % and an AUC score of 0.78, the project successfully highlights the potential of deep learning for sentiment analysis in the healthcare domain.

