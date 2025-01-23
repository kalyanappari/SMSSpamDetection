# SMSSpamDetection
SMS Spam Detection Using Natural Language Processing (NLP) This project focuses on building an efficient SMS spam detection system using Natural Language Processing (NLP) techniques. The goal is to classify SMS messages into two categories: Spam or Ham (Not Spam), enabling effective filtering of unwanted messages.
# KeyFeatures
• Dataset: The project utilizes a publicly available dataset from Kaggle containing labeled SMS messages as "spam" or "ham."
DataSet Link : https://www.kaggle.com/datasets/vishakhdapat/sms-spam-detection-dataset?resource=download
• Text Preprocessing: Messages are cleaned by: • Removing special characters, numbers, and punctuation. • Converting text to lowercase.
• Removing stopwords. • Applying lemmatization to standardize words. • Feature Extraction: Implemented TF-IDF vectorization to convert textual data into numerical features.
• Modeling: Built and trained a machine learning model using the Multinomial Naive Bayes algorithm for classification.
• Evaluation: The model's performance is assessed using metrics like accuracy, precision, recall, F1-score, and a confusion matrix.
