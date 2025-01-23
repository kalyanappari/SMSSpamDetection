import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv(r"C:\Users\asaik\OneDrive\Documents\ML Projects\SMSSpamDetection\spam_sms.csv")  # Adjust encoding if needed

# Display dataset information
print("Dataset Info:")
print(df.info())

# Preview the first few rows of the dataset
print("\nDataset Preview:")
print(df.head())

# Rename columns if needed
df.columns = ['v1', 'v2']  # Example: Assuming two columns (label and text)

# Check for missing values
print("\nChecking for Missing Values:")
print(df.isnull().sum())

# Basic statistics
print("\nClass Distribution:")
print(df['v1'].value_counts())

# Example preprocessing step: Encoding labels (Spam: 1, Ham: 0)
df['v1'] = df['v1'].map({'spam': 1, 'ham': 0})

# Save the preprocessed dataset (optional)
df.to_csv('preprocessed_sms_data.csv', index=False)

nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join the tokens back into a single string
    return ' '.join(tokens)

# Apply preprocessing to the 'Message' column
df['Cleaned_Message'] = df['v2'].apply(preprocess_text)

print("\nPreprocessed Messages:")
print(df[['v2', 'Cleaned_Message']].head())

print("\nPreprocessing Complete!")

# Initialize the vectorizer (TF-IDF as an example)
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 most important features

# Transform the cleaned messages into numerical features
X = vectorizer.fit_transform(df['Cleaned_Message']).toarray()

# Labels
y = df['v1']

print("\nFeature Matrix Shape:", X.shape)
print("\nSample Features (TF-IDF):")
print(vectorizer.get_feature_names_out()[:10])  # Display first 10 feature names

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)

# Initialize the Naive Bayes classifier
classifier = MultinomialNB()

# Train the model
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
