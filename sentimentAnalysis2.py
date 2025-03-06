import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from sklearn.utils import resample
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

# Configure the logging system
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Read the DataFrame from the CSV file
df = pd.read_csv(r'C:\ML Projects\SentimentAnalysis\code\sentimentAnalysis\preprocessed.csv', encoding='utf-8')
logging.info("data is read")


# Separate the numeric features from the text columns
numeric_columns = ['Sentiment']
X_numeric = df[numeric_columns].values

# Apply TF-IDF to the 'Review' column (text column)
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_review_tfidf = tfidf.fit_transform(df['Review'])

# Combine all features (numeric, TF-IDF, and POS tags)
X_combined = hstack([X_review_tfidf, X_numeric])

# 'Label' column contains sentiment labels (0 for negative, 1 for positive)
y = df['Label']

# Perform Stratified Sampling to reduce the dataset to 10,000 samples
def stratified_sampling(X, y, sampling_fraction, random_state=None):
    # Use X.shape[0] to get the number of samples
    n_samples = int(X.shape[0] * sampling_fraction)
    
    X_resampled, y_resampled = resample(X, y, 
                                       n_samples=n_samples, 
                                       random_state=random_state, 
                                       stratify=y)  # Ensure the class proportions are preserved
    return X_resampled, y_resampled

# Apply stratified sampling on the full dataset to get 10,000 samples
X_sampled, y_sampled = stratified_sampling(X_combined, y, sampling_fraction=0.025, random_state=42)

logging.info("Sampled data is created")

# Split the sampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

logging.info("Data is split into training and testing sets")



logging.info("Dimensionality reduced using UMAP")

# Define the SVM model (Support Vector Machine)
svm = SVC(kernel='linear', probability=True, random_state=42)

logging.info("Training started")

# Train the SVM model on the reduced data
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

logging.info("Prediction is done")

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix for visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Define the Logistic Regression model
logreg = LogisticRegression(max_iter=1000, random_state=42)

logging.info("Training started")

# Train the Logistic Regression model on the reduced data
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

logging.info("Prediction is done")

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix for visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Define the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize the Random Forest Classifier

logging.info("Training started")

# Train the Random Forest model on the reduced data
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

logging.info("Prediction is done")

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix for visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
