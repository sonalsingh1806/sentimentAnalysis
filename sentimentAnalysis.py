import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing
from nltk.tokenize import sent_tokenize  # Correctly import sent_tokenize  # Import sent_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')

# Load stopwords once
stop_words = set(stopwords.words('english'))

# Efficient cleaning function
def clean(review):
    review = review.lower()
    review = re.sub('[^a-zA-Z0-9\s-]+', '', review)  # Simplified regex for cleaning
    review = " ".join([word for word in review.split() if word not in stop_words])  # Removed stopwords
    return review

# Load the dataset
df = pd.read_csv("amazon_reviews.csv")


# EDA
plt.pie(df['Label'].value_counts(), labels=df['Label'].unique().tolist(), autopct='%1.1f%%')
plt.show()

lenght = len(df['Review'][0])
print(f'Length of a sample review: {lenght}')

df['Length'] = df['Review'].str.len()
df.head(10)

word_count = df['Review'][0].split()
print(f'Word count in a sample review: {len(word_count)}')

def word_count(review):
    review_list = review.split()
    return len(review_list)

df['Word_count'] = df['Review'].apply(word_count)
print(df.head(10))

df['mean_word_length'] = df['Review'].map(lambda rev: np.mean([len(word) for word in rev.split()]))
print(df.head(10))

np.mean([len(sent) for sent in sent_tokenize(df['Review'][0])])

df['mean_sent_length'] = df['Review'].map(lambda rev: np.mean([len(sent) for sent in sent_tokenize(rev)]))
print(df.head(10))

def visualize(col):
    print()
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df[col], hue=df['Label'])
    plt.ylabel(col, labelpad=12.5)
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df, x=col, hue='Label')  # Update this line to pass the entire dataframe
    plt.legend(df['Label'].unique())
    plt.xlabel('')
    plt.ylabel('')
    
    plt.show()

# Visualize for all features
features = df.columns.tolist()[2:]  # Skipping the first two columns (Review and Label)
for feature in features:
    visualize(feature)

df = df.drop(features, axis=1)
df.head()
df.info()

# Apply cleaning function (using chunking to process large dataset efficiently)
def process_in_chunks(df, chunk_size=50000):
    reviews_cleaned = []
    for start in tqdm(range(0, len(df), chunk_size), desc="Cleaning Reviews"):
        end = min(start + chunk_size, len(df))
        chunk = df['Review'][start:end].apply(clean)
        reviews_cleaned.extend(chunk)
    return reviews_cleaned

# Clean the reviews in chunks
df['Review'] = process_in_chunks(df)

# Function to split reviews into words
def corpus(text):
    return text.split()

# Apply corpus function to get a list of words for each review
df['Review_lists'] = df['Review'].apply(corpus)

# Combine all reviews into a single list of words (more efficient method)
corpus = [word for review in df['Review_lists'] for word in review]

# Find the 10 most common words using nltk.FreqDist
fdist = nltk.FreqDist(corpus)
mostCommon = fdist.most_common(10)
print("Most common words:", mostCommon)

# Prepare data for plotting
words = [item[0] for item in mostCommon]
freq = [item[1] for item in mostCommon]

# Plot the top 10 most common words
sns.barplot(x=freq, y=words)
plt.title('Top 10 Most Frequently Occurring Words')
plt.show()

# Bigram analysis using CountVectorizer
cv = CountVectorizer(ngram_range=(2, 2), stop_words='english', max_features=1000)  # Limit to top 1000 bigrams
bigrams = cv.fit_transform(df['Review'])

count_values = bigrams.toarray().sum(axis=0)
ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv.vocabulary_.items()], reverse=True))
ngram_freq.columns = ["frequency", "ngram"]

# Plot the top 10 most frequent bigrams
sns.barplot(x=ngram_freq['frequency'][:10], y=ngram_freq['ngram'][:10])
plt.title('Top 10 Most Frequently Occurring Bigrams')
plt.show()

# Trigram analysis using CountVectorizer
cv1 = CountVectorizer(ngram_range=(3, 3), stop_words='english', max_features=1000)  # Limit to top 1000 trigrams
trigrams = cv1.fit_transform(df['Review'])

count_values = trigrams.toarray().sum(axis=0)
ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv1.vocabulary_.items()], reverse=True))
ngram_freq.columns = ["frequency", "ngram"]

# Plot the top 10 most frequent trigrams
sns.barplot(x=ngram_freq['frequency'][:10], y=ngram_freq['ngram'][:10])
plt.title('Top 10 Most Frequently Occurring Trigrams')
plt.show()
