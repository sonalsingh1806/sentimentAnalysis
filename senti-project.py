# Import libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load data
data = pd.read_csv("/Users/Aqsa/Desktop/ML/amazon_reviews 2.csv")

# Display basic info
print(data.info())
print(data.head())

# Check for missing values and drop rows with missing values
data.dropna(subset=['Review', 'Label'], inplace=True)
data.drop_duplicates(subset='Review', inplace=True)

# Text Cleaning
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

data['Cleaned_Review'] = data['Review'].apply(clean_text)

# Remove stop words
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])

data['Cleaned_Review'] = data['Cleaned_Review'].apply(remove_stopwords)

# Stratified sampling to reduce dataset size (optional)
sample_size = 10000  # Adjust the sample size as needed
data_sampled, _ = train_test_split(data, train_size=sample_size, stratify=data['Label'], random_state=42)

# Tokenization with DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Check if MPS (Mac GPU) is available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a custom Dataset class
class AmazonReviewsDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten().to(device),  # Move to MPS
            'attention_mask': encoding['attention_mask'].flatten().to(device),  # Move to MPS
            'labels': torch.tensor(label, dtype=torch.long).to(device)  # Move to MPS
        }

# Prepare dataset
reviews = data_sampled['Cleaned_Review'].tolist()
labels = data_sampled['Label'].tolist()
dataset = AmazonReviewsDataset(reviews, labels, tokenizer)

# Split data into training and validation sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Load DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Move model to MPS
model.to(device)

# Define training arguments
""" training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
) """

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,  # Increase epochs
    per_device_train_batch_size=16,  # Experiment with batch size
    per_device_eval_batch_size=16,
    learning_rate=2e-5,  # Adjust learning rate
    warmup_steps=500,
    weight_decay=0.01,  # Adjust weight decay
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Fine-tune the model
trainer.train()

# Evaluate the model on the test set
eval_results = trainer.evaluate()

# Print evaluation metrics
print("\n--- Evaluation Results ---")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall: {eval_results['eval_recall']:.4f}")
print(f"F1 Score: {eval_results['eval_f1']:.4f}")

# Generate predictions on the test set
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)  # Ensure preds is 1D

# Collect all labels and predictions
all_labels = []
all_preds = []

for i in range(len(test_dataset)):
    inputs = test_dataset[i]
    labels = inputs['labels'].cpu().numpy()
    pred = preds[i]  # Access the i-th prediction
    all_labels.append(labels)
    all_preds.append(pred)

# Convert lists to NumPy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(all_labels, all_preds, target_names=['Negative', 'Positive'])
print("\nClassification Report:")
print(class_report)
