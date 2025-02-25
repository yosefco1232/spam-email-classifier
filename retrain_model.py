import pandas as pd
from utils import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(filename='retrain.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the original dataset
try:
    original_df = pd.read_csv("spam_ham_dataset.csv")
    logging.info("Original dataset loaded successfully.")
except Exception as e:
    logging.error(f"Error loading original dataset: {e}")
    raise

# Load the updated dataset
try:
    updated_df = pd.read_csv("updated_dataset.csv")
    logging.info("Updated dataset loaded successfully.")
except Exception as e:
    logging.error(f"Error loading updated dataset: {e}")
    raise

# Validate updated dataset
if not all(col in updated_df.columns for col in ["text", "label"]):
    logging.error("Updated dataset is missing required columns: 'text' or 'label'.")
    raise ValueError("Updated dataset must contain 'text' and 'label' columns.")

if not all(label in ["spam", "ham"] for label in updated_df["label"].unique()):
    logging.error("Updated dataset contains invalid labels.")
    raise ValueError("Labels in updated dataset must be 'spam' or 'ham'.")

# Combine datasets
combined_df = pd.concat([original_df, updated_df], ignore_index=True)
logging.info(f"Combined dataset size: {len(combined_df)}")

# Preprocess the data
combined_df['cleaned_text'] = combined_df['text'].apply(clean_text)

# Feature extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(combined_df['cleaned_text'])
y = combined_df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="spam")
recall = recall_score(y_test, y_pred, pos_label="spam")
f1 = f1_score(y_test, y_pred, pos_label="spam")

logging.info(f"Model evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Save the updated model and vectorizer with versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"spam_classifier_{timestamp}.pkl"
tfidf_filename = f"tfidf_vectorizer_{timestamp}.pkl"

joblib.dump(model, model_filename)
joblib.dump(tfidf, tfidf_filename)
logging.info(f"Model and vectorizer saved as {model_filename} and {tfidf_filename}.")