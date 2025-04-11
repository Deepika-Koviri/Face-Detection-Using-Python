import pandas as pd
import numpy as np
import re
import tldextract
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("malicious.csv")

# Map labels to numeric values
label_mapping = {'benign': 0, 'defacement': 1, 'phishing': 2, 'malware': 3}
df['label'] = df['type'].map(label_mapping)

# Feature Extraction Function
def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special_chars'] = len(re.findall(r"[!@#$%^&*(),.?\":{}|<>]", url))
    
    # Extract domain/subdomain details
    extracted = tldextract.extract(url)
    features['domain_length'] = len(extracted.domain)
    features['subdomain_length'] = len(extracted.subdomain)
    
    return features

# Apply feature extraction
features_df = df['url'].apply(lambda x: pd.Series(extract_features(x)))

# TF-IDF Vectorization on URLs
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(df['url']).toarray()

# Combine extracted numerical features and TF-IDF
X_combined = np.hstack((features_df.values, X_tfidf))

# Target labels
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Save model & vectorizer
joblib.dump(rf, "malicious_url_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

# Predictions & Evaluation
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

print("Model Training Complete. Model Saved!")
