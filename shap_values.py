import shap
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset for demonstration
data = {
    'text': ["this is a good product", "not satisfied with the service", "awesome experience"],
    'label': [1, 0, 1]  # Example binary labels
}

df = pd.DataFrame(data)

# Preprocess the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Train a classifier (you can replace this with any classifier you prefer)
classifier = make_pipeline(TfidfVectorizer(), LogisticRegression())
classifier.fit(df['text'], df['label'])

# Create a SHAP explainer for the model
explainer = shap.Explainer(classifier.named_steps['logisticregression'], X_train)

# Define a function to evaluate the effect of each word
def evaluate_word_effects(text):
    # Transform the input text using the vectorizer
    text_vectorized = vectorizer.transform([text])
    # Compute SHAP values for the input text
    shap_values = explainer.shap_values(text_vectorized)
    # Calculate the average SHAP value for each feature (word)
    avg_shap_values = np.mean(shap_values[1], axis=0)  # Assuming binary classification
    # Pair each word with its SHAP value
    word_shap_pairs = list(zip(vectorizer.get_feature_names(), avg_shap_values))
    # Sort the pairs based on SHAP value magnitude
    word_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return word_shap_pairs

# Example usage
input_text = "this product is not good"
word_effects = evaluate_word_effects(input_text)
print("Word effects:")
for word, effect in word_effects:
    print(f"Word: {word}, Effect: {effect:.4f}")
