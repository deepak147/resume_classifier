import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


def preprocess(text):
    text = text.encode("utf-8").decode("utf-8")
    alpha_pattern = re.compile("[a-zA-Z]+")

    lines = text.split("\n")
    for i, line in enumerate(lines):
        alpha_only = " ".join(alpha_pattern.findall(line))
        lines[i] = alpha_only

    result_string = " ".join(lines)

    resume_text = result_string.lower()
    tokens = word_tokenize(resume_text)
    stop_words = set(stopwords.words("english"))
    stop_words.update(["name", "city", "company", "state"])
    filtered_tokens = [word for word in tokens if word not in stop_words]

    processed_text = " ".join(filtered_tokens)

    return processed_text


df = pd.read_csv("Resume.csv")
df["Resume"] = df["Resume_str"].apply(preprocess)
label_encoder = LabelEncoder()
df["CategoryEncoded"] = label_encoder.fit_transform(df["Category"])
X, y = df["Resume"], df["CategoryEncoded"]

X_train, X_test, y_train, y_test = train_test_split(
    df["Resume"], df["CategoryEncoded"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create an XGBoost classifier
xgb_classifier = XGBClassifier()

"""cv_scores = cross_val_score(xgb_classifier, X_tfidf, y, cv=5, scoring='accuracy')
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Accuracy: {cv_scores.mean()}')
"""
# Train the XGBoost classifier
xgb_classifier.fit(X_train_tfidf, y_train)

# Test the model
y_pred = xgb_classifier.predict(X_test_tfidf)

y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_decoded, y_pred_decoded)

# Display the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - XGBoost Classifier")

plt.savefig("confusion_matrix_xgboost.png")

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test_decoded, y_pred_decoded))


with open("model.pkl", "wb") as file:
    pickle.dump(xgb_classifier, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

with open("label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)