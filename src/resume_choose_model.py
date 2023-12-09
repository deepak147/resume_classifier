import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


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
    filtered_tokens = [word for word in tokens if word not in stop_words]

    processed_text = " ".join(filtered_tokens)

    return processed_text


df = pd.read_csv("Resume.csv")
df["Resume"] = df["Resume_str"].apply(preprocess)
label_encoder = LabelEncoder()
df["CategoryEncoded"] = label_encoder.fit_transform(df["Category"])

X_train, X_test, y_train, y_test = train_test_split(
    df["Resume"], df["CategoryEncoded"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize models
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
linear_svc_model = LinearSVC(random_state=42)
xgboost_model = XGBClassifier(random_state=42)
naive_bayes_model = MultinomialNB()

# List of models
models = [
    ("Random Forest", random_forest_model),
    ("Linear SVC", linear_svc_model),
    ("XGBoost", xgboost_model),
    ("Naive Bayes", naive_bayes_model),
]

for name, model in models:
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring="accuracy")
    print(f"{name} - Cross-Validation Scores: {cv_scores}")
    print(f"{name} - Mean CV Accuracy: {cv_scores.mean()}")
    print("\n")

# Train models
random_forest_model.fit(X_train_tfidf, y_train)
linear_svc_model.fit(X_train_tfidf, y_train)
xgboost_model.fit(X_train_tfidf, y_train)
naive_bayes_model.fit(X_train_tfidf, y_train)

# Make predictions
random_forest_predictions = random_forest_model.predict(X_test_tfidf)
linear_svc_predictions = linear_svc_model.predict(X_test_tfidf)
xgboost_predictions = xgboost_model.predict(X_test_tfidf)
naive_bayes_predictions = naive_bayes_model.predict(X_test_tfidf)

# Evaluate models and create confusion matrices
models = [
    ("Random Forest", random_forest_predictions),
    ("Linear SVC", linear_svc_predictions),
    ("XGBoost", xgboost_predictions),
    ("Naive Bayes", naive_bayes_predictions),
]

y_test_decoded = label_encoder.inverse_transform(y_test)

for model_name, predictions in models:
    y_pred_decoded = label_encoder.inverse_transform(predictions)
    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    conf_matrix = confusion_matrix(y_test_decoded, y_pred_decoded)

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
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    # Classification Report
    print(f"\nClassification Report - {model_name}:\n")
    print(classification_report(y_test_decoded, y_pred_decoded))
