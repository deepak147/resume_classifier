import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


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

    with open("cleaned.txt", "w", encoding="utf-8") as file:
        file.write(str(processed_text))

    return processed_text
