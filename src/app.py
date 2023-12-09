import PyPDF2
import pickle
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from preprocess import preprocess


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
    with open("label_encoder.pkl", "rb") as file:
        label_encoder = pickle.load(file)

    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files["file"]

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        # Check file extension
        if file and file.filename.rsplit(".", 1)[1].lower() == "pdf":
            # Save the file to the uploads folder
            filename = secure_filename(file.filename)
            file_path = "uploads/" + filename
            file.save(file_path)

            # Extract text from PDF
            raw_text = extract_text_from_pdf(file_path)
            with open("rawtext.txt", "w", encoding="utf-8") as file:
                file.write(str(raw_text))
            clean_text = preprocess(raw_text)
            vectorized_text = vectorizer.transform([clean_text])
            probabilities = model.predict_proba(vectorized_text)
            sorted_probabilities = np.sort(probabilities[0])
            print(sorted_probabilities)
            top_two_indices = np.argsort(probabilities[0])[-2:]
            top_two_categories = model.classes_[top_two_indices]
            print(top_two_indices)
            decoded_label_top1 = label_encoder.inverse_transform(
                [top_two_categories[1]]
            )
            decoded_label_top2 = label_encoder.inverse_transform(
                [top_two_categories[0]]
            )
            print("Top Two Predicted Categories:")
            print(top_two_categories)
            text = (
                str(decoded_label_top1[0])
                + "  Probability: "
                + str(round(sorted_probabilities[-1] * 100, 2))
                + "\n"
                + str(decoded_label_top2[0])
                + "  Probability: "
                + str(round(sorted_probabilities[-2] * 100, 2))
            )

            # Render result in UI
            return render_template("result.html", text=text)

        else:
            return render_template(
                "index.html", error="Invalid file format. Please upload a PDF file."
            )

    return render_template("index.html", error=None)


def extract_text_from_pdf(file_path):
    # Extract text from a PDF file
    text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text


if __name__ == "__main__":
    app.run(debug=True)
