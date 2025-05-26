import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pdfplumber

file_list = os.listdir('files')
print(file_list)

resumes = {}


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

for file in file_list:
    emp_name =  file.split("_")[0]
    resumes[emp_name] = extract_text_from_pdf(f"files\\{file}")


# job_description = "Looking for a python developer who skilled at python and pandas."
job_description = "Looking for a Test engineer who skilled at java and selenium."

resume_df = pd.DataFrame({
    'resume_id': list(resumes.keys()),
    'resume_text': list(resumes.values())
})

documents = resume_df['resume_text'].to_list()
documents.append(job_description)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

similarity_score = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

resume_df["Score"] = similarity_score

print("Resume similarity scores: \n", resume_df[['resume_id', 'Score']])


threshold = 0.2
matching_resumes = resume_df[resume_df['Score'] >= threshold]
print("Matching resume similarity scores: \n", matching_resumes[['resume_id', 'Score']])