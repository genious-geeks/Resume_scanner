Resume Matching using TF-IDF and Cosine Similarity
This script performs automated matching between a set of resumes (PDF files) and a given job description by measuring textual similarity.

How It Works
Reads all PDF files from the files directory.

Extracts the text content from each resume PDF.

Builds a pandas DataFrame with each resume’s ID and text.

Converts resumes and job description text into TF-IDF vectors.

Calculates cosine similarity scores between each resume and the job description.

Outputs resumes that have a similarity score above a defined threshold.

Requirements
pip install pandas scikit-learn pdfplumber

Usage
Place all resume PDFs inside a files folder.

Update the job_description string with the role requirements.

Run the script.

The output will display similarity scores for all resumes, followed by resumes matching the threshold.
