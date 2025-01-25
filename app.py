from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from serpapi.google_search import GoogleSearch
import pdfplumber
from io import BytesIO
from datetime import datetime, timedelta
import random
import time
from dateutil.parser import parse
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Calculate the date 24 hours ago
time_24_hours_ago = datetime.now() - timedelta(days=1)
formatted_time = time_24_hours_ago.strftime('%Y-%m-%d')

# Temporary storage for jobs
job_listings_temp = []

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Utility: Extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            return " ".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# List of common User-Agent strings to rotate
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/17.17134",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0"
]

# Utility: Fetch job descriptions from a URL
def fetch_job_description(job_url):
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    try:
        response = requests.get(job_url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching job description: {e}")
        return ""

    time.sleep(2)

# Utility: Format job posting date (in hours ago, days ago, or exact date)
def format_posting_date(posting_time):
    try:
        posting_date = parse(posting_time)
        now = datetime.now()
        diff = now - posting_date

        if diff < timedelta(hours=1):
            return f"{int(diff.total_seconds() // 60)} minutes ago"
        elif diff < timedelta(days=1):
            return f"{int(diff.total_seconds() // 3600)} hours ago"
        elif diff < timedelta(days=7):
            return f"{int(diff.total_seconds() // 86400)} days ago"
        else:
            return posting_date.strftime("%Y-%m-%d")
    except Exception as e:
        return "Unknown date"

# Function to get BERT embeddings for a given text
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Function to calculate match score and provide a one-liner explanation
def calculate_match_score_with_explanation(resume_text, job_description):
    if not resume_text or not job_description:
        return 0, "No relevant data to match"

    resume_embeddings = get_bert_embeddings(resume_text)
    job_embeddings = get_bert_embeddings(job_description)

    similarity_score = cosine_similarity(resume_embeddings.numpy(), job_embeddings.numpy())[0][0]

    if similarity_score < 0.2:
        return 0, "The resume does not match the job description."
    
    score = int(similarity_score * 10)

    # Explanation based on matching concepts
    explanation = f"Semantic similarity score: {similarity_score:.2f} (mapped to {score}/10)"
    return score, explanation

@app.route('/fetch_jobs', methods=['POST'])
def fetch_jobs():
    try:
        if 'designation' not in request.form:
            return jsonify({"message": "Missing 'designation' in the request."}), 400

        designation = request.form['designation']

        api_key = "ed668e92ed4deb47b03aeaee4b939b0b690eb2bed1f8e394663bacb5fba493f7"
        query = f'intitle:"{designation}" AND ("Remote" OR "Anywhere" OR "WFH" OR "Work from home") AND "India" after:{formatted_time}'
        search_params = {"q": query, "api_key": api_key, "engine": "google"}

        try:
            search = GoogleSearch(search_params)
            results = search.get_dict()
            jobs = [
                {"name": r.get("title"), "link": r.get("link"), "posted_date": format_posting_date(r.get("date")), "explanation": ""}
                for r in results.get("organic_results", [] )
                if r.get("title") and r.get("link")
            ]
        except Exception as e:
            print(f"Error during SerpApi call: {e}")
            return jsonify({"message": "Failed to fetch jobs."}), 500

        if not jobs:
            return jsonify({"message": "No jobs found."}), 404

        global job_listings_temp
        job_listings_temp = jobs
        return jsonify(jobs)

    except Exception as e:
        print(f"Error processing /fetch_jobs: {e}")
        return jsonify({"message": "Internal server error."}), 500

@app.route('/process_resume', methods=['POST'])
def process_resume():
    print("Processing resume...")  
    if "resume" not in request.files:
        return jsonify({"message": "Missing 'resume' in the request."}), 400

    resume_file = request.files["resume"]
    if resume_file.filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(resume_file)
    else:
        resume_text = resume_file.read().decode("utf-8", errors="ignore")

    if not job_listings_temp:
        return jsonify({"message": "No job listings available to score."}), 400

    top_jobs = []
    for job in job_listings_temp:
        job_description = fetch_job_description(job["link"])
        score, explanation = calculate_match_score_with_explanation(resume_text, job_description)
        if score >= 5:
            job["explanation"] = explanation  # Update explanation in the job listing
            top_jobs.append(
                {
                    "name": job["name"],
                    "link": job["link"],
                    "score": score,
                    "explanation": explanation,
                    "posted_date": job["posted_date"]
                }
            )

    if not top_jobs:
        return jsonify({"message": "No top jobs found for your resume."}), 404

    return jsonify(top_jobs)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
