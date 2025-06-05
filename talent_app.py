from flask import Flask, render_template, request, jsonify, url_for
import re
import os
from collections import Counter
from werkzeug.utils import secure_filename 
from sentence_transformers import SentenceTransformer, util

# Libraries for Parsing PDF Files 
from PyPDF2 import PdfReader

app = Flask(__name__)


# Load BERT model once
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Configuration 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOB_DESCRIPTIONS_DIR = os.path.join(BASE_DIR, 'data', 'job_descriptions') 
RESUMES_DIR = os.path.join(BASE_DIR, 'data', 'resumes') 
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload folder: {UPLOAD_FOLDER}")

# Helper Functions
def list_text_files(directory):
    """Lists .txt files in the given directory (for JD dropdowns)."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            return []
        return []
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        return files
    except Exception as e:
        print(f"Error listing files in {directory}: {e}")
        return []

def read_text_from_file(filepath):
    """Reads text from .txt or .pdf file."""
    text = ""
    print(f"Attempting to read file: {filepath}")
    try:
        if not os.path.exists(filepath):
            print(f"File does not exist: {filepath}")
            return None
        
        file_extension = filepath.rsplit('.', 1)[1].lower()

        if file_extension == 'txt':
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
            read_success = False
            for enc in encodings_to_try:
                try:
                    with open(filepath, 'r', encoding=enc) as f:
                        text = f.read()
                    print(f"Successfully read .txt file {filepath} with encoding {enc}")
                    read_success = True
                    break 
                except UnicodeDecodeError:
                    print(f"Failed to decode {filepath} with {enc}, trying next...")
                except Exception as e_open: 
                    print(f"Error opening/reading {filepath} with {enc}: {e_open}")
                    text = None 
                    break 
            if not read_success and (text is None or (not text and os.path.getsize(filepath) > 0)):
                 print(f"Could not decode .txt file {filepath} with tried encodings.")
                 return None
        elif file_extension == 'pdf': 
            reader = PdfReader(filepath)
            if not reader.pages:
                print(f"Warning: PDF file {filepath} has 0 pages or is unreadable.")
                return None
            for page_num, page in enumerate(reader.pages):
                extracted_page_text = page.extract_text()
                if extracted_page_text: 
                    text += extracted_page_text + "\n"
                else:
                    print(f"Warning: No text extracted from page {page_num + 1} of PDF: {filepath}")
            if not text: 
                 print(f"Warning: No text extracted from any page of PDF: {filepath}")
            else:
                print(f"Successfully extracted text from PDF: {filepath}")
        else:
            print(f"Unsupported file type for reading (expected .txt or .pdf for this function): {filepath}")
            return None
        
        if not text and os.path.getsize(filepath) > 0: 
            print(f"Warning: File {filepath} has size but no text could be extracted.")
        
        print(f"Read {len(text)} characters from file: {filepath}")
        return text
    except Exception as e:
        print(f"ERROR reading file {filepath}: {e}")
        return None
    

def analyze_suitability_bert(job_desc_text, resume_text):
    """Analyzes similarity using BERT embeddings and cosine similarity."""
    if not job_desc_text or not resume_text:
        return 0.0, [], {"error": "JD or Resume content is empty."}

    # Generate embeddings
    jd_embedding = bert_model.encode(job_desc_text, convert_to_tensor=True)
    resume_embedding = bert_model.encode(resume_text, convert_to_tensor=True)

    # Cosine similarity
    similarity_score = util.pytorch_cos_sim(jd_embedding, resume_embedding).item() * 100

    return similarity_score, [], {
        "jd_length": len(job_desc_text),
        "resume_length": len(resume_text),
        "similarity_method": "BERT cosine similarity"
    }


# Flask Routes for Pages 
@app.route('/')
def home():
    return render_template('home.html', page_title="Welcome")

@app.route('/analyzer-tool')
def analyzer_page():
    job_description_files = list_text_files(JOB_DESCRIPTIONS_DIR)
    return render_template('analyzer.html', 
                           page_title="Candidate Analyzer",
                           job_description_files=job_description_files,
                           resume_files=[]) # Resume dropdown not used for PDF only upload

@app.route('/about')
def about():
    return render_template('about.html', page_title="About Us")

@app.route('/faq')
def faq():
    return render_template('faq.html', page_title="FAQ")

@app.route('/contact')
def contact():
    return render_template('contact.html', page_title="Contact Us")

@app.route('/analyze', methods=['POST'])
def analyze_route_ajax():
    print("\n--- Received /analyze request (Specific Inputs: JD=select/paste, Resume=PDF upload ONLY - SIMPLE ANALYSIS) ---")
    data = request.form
    job_title = data.get('job_title', 'N/A')
    print(f"Job Title from form: {job_title}")

    job_description_content = None
    # For Job Description: Select Existing.txt OR Paste Content
    selected_jd_txt_file = data.get('job_description_file')    
    jd_textarea = data.get('job_description_textarea')        

    print(f"JD Select: {selected_jd_txt_file}")
    print(f"JD Textarea non-empty: {bool(jd_textarea)}")

    if selected_jd_txt_file:
        print(f"Processing SELECTED .txt Job Description: {selected_jd_txt_file}")
        job_description_content = read_text_from_file(os.path.join(JOB_DESCRIPTIONS_DIR, selected_jd_txt_file))
    elif jd_textarea:
        print("Processing PASTED Job Description...")
        job_description_content = jd_textarea
    
    if not job_description_content:
        print("ERROR: No Job Description content was successfully obtained.")
        return jsonify({"error": "Please provide a job description (select .txt or paste)."}), 400
    print(f"Final JD Content length: {len(job_description_content)}")

    # Get Resume Content (PDF Upload Only) 
    resume_content = None
    resume_file_storage = request.files.get('resume_upload') 

    print(f"Resume Upload: {resume_file_storage.filename if resume_file_storage and resume_file_storage.filename else 'No file uploaded'}")

    if resume_file_storage and resume_file_storage.filename != '':
        if resume_file_storage.filename.rsplit('.', 1)[1].lower() == 'pdf': 
            print("Processing UPLOADED Resume (PDF)...")
            filename = secure_filename(resume_file_storage.filename)
            resume_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                resume_file_storage.save(resume_filepath)
                print(f"Resume Upload saved to: {resume_filepath}")
                resume_content = read_text_from_file(resume_filepath)
                # if resume content: os.remove(resume_filepath) 
            except Exception as e:
                print(f"Error saving/reading uploaded resume PDF: {e}")
                return jsonify({"error": f"Error processing uploaded resume: {str(e)}"}), 500
        else:
            print("ERROR: Uploaded resume is not a PDF.")
            return jsonify({"error": "Invalid resume file type. Please upload a PDF."}), 400
            
    if not resume_content:
        print("ERROR: No Resume content was successfully obtained (PDF upload required).")
        return jsonify({"error": "Please upload a resume as a PDF file."}), 400
    print(f"Final Resume Content length: {len(resume_content)}")

    # Perform BERT Analysis 
    match_score, common_terms, debug_info = analyze_suitability_bert(job_description_content, resume_content)
    
    if "error" in debug_info and debug_info["error"]: 
        print(f"Simple Analysis Error: {debug_info['error']}")
        return jsonify({"error": f"Analysis Error: {debug_info['error']}"}), 500

    analysis_results = {
        "job_title": job_title,
        "match_score": float(f"{match_score:.2f}"),
        "common_keywords": common_terms, 
        "debug_info": debug_info 
    }
    print(f"Sending back results (simple analysis): Score {analysis_results['match_score']}")
    return jsonify(analysis_results)

if __name__ == '__main__':
    if not os.path.exists(JOB_DESCRIPTIONS_DIR): os.makedirs(JOB_DESCRIPTIONS_DIR)
    if not os.path.exists(RESUMES_DIR): os.makedirs(RESUMES_DIR) 
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
