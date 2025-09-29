import spacy
import docx2txt
from PyPDF2 import PdfReader
import re
import os
import requests
import tarfile
from pathlib import Path

def download_and_extract_model(model_name="en_core_web_sm", version="3.8.0"):
    """
    Downloads and extracts a spaCy model to a local directory.
    This avoids permission errors on read-only filesystems like Streamlit Cloud.
    """
    download_url = f"https://github.com/explosion/spacy-models/releases/download/{model_name}-{version}/{model_name}-{version}.tar.gz"
    model_path = Path(f"./{model_name}-{version}")

    # If model is already downloaded, do nothing
    if model_path.exists():
        print(f"Model '{model_name}' already downloaded.")
        return str(model_path / model_name / f"{model_name}-{version}")

    print(f"Downloading model '{model_name}' from {download_url}...")
    
    # Download the model file
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the downloaded file
        tar_path = f"./{model_name}-{version}.tar.gz"
        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete. Extracting...")

        # Extract the tar.gz file
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall()
        print("Extraction complete.")

        # Clean up the downloaded tar.gz file
        os.remove(tar_path)
        
        # Return the path to the extracted model
        return str(model_path / model_name / f"{model_name}-{version}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during model setup: {e}")
        return None


def load_model():
    """
    Loads the spaCy model, downloading it if necessary.
    """
    try:
        # First, try to load it conventionally. This might work in local dev.
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If that fails, use our custom downloader for cloud environments
        print("Could not find 'en_core_web_sm'. Attempting to download and load from local path.")
        model_dir = download_and_extract_model()
        if model_dir:
            nlp = spacy.load(model_dir)
        else:
            print("Failed to download or load the spaCy model. Parsing will be limited.")
            return None
    return nlp

# Load the spaCy model at startup
nlp = load_model()

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        return f"Error reading DOCX: {e}"

def extract_name(text):
    """Extracts the name from the resume text using spaCy's NER."""
    if not nlp: return None
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def extract_contact_info(text):
    """Extracts email and phone number using regex."""
    email = re.search(r'[\w\.\+-]+@[\w\.\+-]+\.[\w\.-]+', text)
    phone = re.search(r'(\(?\+?\d{1,3}\)?[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    return email.group(0) if email else None, phone.group(0) if phone else None

def extract_skills(text, skills_list):
    """Extracts skills from the text based on a predefined list."""
    found_skills = set()
    for skill in skills_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            found_skills.add(skill.capitalize())
    return list(found_skills)

def parse_resume(file_path, skills_list):
    """
    Main function to parse a resume file.
    It extracts text and then specific fields.
    """
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        return {"error": "Unsupported file type"}

    if "Error reading" in text:
        return {"error": text}
    
    if not nlp:
         return {"error": "spaCy model not loaded. Cannot parse."}

    name = extract_name(text)
    email, phone = extract_contact_info(text)
    skills = extract_skills(text, skills_list)

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "full_text": text
    }

