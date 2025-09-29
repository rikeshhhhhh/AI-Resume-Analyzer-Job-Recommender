import spacy
import docx2txt
from PyPDF2 import PdfReader
import re
from spacy.cli import download as spacy_download
import os

def load_model(model_name="en_core_web_sm"):
    """
    Loads a spaCy model. If the model is not found, it downloads it.
    This is a workaround for deployment environments with restricted permissions.
    """
    try:
        # Try loading the model directly
        nlp = spacy.load(model_name)
    except OSError:
        # If model is not found, download it
        print(f"Spacy model '{model_name}' not found. Downloading...")
        spacy_download(model_name)
        nlp = spacy.load(model_name)
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

