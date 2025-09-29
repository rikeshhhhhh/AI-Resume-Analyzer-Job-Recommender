import re
import spacy
import docx2txt
from PyPDF2 import PdfReader
import os

# Ensure spaCy model is downloaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() if page.extract_text() else ''
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    text = ""
    try:
        text = docx2txt.process(file_path)
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text

def extract_name(text):
    """Extracts the name from the text using spaCy's NER."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Often, the first PERSON entity is the candidate's name
            return ent.text
    # Fallback: check first few lines for a potential name
    for line in text.split('\n')[:3]:
        if len(line.split()) < 4 and len(line.strip()) > 0 and '@' not in line:
            return line.strip()
    return None

def extract_email(text):
    """Extracts email address using regex."""
    match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return match.group(0) if match else None

def extract_phone(text):
    """Extracts phone number using a more comprehensive regex."""
    # This regex is designed to find various phone number formats
    pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    match = re.search(pattern, text)
    return match.group(0) if match else None

def extract_skills(text, skills_list):
    """Extracts skills from the text based on a predefined list."""
    found_skills = set()
    text_lower = text.lower()
    for skill in skills_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.add(skill.capitalize())
    return list(found_skills)

def parse_resume(file_path, skills_list):
    """
    Parses a resume file (PDF or DOCX) to extract key information.
    """
    file_extension = os.path.splitext(file_path)[1]
    
    if file_extension == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif file_extension == ".docx":
        text = extract_text_from_docx(file_path)
    else:
        return {}

    # Clean up text by removing extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    name = extract_name(text)
    email = extract_email(text)
    phone = extract_phone(text)
    skills = extract_skills(text, skills_list)
    
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "full_text": text  # Include the full text for comprehensive matching
    }
