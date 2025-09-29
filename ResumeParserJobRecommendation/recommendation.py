from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_tfidf_matrix(job_descriptions):
    """
    Builds a TF-IDF matrix from a list of job descriptions.
    
    Args:
        job_descriptions (pd.Series): A pandas Series containing job descriptions.
        
    Returns:
        tuple: A tuple containing the TF-IDF matrix and the vectorizer instance.
    """
    # Using sublinear_tf for diminishing returns on term frequency and max_features to keep the matrix manageable
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=5000, 
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(job_descriptions)
    return tfidf_matrix, vectorizer

def recommend_jobs(parsed_data, job_df, tfidf_matrix, vectorizer, top_n=5):
    """
    Recommends jobs based on the parsed resume data with score boosting for priority roles.
    
    Args:
        parsed_data (dict): The dictionary of parsed resume data.
        job_df (pd.DataFrame): The DataFrame of job postings.
        tfidf_matrix: The pre-computed TF-IDF matrix for the job postings.
        vectorizer: The fitted TfidfVectorizer instance.
        top_n (int): The number of top recommendations to return.
        
    Returns:
        pd.DataFrame: A DataFrame containing the top N recommended jobs.
    """
    # Create a comprehensive text profile from the resume
    resume_skills = ' '.join(parsed_data.get('skills', [])).lower()
    resume_text = parsed_data.get('full_text', '').lower()
    
    # Give more weight to explicitly listed skills for better matching
    profile_text = (resume_skills + ' ') * 3 + resume_text
    
    if not profile_text.strip():
        return pd.DataFrame() # Return empty DataFrame if profile is empty

    # Transform the resume profile into a TF-IDF vector
    profile_vector = vectorizer.transform([profile_text])
    
    # Calculate initial cosine similarity between the resume and all jobs
    cosine_similarities = cosine_similarity(profile_vector, tfidf_matrix).flatten()
    
    # --- Score Boosting for Priority Roles ---
    # Boost the similarity score for jobs with certain keywords in their title.
    # Expanded this list based on the new dataset.
    priority_roles = [
        'data scientist', 'data analyst', 'machine learning engineer', 'ml engineer', 
        'data engineer', 'ai engineer', 'developer', 'engineer', 'analyst', 
        'manager', 'consultant', 'business analyst', 'technician', 'installer',
        'project manager'
    ]
    boost_factor = 0.2  # The amount to boost the score by

    # Ensure job titles are lowercase strings for matching
    job_titles = job_df.get('title', pd.Series([''] * len(job_df))).fillna('').str.lower()
    
    # Create a copy of similarities to modify
    boosted_sim = cosine_similarities.copy()
    
    for i, title in enumerate(job_titles):
        if any(role in title for role in priority_roles):
            boosted_sim[i] += boost_factor
            
    # --- Get Top Recommendations ---
    # Get the indices of the top N most similar jobs from the boosted scores
    if len(boosted_sim) > top_n:
        top_job_indices = boosted_sim.argsort()[-top_n:][::-1]
    else:
        top_job_indices = boosted_sim.argsort()[::-1]

    # Create a DataFrame with the recommended jobs and their boosted scores
    recommendations = job_df.iloc[top_job_indices].copy()
    recommendations['similarity_score'] = boosted_sim[top_job_indices]
    
    # Clamp scores at 1.0 in case boosting pushes them over
    recommendations['similarity_score'] = recommendations['similarity_score'].clip(upper=1.0)
    
    return recommendations

