import streamlit as st
import os
import pandas as pd
import re
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from parser import parse_resume
from recommendation import build_tfidf_matrix, recommend_jobs

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(layout="wide")
    st.title("📄 AI Resume Analyzer & Job Recommender ")

    # --- Initialize Session State for History ---
    if 'history' not in st.session_state:
        st.session_state.history = []

    # --- Sidebar for Mode Selection and History ---
    st.sidebar.header("Controls")
    app_mode = st.sidebar.radio("Choose Mode", ["Single Resume Analysis", "Batch Processing"])

    st.sidebar.header("📜 Processing History")
    if st.session_state.history:
        if st.sidebar.button("Clear History"):
            st.session_state.history = []
            st.rerun()
        
        for record in reversed(st.session_state.history):
            with st.sidebar.expander(f"{record['filename']} at {record['timestamp']}"):
                st.json(record['recommendations'])
    else:
        st.sidebar.info("No processing history yet.")


    # --- Load Job Postings Dataset ---
    job_df = None
    st.sidebar.header("Job Data")
    job_data_file = st.sidebar.file_uploader("Upload your job postings CSV", type=["csv"])
    if job_data_file is not None:
        job_df = pd.read_csv(job_data_file)
    else:
        # Fallback to a default file if one exists
        default_dataset_path = "reduced_postings.csv" 
        if os.path.exists(default_dataset_path):
            job_df = pd.read_csv(default_dataset_path)
        else:
            st.warning("Please upload a job postings CSV file in the sidebar to begin.")
            st.stop()
            
    # --- Data Cleaning and Preparation ---
    job_df.columns = job_df.columns.str.strip().str.lower()
    if 'skills_desc' not in job_df.columns or job_df['skills_desc'].isnull().all():
        if 'description' in job_df.columns:
            job_df['skills_desc'] = job_df['description']
        else:
            job_df['skills_desc'] = ''
    job_df['skills_desc'] = job_df['skills_desc'].fillna('')

    def clean_text(text):
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    job_df['cleaned_description'] = job_df['skills_desc'].apply(clean_text)
    
    # Pre-build TF-IDF matrix
    tfidf_matrix, vectorizer = build_tfidf_matrix(job_df['cleaned_description'])

    # --- Main Application Logic based on Mode ---
    if app_mode == "Single Resume Analysis":
        run_single_analysis(job_df, tfidf_matrix, vectorizer)
    else: # Batch Processing
        run_batch_processing(job_df, tfidf_matrix, vectorizer)


def run_single_analysis(job_df, tfidf_matrix, vectorizer):
    st.header("Single Resume Analysis")
    uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"], key="single_uploader")

    if uploaded_file is not None:
        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            # Setup temp directory
            temp_dir = "temp"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # A more comprehensive list of skills for better matching
            skills_list = ['python', 'java', 'sql', 'machine learning', 'nlp', 'data analysis', 'deep learning', 'tableau', 'power bi', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'react', 'javascript', 'html', 'css', 'selenium', 'beautifulsoup', 'scikit-learn', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'r programming']
            
            parsed_data = parse_resume(file_path, skills_list)

        st.success(f"Successfully analyzed {uploaded_file.name}!")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📝 Extracted Information")
            st.json({
                "Name": parsed_data.get("name"),
                "Email": parsed_data.get("email"),
                "Phone": parsed_data.get("phone"),
                "Skills": parsed_data.get("skills", [])
            })

            st.subheader("☁️ Resume Word Cloud")
            if parsed_data.get("full_text"):
                try:
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(parsed_data["full_text"])
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not generate word cloud: {e}")

        with col2:
            st.subheader("✨ Top Job Recommendations")
            recommendations = recommend_jobs(parsed_data, job_df, tfidf_matrix, vectorizer, top_n=5)
            if not recommendations.empty:
                # Add to history
                history_record = {
                    'filename': uploaded_file.name,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'recommendations': recommendations[['title', 'similarity_score']].to_dict('records')
                }
                st.session_state.history.append(history_record)
                
                for _, row in recommendations.iterrows():
                    st.markdown("---")
                    st.markdown(f"#### {row.get('title', 'No Title')}")
                    st.write(f"**Similarity Score:** {row['similarity_score']:.2f}")
                    st.progress(row['similarity_score'])
                    link = row.get('application_url', '')
                    if pd.notna(link) and link.strip() != '':
                        st.markdown(f"**Apply Here:** [{link}]({link})")
            else:
                st.warning("Could not generate recommendations based on the resume.")

        # Clean up
        os.remove(file_path)

def run_batch_processing(job_df, tfidf_matrix, vectorizer):
    st.header("Batch Resume Processing")
    uploaded_files = st.file_uploader("Upload multiple resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True, key="batch_uploader")

    if uploaded_files:
        if st.button("Start Batch Analysis", type="primary"):
            st.info(f"Analyzing {len(uploaded_files)} resumes. This may take a moment...")
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Setup temp directory
                    temp_dir = "temp"
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    skills_list = ['python', 'java', 'sql', 'machine learning', 'nlp', 'data analysis', 'deep learning', 'tableau', 'power bi', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'react', 'javascript', 'html', 'css', 'selenium', 'beautifulsoup', 'scikit-learn', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'r programming']
                    
                    parsed_data = parse_resume(file_path, skills_list)
                    recommendations = recommend_jobs(parsed_data, job_df, tfidf_matrix, vectorizer, top_n=3)

                with st.expander(f"Results for: {uploaded_file.name}", expanded=True):
                    st.json({
                        "Name": parsed_data.get("name"),
                        "Email": parsed_data.get("email"),
                        "Skills": parsed_data.get("skills", [])
                    })
                    st.write("**Top Job Recommendations:**")
                    st.dataframe(recommendations[['title', 'similarity_score', 'application_url']])
                
                # Add to history
                history_record = {
                    'filename': uploaded_file.name,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'recommendations': recommendations[['title', 'similarity_score']].to_dict('records')
                }
                st.session_state.history.append(history_record)
                
                # Clean up
                os.remove(file_path)
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.success("Batch processing complete!")


if __name__ == "__main__":
    main()


