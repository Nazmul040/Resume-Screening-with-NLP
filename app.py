import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load Models and Encoders
# -------------------------------
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
clf = pickle.load(open('clf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# -------------------------------
# Helper Function to Clean Resume Text
# -------------------------------
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Resume Category Classifier", layout="wide")
st.title("📄 Resume Category Prediction App")
st.markdown("Upload or paste your resume text below to predict the job category using a trained ML model (TF-IDF + SVM).")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("This app uses a pre-trained Support Vector Machine (SVM) model with TF-IDF features to classify resumes into job categories.")

# Option to upload a text file or paste text
option = st.radio("Select Input Method:", ("📝 Paste Resume Text", "📁 Upload Resume Text File"))

resume_text = ""

if option == "📝 Paste Resume Text":
    resume_text = st.text_area("Paste your resume text here:", height=250)
else:
    uploaded_file = st.file_uploader("Upload a .txt file containing resume text", type=['txt'])
    if uploaded_file is not None:
        resume_text = uploaded_file.read().decode("utf-8")

# -------------------------------
# Prediction Section
# -------------------------------
if st.button("🔍 Predict Category"):
    if resume_text.strip() == "":
        st.warning("⚠️ Please paste or upload a resume first.")
    else:
        cleaned_text = cleanResume(resume_text)
        vectorized_text = tfidf.transform([cleaned_text]).toarray()
        prediction = clf.predict(vectorized_text)
        category = le.inverse_transform(prediction)[0]
        st.success(f"✅ Predicted Job Category: **{category}**")

# -------------------------------
# (Optional) Data Insights Section
# -------------------------------
st.markdown("---")
st.subheader("📊 Optional: Dataset Category Distribution")
show_plot = st.checkbox("Show Category Distribution Chart")

if show_plot:
    try:
        df = pd.read_csv("UpdatedResumeDataSet.csv")
        plt.figure(figsize=(12, 5))
        sns.countplot(y=df['Category'], order=df['Category'].value_counts().index)
        plt.title("Resume Category Distribution")
        st.pyplot(plt)
    except Exception as e:
        st.error("⚠️ Could not load dataset file. Please ensure 'UpdatedResumeDataSet.csv' exists in the app directory.")
