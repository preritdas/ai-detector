import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("GPTZERO_API_KEY")

def check_ai_content(text):
    url = "https://api.gptzero.me/v2/predict/text"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
    data = {
        "document": text
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

st.set_page_config(page_title="Prerit's AI Detector", page_icon="🤖", layout="wide")

st.title("Prerit's AI Detector")
st.write("Paste your text below to check for the probability of AI-generated content.")

user_input = st.text_area("Enter your text here:", height=200)

if st.button("Check for AI Content"):
    if user_input:
        with st.spinner("Analyzing..."):
            result = check_ai_content(user_input)
        
        if "documents" in result and len(result["documents"]) > 0:
            doc = result["documents"][0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ai_prob = doc['class_probabilities']['ai']
                st.metric("AI Probability", f"{ai_prob:.2%}")
            
            with col2:
                st.metric("Predicted Class", doc['predicted_class'].capitalize())
            
            with col3:
                st.metric("Confidence", doc['confidence_category'].capitalize())
            
            st.subheader("Class Probabilities")
            st.json(doc['class_probabilities'])
            
            st.subheader("Detailed Results")
            st.json(doc)
        else:
            st.error("Error in processing the request. Please try again.")
    else:
        st.warning("Please enter some text to analyze.")

st.sidebar.title("About")
st.sidebar.info("""
How it works:
1. Text Analysis: The detector examines various linguistic features of the input text, such as sentence structure, vocabulary usage, and writing style.
2. Pattern Recognition: It looks for patterns that are commonly associated with AI-generated text, such as unusual word combinations or overly consistent writing styles.
3. Statistical Models: The detector employs machine learning models trained on large datasets of both human-written and AI-generated text to make its predictions.
4. Probability Calculation: Based on the analysis, it calculates the probability of the text being AI-generated and provides a confidence level for its prediction.

Important Note:
This detector has a tendency to produce false negatives. Effective prompt engineering can sometimes evade detection, resulting in AI-generated text being classified as human-written. However, it rarely produces false positives. This is because certain characteristics of default AI behavior are almost never seen in human-written content. As a result, when the detector identifies text as AI-generated, it's usually highly accurate.
""")