import streamlit as st
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import altair as alt

# Load environment variables
load_dotenv()

# Get API key and endpoint URL from environment variables
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY") or st.secrets["BACKEND_API_KEY"]
BACKEND_ENDPOINT = os.getenv("BACKEND_ENDPOINT") or st.secrets["BACKEND_ENDPOINT"]

def analyze_text(text):
    headers = {
        "Content-Type": "application/json",
        "x-api-key": BACKEND_API_KEY
    }
    data = {
        "document": text
    }
    
    response = requests.post(BACKEND_ENDPOINT, headers=headers, json=data)
    return response.json()

def display_detailed_results(result):
    try:
        st.subheader("Detailed Results")
        
        # Display overall results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Class", result['predicted_class'].capitalize())
        with col2:
            st.metric("Confidence Category", result['confidence_category'].capitalize())
        with col3:
            st.metric("Confidence Score", f"{result['confidence_score']:.2%}")

        st.info(result['result_message'])
        if result['result_sub_message']:
            st.info(result['result_sub_message'])
        
        # Display class probabilities
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame.from_dict(result['class_probabilities'], orient='index', columns=['Probability'])
        prob_df.index.name = 'Class'
        prob_df = prob_df.reset_index()
        
        chart = alt.Chart(prob_df).mark_bar().encode(
            x='Class',
            y='Probability',
            color='Class'
        ).properties(width=600)
        
        st.altair_chart(chart, use_container_width=True)
        
        # Display sentence-level analysis
        st.subheader("Sentence Analysis")
        sentences_df = pd.DataFrame(result['sentences'])
        sentences_df['generated_prob'] = sentences_df['generated_prob'].apply(lambda x: f"{x:.2%}")
        sentences_df['perplexity'] = sentences_df['perplexity'].apply(lambda x: f"{x:.2f}")
        sentences_df = sentences_df.rename(columns={
            'generated_prob': 'AI Probability',
            'sentence': 'Sentence',
            'perplexity': 'Perplexity',
            'highlight_sentence_for_ai': 'Highlighted for AI'
        })
        st.dataframe(sentences_df, use_container_width=True)
        
        # Display paragraph-level analysis
        st.subheader("Paragraph Analysis")
        paragraphs_df = pd.DataFrame(result['paragraphs'])
        paragraphs_df['completely_generated_prob'] = paragraphs_df['completely_generated_prob'].apply(lambda x: f"{x:.2%}")
        paragraphs_df = paragraphs_df.rename(columns={
            'start_sentence_index': 'Start Sentence Index',
            'num_sentences': 'Number of Sentences',
            'completely_generated_prob': 'AI Generation Probability'
        })
        st.dataframe(paragraphs_df, use_container_width=True)
        
        # Display additional statistics
        st.subheader("Additional Statistics")
        stats_data = {
            "Metric": ["Overall Burstiness", "Average Generated Probability", "Document Classification", "Language", "Detector Version"],
            "Value": [
                f"{result['overall_burstiness']:.2f}",
                f"{result['average_generated_prob']:.2%}",
                result['document_classification'],
                result['language'],
                result['version']
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        st.table(stats_df)
        
    except Exception as e:
        st.error("Error parsing detailed results. Displaying raw JSON instead.")
        st.json(result)

st.set_page_config(page_title="Prerit's AI Detector", page_icon="🤖", layout="wide")

st.title("Prerit's AI Detector")
st.write("Paste your text below to check for the probability of AI-generated content.")

user_input = st.text_area("Enter your text here:", height=200)

if st.button("Analyze Text"):
    if user_input:
        with st.spinner("Analyzing..."):
            result = analyze_text(user_input)
        
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
            
            display_detailed_results(doc)
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
This detector has a tendency to produce false negatives. Sophisticated AI-generated text can sometimes evade detection, resulting in it being classified as human-written. However, it rarely produces false positives. This is because certain characteristics of typical AI-generated text are almost never seen in human-written content. As a result, when the detector identifies text as AI-generated, it's usually highly accurate.
""")