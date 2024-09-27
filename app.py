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
        
        # Dynamically change color based on predicted class
        if result['predicted_class'].lower() == 'human':
            st.success(str(result['result_message']).replace("Our detector is", "I am"))
            if result['result_sub_message']:
                st.success(str(result['result_sub_message']).replace("Our detector is", "I am"))
        else:
            st.error(str(result['result_message']).replace("Our detector is", "I am"))
            if result['result_sub_message']:
                st.error(str(result['result_sub_message']).replace("Our detector is", "I am"))
        
        # Create two columns for class probabilities and paragraph analysis
        col_class, col_para = st.columns(2)
        
        with col_class:
            # Display class probabilities
            st.subheader("Class Probabilities")
            prob_df = pd.DataFrame.from_dict(result['class_probabilities'], orient='index', columns=['Probability'])
            prob_df.index.name = 'Class'
            prob_df = prob_df.reset_index()
            
            chart = alt.Chart(prob_df).mark_bar().encode(
                x='Class',
                y='Probability',
                color='Class'
            ).properties(width=300, height=300)
            
            st.altair_chart(chart, use_container_width=True)
        
        with col_para:
            # Display paragraph-level analysis
            st.subheader("Paragraph Analysis")
            paragraphs_df = pd.DataFrame(result['paragraphs'])
            paragraphs_df['Paragraph'] = range(1, len(paragraphs_df) + 1)
            paragraphs_df['End Sentence Index'] = paragraphs_df['start_sentence_index'] + paragraphs_df['num_sentences'] - 1
            paragraphs_df['AI Probability'] = paragraphs_df['completely_generated_prob'].apply(lambda x: f"{x:.2%}")
            paragraphs_df['AI Prob Value'] = paragraphs_df['completely_generated_prob']
            paragraphs_df = paragraphs_df.rename(columns={
                'start_sentence_index': 'Start Sentence',
                'num_sentences': 'Sentence Count',
            })
            paragraphs_df = paragraphs_df[['Paragraph', 'Start Sentence', 'End Sentence Index', 'Sentence Count', 'AI Probability', 'AI Prob Value']]
            
            # Create a custom Altair chart for paragraph analysis
            base = alt.Chart(paragraphs_df).encode(
                y=alt.Y('Paragraph:O', axis=alt.Axis(title='Paragraph'))
            )

            bar = base.mark_bar().encode(
                x=alt.X('AI Prob Value:Q', title='AI Generation Probability'),
                color=alt.Color('AI Prob Value:Q', scale=alt.Scale(scheme='redblue', domain=[0, 1]), legend=None)
            )

            text = base.mark_text(align='left', dx=5).encode(
                x=alt.value(0),
                text=alt.Text('AI Probability:N')
            )

            paragraph_chart = (bar + text).properties(width=300, height=30 * len(paragraphs_df))
            st.altair_chart(paragraph_chart, use_container_width=True)
        
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

        # Display additional statistics
        st.subheader("Additional Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Overall Burstiness", f"{result['overall_burstiness']:.2f}")
        
        with col2:
            st.metric("Average Generated Probability", f"{result['average_generated_prob']:.2%}")
        
        st.markdown("---")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.markdown("**Document Classification**")
            classification = result['document_classification']
            if classification == "AI_ONLY":
                st.markdown("🤖 AI Only")
            elif classification == "HUMAN_ONLY":
                st.markdown("👤 Human Only")
            else:
                st.markdown("🤖👤 Mixed")
        
        with col4:
            st.markdown("**Language**")
            st.markdown(f"🌐 {result['language']}")
        
        with col5:
            st.markdown("**Detector Version**")
            st.markdown(f"🔍 {result['version']}")
        
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
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                ai_prob = doc['class_probabilities']['ai']
                st.metric("AI Probability", f"{ai_prob:.2%}")
            
            with col2:
                st.metric("Predicted Class", doc['predicted_class'].capitalize())
            
            with col3:
                st.metric("Confidence", doc['confidence_category'].capitalize())
            
            with col4:
                st.metric("Confidence Score", f"{doc['confidence_score']:.2%}")
            
            display_detailed_results(doc)
        else:
            st.error("Error in processing the request. Please try again.")
    else:
        st.warning("Please enter some text to analyze.")

st.sidebar.title("How it works")
st.sidebar.info("""
1. Text Analysis: The detector examines various linguistic features of the input text, such as sentence structure, vocabulary usage, and writing style.
2. Pattern Recognition: It looks for patterns that are commonly associated with AI-generated text, such as unusual word combinations or overly consistent writing styles.
3. Statistical Models: The detector employs machine learning models trained on large datasets of both human-written and AI-generated text to make its predictions.
4. Probability Calculation: Based on the analysis, it calculates the probability of the text being AI-generated and provides a confidence level for its prediction.

Important Note:
This detector has a tendency to produce false negatives. Sophisticated AI-generated text can sometimes evade detection, resulting in it being classified as human-written. However, it rarely produces false positives. This is because certain characteristics of typical AI-generated text are almost never seen in human-written content. So when the detector identifies text as AI-generated, it's usually highly accurate.
""")