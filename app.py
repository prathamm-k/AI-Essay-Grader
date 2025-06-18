import streamlit as st
import pandas as pd
import numpy as np
import spacy
import joblib
from textblob import TextBlob
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="AI Essay Grader",
    page_icon="��",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def simple_sentence_tokenize(text):
    # Split text into sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def extract_essay_features(essay_text):
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Process text with spaCy
    doc = nlp(essay_text)
    
    # Basic statistics
    word_count = len([token.text for token in doc if not token.is_punct])
    sentence_count = len(simple_sentence_tokenize(essay_text))
    avg_word_length = np.mean([len(token.text) for token in doc if not token.is_punct]) if word_count > 0 else 0
    
    # Sentiment analysis
    blob = TextBlob(essay_text)
    sentiment_score = blob.sentiment.polarity
    
    # Vocabulary diversity
    unique_words = len(set([token.text.lower() for token in doc if not token.is_punct]))
    vocabulary_diversity = unique_words / word_count if word_count > 0 else 0
    
    # Grammar features
    noun_count = len([token for token in doc if token.pos_ == "NOUN"])
    verb_count = len([token for token in doc if token.pos_ == "VERB"])
    adj_count = len([token for token in doc if token.pos_ == "ADJ"])
    
    # Named entities
    entity_count = len(doc.ents)
    
    # Create feature dictionary
    features = {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'sentiment_score': sentiment_score,
        'vocabulary_diversity': vocabulary_diversity,
        'noun_count': noun_count,
        'verb_count': verb_count,
        'adj_count': adj_count,
        'entity_count': entity_count
    }
    
    return features

def load_model():
    try:
        model = joblib.load("train_model/essay_scoring_model.pkl")
        scaler = joblib.load("train_model/feature_scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def create_visualizations(essay_text, features):
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Word length distribution
    plt.subplot(1, 3, 1)
    words = [word for word in essay_text.split() if word.isalpha()]
    word_lengths = [len(word) for word in words]
    sns.histplot(word_lengths, bins=20)
    plt.title('Word Length Distribution')
    plt.xlabel('Word Length')
    plt.ylabel('Count')
    
    # Sentence length distribution
    plt.subplot(1, 3, 2)
    sentences = simple_sentence_tokenize(essay_text)
    sentence_lengths = [len(sent.split()) for sent in sentences]
    sns.histplot(sentence_lengths, bins=10)
    plt.title('Sentence Length Distribution')
    plt.xlabel('Words per Sentence')
    plt.ylabel('Count')
    
    # Part of speech distribution
    plt.subplot(1, 3, 3)
    pos_counts = {
        'Nouns': features['noun_count'],
        'Verbs': features['verb_count'],
        'Adjectives': features['adj_count']
    }
    plt.bar(pos_counts.keys(), pos_counts.values())
    plt.title('Part of Speech Distribution')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def analyze_essay(essay_text):
    if not essay_text.strip():
        st.warning("Please enter an essay to analyze.")
        return
    
    with st.spinner("Analyzing your essay..."):
        # Extract features
        features = extract_essay_features(essay_text)
        
        # Load model and scaler
        model, scaler = load_model()
        if model is None or scaler is None:
            st.error("Failed to load the model. Please try again later.")
            return
        
        # Scale features
        feature_values = np.array([list(features.values())])
        scaled_features = scaler.transform(feature_values)
        
        # Make prediction
        score = model.predict(scaled_features)[0]
        
        # Display results
        st.markdown("## 📊 Analysis Results")
        
        # Score display
        st.markdown("### Overall Score")
        st.markdown(f"""
            <div class="metric-card">
                <h2 style="text-align: center; color: #4CAF50;">{score:.1f}/10</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Basic Statistics")
            stats_data = {
                'Metric': ['Word Count', 'Sentence Count', 'Average Word Length'],
                'Value': [str(features['word_count']), 
                         str(features['sentence_count']),
                         f"{features['avg_word_length']:.1f}"]
            }
            st.dataframe(pd.DataFrame(stats_data), hide_index=True)
        
        with col2:
            st.markdown("### Language Features")
            lang_data = {
                'Feature': ['Vocabulary Diversity', 'Sentiment Score', 'Named Entities'],
                'Value': [f"{features['vocabulary_diversity']*100:.1f}%",
                         f"{features['sentiment_score']:.2f}",
                         str(features['entity_count'])]
            }
            st.dataframe(pd.DataFrame(lang_data), hide_index=True)
        
        with col3:
            st.markdown("### Grammar Analysis")
            grammar_data = {
                'Part of Speech': ['Nouns', 'Verbs', 'Adjectives'],
                'Count': [str(features['noun_count']), 
                         str(features['verb_count']),
                         str(features['adj_count'])]
            }
            st.dataframe(pd.DataFrame(grammar_data), hide_index=True)
        
        # Visualizations
        st.markdown("### 📈 Visual Analysis")
        fig = create_visualizations(essay_text, features)
        st.pyplot(fig)
        
        # Feedback section
        st.markdown("### 💡 Feedback")
        if score >= 8:
            st.success("Excellent essay! Strong writing with good structure and vocabulary.")
        elif score >= 6:
            st.info("Good essay. Consider improving vocabulary and sentence structure.")
        else:
            st.warning("The essay needs improvement. Focus on grammar, structure, and vocabulary.")

def main():
    st.title("📝 AI Essay Grader")
    st.markdown("""
    Welcome to the AI Essay Grader! This tool uses natural language processing and machine learning
    to analyze your essay and provide detailed feedback.
    """)
    
    # Text input area
    essay_text = st.text_area(
        "Enter your essay here:",
        height=300,
        placeholder="Type or paste your essay here..."
    )
    
    # Start Analysis button
    if st.button("Start Analysis", key="analyze"):
        analyze_essay(essay_text)

if __name__ == "__main__":
    main()
