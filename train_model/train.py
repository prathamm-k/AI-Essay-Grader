import pandas as pd
import numpy as np
import spacy
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import nltk
import re
from textblob import TextBlob
import os

# Create train_model directory if it doesn't exist
os.makedirs('train_model', exist_ok=True)

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

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

def train_model():
    print("Loading and preparing data...")
    
    # Sample essays with scores (you can replace this with your actual dataset)
    essays = [
        "The impact of technology on modern education has been profound. Digital tools have revolutionized how students learn and teachers teach. Online platforms provide access to vast resources, while interactive software makes learning more engaging. However, challenges like digital divide and screen time concerns remain significant issues to address.",
        "Climate change is one of the most pressing issues of our time. Rising global temperatures, extreme weather events, and melting ice caps are clear indicators of environmental degradation. Immediate action is required to reduce carbon emissions and transition to renewable energy sources. Individual and collective efforts are essential for a sustainable future.",
        "Artificial intelligence is transforming various industries. From healthcare to transportation, AI applications are becoming increasingly common. While these advancements offer numerous benefits, ethical considerations and job displacement concerns must be carefully addressed. The future of AI depends on responsible development and implementation.",
        "The importance of mental health awareness cannot be overstated. Modern life's pressures and social media's influence have increased stress and anxiety levels. Regular exercise, proper sleep, and social connections are crucial for maintaining mental well-being. Society must prioritize mental health education and support systems.",
        "Social media has revolutionized how people connect and communicate. Platforms like Facebook and Twitter have created global communities and facilitated information sharing. However, issues like privacy concerns and misinformation spread require careful consideration. Users must be educated about responsible social media usage."
    ]
    
    scores = [8.5, 9.0, 8.0, 7.5, 7.0]  # Corresponding scores out of 10
    
    # Extract features for each essay
    print("Extracting features...")
    features_list = []
    for essay in essays:
        features = extract_essay_features(essay)
        features_list.append(list(features.values()))
    
    # Convert to numpy array
    X = np.array(features_list)
    y = np.array(scores)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training model...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print("\nModel Performance:")
    print(f"Training R² Score: {train_score:.2f}")
    print(f"Testing R² Score: {test_score:.2f}")
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, "train_model/essay_scoring_model.pkl")
    joblib.dump(scaler, "train_model/feature_scaler.pkl")
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    train_model() 