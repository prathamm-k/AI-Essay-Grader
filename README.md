# AI Essay Grader

## Project Motivation
I created this project to explore the intersection of natural language processing and automated essay grading. The traditional essay grading process is time-consuming and can be subjective. I wanted to develop a tool that could provide consistent, objective feedback while helping students improve their writing skills, I also wanted to learn how NLP and ML go hand in hand and how can it effect in our daily life.

Through this project, I learned:
- How to implement NLP techniques for text analysis
- The importance of feature engineering in machine learning
- Best practices in model training and evaluation
- The challenges of automated text scoring and how to address them

## Project Overview
AI Essay Grader is an intelligent system that automatically evaluates essays and provides detailed feedback. It uses advanced natural language processing and machine learning techniques to analyze various aspects of writing, including:

- Grammar and sentence structure
- Vocabulary usage and complexity
- Essay organization and coherence
- Sentiment analysis
- Readability metrics

### Technologies Used
- **Python**: Core programming language
- **spaCy**: For advanced NLP tasks and text processing
- **scikit-learn**: For machine learning model implementation
- **Streamlit**: For creating an interactive web interface
- **pandas & numpy**: For data manipulation and numerical computations
- **joblib**: For model persistence
- **NLTK**: For additional text processing capabilities

## Setup Instructions

### 1. Windows Setup (Using setup.ps1)
1. Clone the repository:
   ```bash
   git clone https://github.com/prathamm-k/AI-Essay-Grader.git
   cd AI-Essay-Grader
   ```
2. Run the setup script:
   ```powershell
   .\setup.ps1
   ```
3. Start the application:
   ```bash
   streamlit run app.py
   ```

### 2. Arch Linux Setup (Using setup.sh)
1. Clone the repository:
   ```bash
   git clone https://github.com/prathamm-k/AI-Essay-Grader.git
   cd AI-Essay-Grader
   ```
2. Make the setup script executable:
   ```bash
   chmod +x setup.sh
   ```
3. Run the setup script:
   ```bash
   ./setup.sh
   ```
4. Start the application:
   ```bash
   streamlit run app.py
   ```

### 3. Manual Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/prathamm-k/AI-Essay-Grader.git
   cd AI-Essay-Grader
   ```
2. Create a virtual environment:
   ```bash
   # Windows
   python -m venv essay-venv
   .\essay-venv\Scripts\activate

   # Linux/Mac
   python3 -m venv essay-venv
   source essay-venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
5. Train the model:
   ```bash
   python train_model/train.py
   ```
6. Start the application:
   ```bash
   streamlit run app.py
   ```

## Features
- **Automated Essay Scoring**: Provides numerical scores based on multiple criteria
- **Detailed Feedback**: Analyzes various aspects of writing
- **Interactive Interface**: User-friendly web interface for essay submission
- **Real-time Analysis**: Instant feedback on submitted essays
- **Visual Analytics**: Graphical representation of essay metrics

## Project Structure
```
AI-Essay-Grader/
├── train_model/              # Model training and storage
│   ├── train.py             # Training script
│   ├── essay_dataset.csv    # Training data
│   ├── essay_scoring_model.pkl  # Trained model
│   └── feature_scaler.pkl   # Feature scaler
├── app.py                   # Main Streamlit application
├── requirements.txt         # Project dependencies
├── setup.sh                # Linux setup script
├── setup.ps1               # Windows setup script
└── sample_essay.txt        # Example essay
```

## Contributing
Feel free to contribute to this project by:
1. Forking the repository
2. Creating a new branch
3. Making your changes
4. Submitting a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.