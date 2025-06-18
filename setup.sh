#!/bin/bash

# Print colorful messages
print_message() {
    echo -e "\e[1;34m$1\e[0m"
}

print_error() {
    echo -e "\e[1;31m$1\e[0m"
}

print_success() {
    echo -e "\e[1;32m$1\e[0m"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Create virtual environment
print_message "Creating virtual environment..."
python3 -m venv essay-venv

# Activate virtual environment
print_message "Activating virtual environment..."
source essay-venv/bin/activate

# Upgrade pip
print_message "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_message "Installing requirements..."
pip install -r requirements.txt

# Download spaCy model
print_message "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Train the model
print_message "Training the model..."
python train_model/train.py

# Check if model files were created
if [ -f "train_model/essay_scoring_model.pkl" ] && [ -f "train_model/feature_scaler.pkl" ]; then
    print_success "Model training completed successfully!"
else
    print_error "Model training failed. Please check the error messages above."
    exit 1
fi

print_success "Setup completed successfully!"
print_message "To run the application, use: streamlit run app.py" 