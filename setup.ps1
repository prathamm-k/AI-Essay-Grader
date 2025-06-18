# Function to print colored messages
function Write-ColorMessage {
    param(
        [string]$Message,
        [string]$Color
    )
    Write-Host $Message -ForegroundColor $Color
}

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-ColorMessage "Found Python: $pythonVersion" "Green"
} catch {
    Write-ColorMessage "Python is not installed. Please install Python 3 first." "Red"
    exit 1
}

# Check if pip is installed
try {
    $pipVersion = pip --version
    Write-ColorMessage "Found pip: $pipVersion" "Green"
} catch {
    Write-ColorMessage "pip is not installed. Please install pip first." "Red"
    exit 1
}

# Create virtual environment
Write-ColorMessage "Creating virtual environment..." "Blue"
python -m venv essay-venv

# Activate virtual environment
Write-ColorMessage "Activating virtual environment..." "Blue"
.\essay-venv\Scripts\Activate.ps1

# Upgrade pip
Write-ColorMessage "Upgrading pip..." "Blue"
python -m pip install --upgrade pip

# Install requirements
Write-ColorMessage "Installing requirements..." "Blue"
pip install -r requirements.txt

# Download spaCy model
Write-ColorMessage "Downloading spaCy model..." "Blue"
python -m spacy download en_core_web_sm

# Train the model
Write-ColorMessage "Training the model..." "Blue"
python train_model/train.py

# Check if model files were created
if ((Test-Path "train_model/essay_scoring_model.pkl") -and (Test-Path "train_model/feature_scaler.pkl")) {
    Write-ColorMessage "Model training completed successfully!" "Green"
} else {
    Write-ColorMessage "Model training failed. Please check the error messages above." "Red"
    exit 1
}

Write-ColorMessage "Setup completed successfully!" "Green"
Write-ColorMessage "To run the application, use: streamlit run app.py" "Blue" 