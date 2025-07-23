#!/bin/bash

# Amazon Q CLI MLOps Demo Environment Setup Script
# This script sets up the complete demo environment

set -e

echo "🚀 Setting up Amazon Q CLI MLOps Demo Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
else
    print_error "Unsupported operating system: $OSTYPE"
    exit 1
fi

print_status "Detected OS: $OS"

# Check prerequisites
print_status "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
print_status "Python version: $PYTHON_VERSION"

# Check pip
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip3."
    exit 1
fi

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    print_warning "AWS CLI not found. Installing AWS CLI..."
    if [[ "$OS" == "macOS" ]]; then
        curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
        sudo installer -pkg AWSCLIV2.pkg -target /
        rm AWSCLIV2.pkg
    else
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        sudo ./aws/install
        rm -rf aws awscliv2.zip
    fi
fi

# Check Git
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git."
    exit 1
fi

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install Python requirements
print_status "Installing Python requirements..."
pip install -r requirements.txt

# Install Amazon Q CLI (placeholder - actual installation may vary)
print_status "Installing Amazon Q CLI..."
# Note: Replace with actual Q CLI installation command when available
print_warning "Please install Amazon Q CLI manually from AWS documentation"

# Setup AWS credentials check
print_status "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    print_warning "AWS credentials not configured. Please run 'aws configure' to set up your credentials."
else
    print_status "AWS credentials configured successfully"
fi

# Create demo directories
print_status "Creating demo directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p logs
mkdir -p outputs

# Download sample data (placeholder)
print_status "Setting up sample data..."
# Create sample healthcare data
cat > sample-data/healthcare-data.csv << EOF
patient_id,age,gender,diagnosis,treatment_cost,length_of_stay
1,45,M,diabetes,1200,3
2,67,F,hypertension,800,2
3,34,M,asthma,600,1
4,56,F,diabetes,1400,4
5,78,M,heart_disease,3200,7
EOF

# Create sample metrics file
cat > sample-data/model-performance-metrics.json << EOF
{
  "model_name": "healthcare_predictor",
  "version": "1.0.0",
  "metrics": {
    "accuracy": 0.85,
    "precision": 0.82,
    "recall": 0.88,
    "f1_score": 0.85,
    "auc_roc": 0.91
  },
  "training_time": 1800,
  "inference_latency_ms": 45,
  "model_size_mb": 12.5
}
EOF

# Setup Jupyter kernel
print_status "Setting up Jupyter kernel..."
python -m ipykernel install --user --name=amazon-q-mlops-demo --display-name="Amazon Q MLOps Demo"

# Create environment variables file
print_status "Creating environment configuration..."
cat > .env << EOF
# Amazon Q CLI MLOps Demo Environment Variables
AWS_DEFAULT_REGION=us-east-1
SAGEMAKER_ROLE_ARN=arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole
S3_BUCKET=your-demo-bucket
DEMO_PREFIX=amazon-q-mlops-demo

# Cost optimization settings
COST_OPTIMIZATION_TARGET=60
SPOT_INSTANCE_ENABLED=true

# Performance monitoring
CLOUDWATCH_NAMESPACE=MLOps/Demo
ENABLE_DETAILED_MONITORING=true

# Demo settings
DEMO_MODE=true
VERBOSE_LOGGING=true
EOF

print_status "Environment setup completed successfully! 🎉"
print_status ""
print_status "Next steps:"
print_status "1. Activate the virtual environment: source venv/bin/activate"
print_status "2. Configure AWS credentials: aws configure"
print_status "3. Update .env file with your AWS account details"
print_status "4. Start Jupyter Lab: jupyter lab"
print_status "5. Open the demo notebooks in the notebooks/ directory"
print_status ""
print_status "For VSCode users:"
print_status "1. Install Amazon Q Developer extension"
print_status "2. Open this repository in VSCode"
print_status "3. Select the 'Amazon Q MLOps Demo' kernel for notebooks"
print_status ""
print_warning "Remember to update the SAGEMAKER_ROLE_ARN and S3_BUCKET in .env file!"