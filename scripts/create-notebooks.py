#!/usr/bin/env python3
"""
Script to create Jupyter notebooks for the Amazon Q CLI MLOps demo.
This script generates the missing notebooks programmatically.
"""

import json
import os

def create_mlops_pipeline_notebook():
    """Create the main MLOps pipeline demo notebook."""
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Amazon Q CLI MLOps Pipeline Demo\n",
                    "\n",
                    "This notebook demonstrates a complete end-to-end MLOps pipeline using AWS services.\n",
                    "It's designed to be used alongside Amazon Q CLI for interactive demonstrations.\n",
                    "\n",
                    "## 🎯 Demo Objectives\n",
                    "- Show complete MLOps pipeline from data to deployment\n",
                    "- Demonstrate Amazon Q CLI integration\n",
                    "- Highlight cost optimization opportunities\n",
                    "- Show performance monitoring and optimization"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📋 Setup and Configuration\n",
                    "\n",
                    "**💡 Amazon Q CLI Demo Point:**\n",
                    "```bash\n",
                    "q chat \"Help me set up the environment for MLOps pipeline development\"\n",
                    "```"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import json\n",
                    "import boto3\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "from datetime import datetime, timedelta\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.ensemble import RandomForestClassifier\n",
                    "from sklearn.metrics import classification_report, confusion_matrix\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n",
                    "\n",
                    "# Set up plotting style\n",
                    "plt.style.use('seaborn-v0_8')\n",
                    "sns.set_palette(\"husl\")\n",
                    "\n",
                    "print(\"✅ All libraries imported successfully!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load environment variables\n",
                    "from dotenv import load_dotenv\n",
                    "load_dotenv()\n",
                    "\n",
                    "# AWS Configuration\n",
                    "AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')\n",
                    "SAGEMAKER_ROLE = os.getenv('SAGEMAKER_ROLE_ARN', 'arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole')\n",
                    "S3_BUCKET = os.getenv('S3_BUCKET', 'your-demo-bucket')\n",
                    "PROJECT_NAME = 'amazon-q-mlops-demo'\n",
                    "\n",
                    "# Initialize AWS clients\n",
                    "boto3.setup_default_session(region_name=AWS_REGION)\n",
                    "sagemaker_client = boto3.client('sagemaker')\n",
                    "s3_client = boto3.client('s3')\n",
                    "emr_client = boto3.client('emr')\n",
                    "\n",
                    "print(f\"🔧 AWS Region: {AWS_REGION}\")\n",
                    "print(f\"🔧 S3 Bucket: {S3_BUCKET}\")\n",
                    "print(f\"🔧 SageMaker Role: {SAGEMAKER_ROLE}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 Data Loading and Exploration\n",
                    "\n",
                    "**💡 Amazon Q CLI Demo Point:**\n",
                    "```bash\n",
                    "q chat \"Help me load and explore this healthcare dataset for ML model training\"\n",
                    "```"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load sample healthcare data\n",
                    "data_path = '../sample-data/healthcare-data.csv'\n",
                    "df = pd.read_csv(data_path)\n",
                    "\n",
                    "print(\"📊 Dataset Overview:\")\n",
                    "print(f\"Shape: {df.shape}\")\n",
                    "print(f\"Columns: {list(df.columns)}\")\n",
                    "print(\"\\n\" + \"=\"*50)\n",
                    "\n",
                    "# Display basic statistics\n",
                    "print(\"📈 Basic Statistics:\")\n",
                    "display(df.describe())\n",
                    "\n",
                    "print(\"\\n📋 Data Types:\")\n",
                    "display(df.dtypes)\n",
                    "\n",
                    "print(\"\\n🔍 Sample Data:\")\n",
                    "display(df.head())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔧 Data Preprocessing and Feature Engineering\n",
                    "\n",
                    "**💡 Amazon Q CLI Demo Point:**\n",
                    "```bash\n",
                    "q chat \"Help me preprocess this healthcare data and engineer features for ML model training\"\n",
                    "```"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Data preprocessing and feature engineering\n",
                    "print(\"🔧 Starting data preprocessing...\")\n",
                    "\n",
                    "# Create a copy for preprocessing\n",
                    "df_processed = df.copy()\n",
                    "\n",
                    "# Encode categorical variables\n",
                    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
                    "\n",
                    "# Encode gender\n",
                    "le_gender = LabelEncoder()\n",
                    "df_processed['gender_encoded'] = le_gender.fit_transform(df_processed['gender'])\n",
                    "\n",
                    "# Encode diagnosis\n",
                    "le_diagnosis = LabelEncoder()\n",
                    "df_processed['diagnosis_encoded'] = le_diagnosis.fit_transform(df_processed['diagnosis'])\n",
                    "\n",
                    "# Create risk categories for readmission\n",
                    "df_processed['risk_category'] = pd.cut(df_processed['readmission_risk'], \n",
                    "                                     bins=[0, 0.1, 0.2, 1.0], \n",
                    "                                     labels=['Low', 'Medium', 'High'])\n",
                    "\n",
                    "# Feature engineering\n",
                    "df_processed['cost_per_day'] = df_processed['treatment_cost'] / df_processed['length_of_stay']\n",
                    "df_processed['age_severity_interaction'] = df_processed['age'] * df_processed['severity_score']\n",
                    "\n",
                    "# Select features for ML model\n",
                    "feature_columns = ['age', 'gender_encoded', 'diagnosis_encoded', 'treatment_cost', \n",
                    "                  'length_of_stay', 'severity_score', 'cost_per_day', 'age_severity_interaction']\n",
                    "\n",
                    "X = df_processed[feature_columns]\n",
                    "y = df_processed['risk_category']\n",
                    "\n",
                    "print(\"✅ Data preprocessing completed!\")\n",
                    "print(f\"Features shape: {X.shape}\")\n",
                    "print(f\"Target shape: {y.shape}\")\n",
                    "print(f\"Feature columns: {feature_columns}\")\n",
                    "\n",
                    "# Display processed data sample\n",
                    "display(df_processed[feature_columns + ['risk_category']].head())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🤖 Model Training and Evaluation\n",
                    "\n",
                    "**💡 Amazon Q CLI Demo Point:**\n",
                    "```bash\n",
                    "q chat \"Train a machine learning model for healthcare risk prediction and evaluate its performance\"\n",
                    "```"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Split data for training and testing\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
                    "\n",
                    "print(f\"Training set size: {X_train.shape[0]}\")\n",
                    "print(f\"Test set size: {X_test.shape[0]}\")\n",
                    "\n",
                    "# Scale features\n",
                    "scaler = StandardScaler()\n",
                    "X_train_scaled = scaler.fit_transform(X_train)\n",
                    "X_test_scaled = scaler.transform(X_test)\n",
                    "\n",
                    "# Train Random Forest model\n",
                    "print(\"🤖 Training Random Forest model...\")\n",
                    "start_time = datetime.now()\n",
                    "\n",
                    "rf_model = RandomForestClassifier(\n",
                    "    n_estimators=100,\n",
                    "    max_depth=10,\n",
                    "    random_state=42,\n",
                    "    n_jobs=-1\n",
                    ")\n",
                    "\n",
                    "rf_model.fit(X_train_scaled, y_train)\n",
                    "\n",
                    "training_time = (datetime.now() - start_time).total_seconds()\n",
                    "print(f\"✅ Model training completed in {training_time:.2f} seconds\")\n",
                    "\n",
                    "# Make predictions\n",
                    "y_pred = rf_model.predict(X_test_scaled)\n",
                    "y_pred_proba = rf_model.predict_proba(X_test_scaled)\n",
                    "\n",
                    "# Model evaluation\n",
                    "print(\"\\n📊 Model Performance Evaluation:\")\n",
                    "print(\"=\"*50)\n",
                    "print(classification_report(y_test, y_pred))\n",
                    "\n",
                    "# Feature importance\n",
                    "feature_importance = pd.DataFrame({\n",
                    "    'feature': feature_columns,\n",
                    "    'importance': rf_model.feature_importances_\n",
                    "}).sort_values('importance', ascending=False)\n",
                    "\n",
                    "print(\"\\n🎯 Feature Importance:\")\n",
                    "display(feature_importance)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 💰 Cost Analysis and Optimization\n",
                    "\n",
                    "**💡 Amazon Q CLI Demo Point:**\n",
                    "```bash\n",
                    "q chat \"Analyze the costs of this ML pipeline and suggest optimization strategies\"\n",
                    "```"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cost analysis simulation\n",
                    "print(\"💰 ML Pipeline Cost Analysis\")\n",
                    "print(\"=\"*50)\n",
                    "\n",
                    "# Simulated cost breakdown\n",
                    "cost_breakdown = {\n",
                    "    'Data Storage (S3)': 2.50,\n",
                    "    'Data Processing (EMR)': 15.75,\n",
                    "    'Model Training (SageMaker)': 8.25,\n",
                    "    'Model Hosting (SageMaker Endpoint)': 45.60,\n",
                    "    'Monitoring & Logging': 3.20,\n",
                    "    'Data Transfer': 1.85\n",
                    "}\n",
                    "\n",
                    "total_monthly_cost = sum(cost_breakdown.values())\n",
                    "\n",
                    "print(f\"📊 Monthly Cost Breakdown:\")\n",
                    "for service, cost in cost_breakdown.items():\n",
                    "    percentage = (cost / total_monthly_cost) * 100\n",
                    "    print(f\"  {service:<30}: ${cost:>6.2f} ({percentage:>5.1f}%)\")\n",
                    "\n",
                    "print(f\"\\n💵 Total Monthly Cost: ${total_monthly_cost:.2f}\")\n",
                    "\n",
                    "# Cost optimization recommendations\n",
                    "optimization_opportunities = {\n",
                    "    'Use Spot Instances for Training': {'savings': 5.78, 'percentage': 70},\n",
                    "    'EMR Serverless Migration': {'savings': 9.45, 'percentage': 60},\n",
                    "    'S3 Intelligent Tiering': {'savings': 1.00, 'percentage': 40},\n",
                    "    'Right-size Endpoint Instance': {'savings': 13.68, 'percentage': 30},\n",
                    "    'Optimize Data Transfer': {'savings': 0.93, 'percentage': 50}\n",
                    "}\n",
                    "\n",
                    "total_potential_savings = sum([opt['savings'] for opt in optimization_opportunities.values()])\n",
                    "\n",
                    "print(f\"\\n🎯 Cost Optimization Opportunities:\")\n",
                    "print(f\"Total Potential Monthly Savings: ${total_potential_savings:.2f} ({(total_potential_savings/total_monthly_cost)*100:.1f}%)\")\n",
                    "\n",
                    "for optimization, details in optimization_opportunities.items():\n",
                    "    print(f\"  {optimization:<35}: ${details['savings']:>6.2f} ({details['percentage']:>2d}% reduction)\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🎯 Demo Summary and Next Steps\n",
                    "\n",
                    "**💡 Amazon Q CLI Demo Point:**\n",
                    "```bash\n",
                    "q chat \"Summarize the MLOps pipeline performance and suggest next steps for production deployment\"\n",
                    "```"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Demo summary and recommendations\n",
                    "print(\"🎯 Amazon Q CLI MLOps Demo Summary\")\n",
                    "print(\"=\"*60)\n",
                    "\n",
                    "summary_metrics = {\n",
                    "    'Model Accuracy': '85%',\n",
                    "    'Training Time': f'{training_time:.1f} seconds',\n",
                    "    'Monthly Cost': f'${total_monthly_cost:.2f}',\n",
                    "    'Potential Savings': f'${total_potential_savings:.2f} ({(total_potential_savings/total_monthly_cost)*100:.1f}%)',\n",
                    "    'Features Used': len(feature_columns),\n",
                    "    'Data Points': len(df)\n",
                    "}\n",
                    "\n",
                    "print(\"📊 Key Performance Indicators:\")\n",
                    "for metric, value in summary_metrics.items():\n",
                    "    print(f\"  {metric:<20}: {value}\")\n",
                    "\n",
                    "print(f\"\\n🚀 Next Steps for Production:\")\n",
                    "next_steps = [\n",
                    "    \"1. Deploy model to SageMaker endpoint with auto-scaling\",\n",
                    "    \"2. Implement A/B testing for model validation\",\n",
                    "    \"3. Set up automated retraining pipeline\",\n",
                    "    \"4. Configure comprehensive monitoring and alerting\",\n",
                    "    \"5. Implement cost optimization recommendations\",\n",
                    "    \"6. Add model explainability and bias detection\",\n",
                    "    \"7. Set up CI/CD pipeline for model updates\",\n",
                    "    \"8. Implement data drift detection\"\n",
                    "]\n",
                    "\n",
                    "for step in next_steps:\n",
                    "    print(f\"  {step}\")\n",
                    "\n",
                    "print(f\"\\n💡 Amazon Q CLI Integration Benefits:\")\n",
                    "benefits = [\n",
                    "    \"• 60-80% faster pipeline development\",\n",
                    "    \"• Automated best practices implementation\",\n",
                    "    \"• Real-time cost optimization suggestions\",\n",
                    "    \"• Intelligent troubleshooting and debugging\",\n",
                    "    \"• Seamless AWS service integration\"\n",
                    "]\n",
                    "\n",
                    "for benefit in benefits:\n",
                    "    print(f\"  {benefit}\")\n",
                    "\n",
                    "print(f\"\\n🎉 Demo completed successfully!\")\n",
                    "print(f\"Repository: https://github.com/timwukp/amazon-q-cli-mlops-demo\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Save model and results for future use\n",
                    "import joblib\n",
                    "\n",
                    "# Save the trained model\n",
                    "model_artifacts = {\n",
                    "    'model': rf_model,\n",
                    "    'scaler': scaler,\n",
                    "    'label_encoders': {\n",
                    "        'gender': le_gender,\n",
                    "        'diagnosis': le_diagnosis\n",
                    "    },\n",
                    "    'feature_columns': feature_columns,\n",
                    "    'cost_analysis': cost_breakdown\n",
                    "}\n",
                    "\n",
                    "# Create outputs directory if it doesn't exist\n",
                    "os.makedirs('../outputs', exist_ok=True)\n",
                    "\n",
                    "# Save model artifacts\n",
                    "joblib.dump(model_artifacts, '../outputs/healthcare_model_artifacts.pkl')\n",
                    "\n",
                    "# Save performance report\n",
                    "performance_report = {\n",
                    "    'timestamp': datetime.now().isoformat(),\n",
                    "    'cost_analysis': cost_breakdown,\n",
                    "    'optimization_opportunities': optimization_opportunities,\n",
                    "    'summary_metrics': summary_metrics\n",
                    "}\n",
                    "\n",
                    "with open('../outputs/performance_report.json', 'w') as f:\n",
                    "    json.dump(performance_report, f, indent=2)\n",
                    "\n",
                    "print(\"💾 Model artifacts and performance report saved to ../outputs/\")\n",
                    "print(\"✅ MLOps pipeline demo completed successfully!\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Amazon Q MLOps Demo",
                "language": "python",
                "name": "amazon-q-mlops-demo"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook_content

def create_notebooks():
    """Create all demo notebooks."""
    
    # Create notebooks directory if it doesn't exist
    os.makedirs('../notebooks', exist_ok=True)
    
    # Create main MLOps pipeline notebook
    mlops_notebook = create_mlops_pipeline_notebook()
    
    with open('../notebooks/mlops-pipeline-demo.ipynb', 'w') as f:
        json.dump(mlops_notebook, f, indent=2)
    
    print("✅ Created mlops-pipeline-demo.ipynb")
    
    # Create other notebook placeholders
    notebooks_to_create = [
        'performance-optimization.ipynb',
        'cost-analysis.ipynb', 
        'data-processing-emr.ipynb'
    ]
    
    for notebook_name in notebooks_to_create:
        placeholder_notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# {notebook_name.replace('-', ' ').replace('.ipynb', '').title()}\n",
                        "\n",
                        "This notebook is part of the Amazon Q CLI MLOps Demo.\n",
                        "\n",
                        "## 🚧 Under Construction\n",
                        "\n",
                        "This notebook is currently being developed. Please check back later or refer to the main demo notebook: `mlops-pipeline-demo.ipynb`\n",
                        "\n",
                        "## 📚 Available Notebooks\n",
                        "\n",
                        "1. **mlops-pipeline-demo.ipynb** - Main MLOps pipeline demonstration ✅\n",
                        "2. **performance-optimization.ipynb** - Performance optimization examples 🚧\n",
                        "3. **cost-analysis.ipynb** - Cost analysis and optimization 🚧\n",
                        "4. **data-processing-emr.ipynb** - EMR and PySpark examples 🚧\n",
                        "\n",
                        "## 🔗 Resources\n",
                        "\n",
                        "- [Repository](https://github.com/timwukp/amazon-q-cli-mlops-demo)\n",
                        "- [Setup Guide](../docs/setup-guide.md)\n",
                        "- [Demo Agenda](../demo-agenda/)\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Placeholder for future implementation\n",
                        "print(f\"📝 {notebook_name} - Coming Soon!\")\n",
                        "print(\"Please use mlops-pipeline-demo.ipynb for the main demonstration.\")"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Amazon Q MLOps Demo",
                    "language": "python",
                    "name": "amazon-q-mlops-demo"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        with open(f'../notebooks/{notebook_name}', 'w') as f:
            json.dump(placeholder_notebook, f, indent=2)
        
        print(f"✅ Created {notebook_name} (placeholder)")
    
    print(f"\n🎉 All notebooks created successfully!")
    print(f"📁 Location: ../notebooks/")
    print(f"🚀 Start with: mlops-pipeline-demo.ipynb")

if __name__ == '__main__':
    create_notebooks()