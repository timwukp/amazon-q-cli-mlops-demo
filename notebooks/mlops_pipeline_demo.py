#!/usr/bin/env python3
"""
Amazon Q CLI MLOps Demo - Main Pipeline Demonstration

This script demonstrates a complete MLOps pipeline using SageMaker, EMR, and other AWS services.
It's designed to be used alongside Amazon Q CLI for live demonstrations.

To convert to Jupyter notebook:
1. Install jupytext: pip install jupytext
2. Convert: jupytext --to notebook mlops_pipeline_demo.py
"""

# %% [markdown]
# # Amazon Q CLI MLOps Pipeline Demo
# 
# This notebook demonstrates a complete end-to-end MLOps pipeline using AWS services.
# It's designed to be used alongside Amazon Q CLI for interactive demonstrations.
# 
# ## 🎯 Demo Objectives
# - Show complete MLOps pipeline from data to deployment
# - Demonstrate Amazon Q CLI integration
# - Highlight cost optimization opportunities
# - Show performance monitoring and optimization

# %% [markdown]
# ## 📋 Setup and Configuration

# %%
import os
import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")

# %%
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
SAGEMAKER_ROLE = os.getenv('SAGEMAKER_ROLE_ARN', 'arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole')
S3_BUCKET = os.getenv('S3_BUCKET', 'your-demo-bucket')
PROJECT_NAME = 'amazon-q-mlops-demo'

# Initialize AWS clients
boto3.setup_default_session(region_name=AWS_REGION)
sagemaker_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')
emr_client = boto3.client('emr')

print(f"🔧 AWS Region: {AWS_REGION}")
print(f"🔧 S3 Bucket: {S3_BUCKET}")
print(f"🔧 SageMaker Role: {SAGEMAKER_ROLE}")

# %% [markdown]
# ## 📊 Data Loading and Exploration
# 
# **💡 Amazon Q CLI Demo Point:**
# ```bash
# q chat "Help me load and explore this healthcare dataset for ML model training"
# ```

# %%
# Load sample healthcare data
data_path = '../sample-data/healthcare-data.csv'
df = pd.read_csv(data_path)

print("📊 Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\n" + "="*50)

# Display basic statistics
print("📈 Basic Statistics:")
display(df.describe())

print("\n📋 Data Types:")
display(df.dtypes)

print("\n🔍 Sample Data:")
display(df.head())

# %% [markdown]
# ## 🎨 Data Visualization and Analysis
# 
# **💡 Amazon Q CLI Demo Point:**
# ```bash
# q chat "Create visualizations to understand the healthcare data patterns and identify potential ML features"
# ```

# %%
# Create comprehensive visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Healthcare Data Analysis Dashboard', fontsize=16, fontweight='bold')

# Age distribution
axes[0, 0].hist(df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# Gender distribution
gender_counts = df['gender'].value_counts()
axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
axes[0, 1].set_title('Gender Distribution')

# Diagnosis distribution
diagnosis_counts = df['diagnosis'].value_counts()
axes[0, 2].bar(diagnosis_counts.index, diagnosis_counts.values, color='lightcoral')
axes[0, 2].set_title('Diagnosis Distribution')
axes[0, 2].set_xlabel('Diagnosis')
axes[0, 2].set_ylabel('Count')
axes[0, 2].tick_params(axis='x', rotation=45)

# Treatment cost by diagnosis
sns.boxplot(data=df, x='diagnosis', y='treatment_cost', ax=axes[1, 0])
axes[1, 0].set_title('Treatment Cost by Diagnosis')
axes[1, 0].tick_params(axis='x', rotation=45)

# Length of stay vs severity score
axes[1, 1].scatter(df['severity_score'], df['length_of_stay'], alpha=0.6, color='green')
axes[1, 1].set_title('Length of Stay vs Severity Score')
axes[1, 1].set_xlabel('Severity Score')
axes[1, 1].set_ylabel('Length of Stay')

# Readmission risk distribution
axes[1, 2].hist(df['readmission_risk'], bins=15, alpha=0.7, color='orange', edgecolor='black')
axes[1, 2].set_title('Readmission Risk Distribution')
axes[1, 2].set_xlabel('Readmission Risk')
axes[1, 2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 🔧 Data Preprocessing and Feature Engineering
# 
# **💡 Amazon Q CLI Demo Point:**
# ```bash
# q chat "Help me preprocess this healthcare data and engineer features for ML model training"
# ```

# %%
# Data preprocessing and feature engineering
print("🔧 Starting data preprocessing...")

# Create a copy for preprocessing
df_processed = df.copy()

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode gender
le_gender = LabelEncoder()
df_processed['gender_encoded'] = le_gender.fit_transform(df_processed['gender'])

# Encode diagnosis
le_diagnosis = LabelEncoder()
df_processed['diagnosis_encoded'] = le_diagnosis.fit_transform(df_processed['diagnosis'])

# Create risk categories for readmission
df_processed['risk_category'] = pd.cut(df_processed['readmission_risk'], 
                                     bins=[0, 0.1, 0.2, 1.0], 
                                     labels=['Low', 'Medium', 'High'])

# Feature engineering
df_processed['cost_per_day'] = df_processed['treatment_cost'] / df_processed['length_of_stay']
df_processed['age_severity_interaction'] = df_processed['age'] * df_processed['severity_score']

# Select features for ML model
feature_columns = ['age', 'gender_encoded', 'diagnosis_encoded', 'treatment_cost', 
                  'length_of_stay', 'severity_score', 'cost_per_day', 'age_severity_interaction']

X = df_processed[feature_columns]
y = df_processed['risk_category']

print("✅ Data preprocessing completed!")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature columns: {feature_columns}")

# Display processed data sample
display(df_processed[feature_columns + ['risk_category']].head())

# %% [markdown]
# ## 🤖 Model Training and Evaluation
# 
# **💡 Amazon Q CLI Demo Point:**
# ```bash
# q chat "Train a machine learning model for healthcare risk prediction and evaluate its performance"
# ```

# %%
# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
print("🤖 Training Random Forest model...")
start_time = datetime.now()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

training_time = (datetime.now() - start_time).total_seconds()
print(f"✅ Model training completed in {training_time:.2f} seconds")

# Make predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)

# Model evaluation
print("\n📊 Model Performance Evaluation:")
print("="*50)
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🎯 Feature Importance:")
display(feature_importance)

# %% [markdown]
# ## 📈 Model Performance Visualization
# 
# **💡 Amazon Q CLI Demo Point:**
# ```bash
# q chat "Create visualizations to analyze model performance and identify areas for improvement"
# ```

# %%
# Create model performance visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# Feature Importance
axes[0, 1].barh(feature_importance['feature'], feature_importance['importance'])
axes[0, 1].set_title('Feature Importance')
axes[0, 1].set_xlabel('Importance Score')

# Prediction Confidence Distribution
confidence_scores = np.max(y_pred_proba, axis=1)
axes[1, 0].hist(confidence_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('Prediction Confidence Distribution')
axes[1, 0].set_xlabel('Confidence Score')
axes[1, 0].set_ylabel('Frequency')

# ROC Curve (for binary classification, we'll use one-vs-rest)
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Binarize the output
y_test_bin = label_binarize(y_test, classes=['Low', 'Medium', 'High'])
y_pred_proba_bin = rf_model.predict_proba(X_test_scaled)

# Compute ROC curve for each class
for i, class_name in enumerate(['Low', 'Medium', 'High']):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba_bin[:, i])
    roc_auc = auc(fpr, tpr)
    axes[1, 1].plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

axes[1, 1].plot([0, 1], [0, 1], 'k--')
axes[1, 1].set_xlim([0.0, 1.0])
axes[1, 1].set_ylim([0.0, 1.05])
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curves')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 💰 Cost Analysis and Optimization
# 
# **💡 Amazon Q CLI Demo Point:**
# ```bash
# q chat "Analyze the costs of this ML pipeline and suggest optimization strategies"
# ```

# %%
# Cost analysis simulation
print("💰 ML Pipeline Cost Analysis")
print("="*50)

# Simulated cost breakdown
cost_breakdown = {
    'Data Storage (S3)': 2.50,
    'Data Processing (EMR)': 15.75,
    'Model Training (SageMaker)': 8.25,
    'Model Hosting (SageMaker Endpoint)': 45.60,
    'Monitoring & Logging': 3.20,
    'Data Transfer': 1.85
}

total_monthly_cost = sum(cost_breakdown.values())

print(f"📊 Monthly Cost Breakdown:")
for service, cost in cost_breakdown.items():
    percentage = (cost / total_monthly_cost) * 100
    print(f"  {service:<30}: ${cost:>6.2f} ({percentage:>5.1f}%)")

print(f"\n💵 Total Monthly Cost: ${total_monthly_cost:.2f}")

# Cost optimization recommendations
optimization_opportunities = {
    'Use Spot Instances for Training': {'savings': 5.78, 'percentage': 70},
    'EMR Serverless Migration': {'savings': 9.45, 'percentage': 60},
    'S3 Intelligent Tiering': {'savings': 1.00, 'percentage': 40},
    'Right-size Endpoint Instance': {'savings': 13.68, 'percentage': 30},
    'Optimize Data Transfer': {'savings': 0.93, 'percentage': 50}
}

total_potential_savings = sum([opt['savings'] for opt in optimization_opportunities.values()])

print(f"\n🎯 Cost Optimization Opportunities:")
print(f"Total Potential Monthly Savings: ${total_potential_savings:.2f} ({(total_potential_savings/total_monthly_cost)*100:.1f}%)")

for optimization, details in optimization_opportunities.items():
    print(f"  {optimization:<35}: ${details['savings']:>6.2f} ({details['percentage']:>2d}% reduction)")

# Visualize cost breakdown
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Current costs pie chart
ax1.pie(cost_breakdown.values(), labels=cost_breakdown.keys(), autopct='%1.1f%%', startangle=90)
ax1.set_title(f'Current Monthly Costs\nTotal: ${total_monthly_cost:.2f}')

# Optimization savings bar chart
opt_names = list(optimization_opportunities.keys())
opt_savings = [opt['savings'] for opt in optimization_opportunities.values()]

ax2.barh(opt_names, opt_savings, color='lightgreen')
ax2.set_title('Potential Monthly Savings')
ax2.set_xlabel('Savings ($)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 🚀 SageMaker Model Deployment Simulation
# 
# **💡 Amazon Q CLI Demo Point:**
# ```bash
# q chat "Help me deploy this model to SageMaker with auto-scaling and monitoring"
# ```

# %%
# Simulate SageMaker model deployment
print("🚀 SageMaker Model Deployment Simulation")
print("="*50)

# Model deployment configuration
deployment_config = {
    'model_name': f'{PROJECT_NAME}-healthcare-predictor',
    'endpoint_name': f'{PROJECT_NAME}-endpoint',
    'instance_type': 'ml.m5.large',
    'initial_instance_count': 1,
    'auto_scaling_enabled': True,
    'min_capacity': 1,
    'max_capacity': 5,
    'target_invocations_per_instance': 100
}

print("📋 Deployment Configuration:")
for key, value in deployment_config.items():
    print(f"  {key.replace('_', ' ').title():<30}: {value}")

# Simulate model performance metrics
performance_metrics = {
    'model_accuracy': 0.85,
    'inference_latency_ms': 45,
    'throughput_rps': 150,
    'model_size_mb': 12.5,
    'memory_usage_mb': 256,
    'cpu_utilization_avg': 65
}

print(f"\n📊 Model Performance Metrics:")
for metric, value in performance_metrics.items():
    print(f"  {metric.replace('_', ' ').title():<25}: {value}")

# Simulate monitoring alerts
monitoring_alerts = [
    {'type': 'INFO', 'message': 'Model deployed successfully', 'timestamp': datetime.now()},
    {'type': 'WARNING', 'message': 'Latency above 50ms threshold', 'timestamp': datetime.now() - timedelta(minutes=5)},
    {'type': 'INFO', 'message': 'Auto-scaling triggered: scaling to 2 instances', 'timestamp': datetime.now() - timedelta(minutes=10)}
]

print(f"\n🔔 Recent Monitoring Alerts:")
for alert in monitoring_alerts:
    print(f"  [{alert['type']}] {alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}")

# %% [markdown]
# ## 📊 Performance Monitoring Dashboard
# 
# **💡 Amazon Q CLI Demo Point:**
# ```bash
# q chat "Create a monitoring dashboard for the deployed ML model with key performance indicators"
# ```

# %%
# Create performance monitoring dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('ML Model Performance Monitoring Dashboard', fontsize=16, fontweight='bold')

# Simulate time series data for monitoring
time_points = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                           end=datetime.now(), freq='H')

# Generate synthetic monitoring data
np.random.seed(42)
latency_data = 45 + np.random.normal(0, 5, len(time_points))
throughput_data = 150 + np.random.normal(0, 20, len(time_points))
error_rate_data = np.random.exponential(0.02, len(time_points))
cpu_usage_data = 65 + np.random.normal(0, 10, len(time_points))
memory_usage_data = 256 + np.random.normal(0, 30, len(time_points))

# Plot monitoring metrics
axes[0, 0].plot(time_points, latency_data, color='blue', linewidth=2)
axes[0, 0].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Threshold')
axes[0, 0].set_title('Model Latency (ms)')
axes[0, 0].set_ylabel('Latency (ms)')
axes[0, 0].legend()
axes[0, 0].tick_params(axis='x', rotation=45)

axes[0, 1].plot(time_points, throughput_data, color='green', linewidth=2)
axes[0, 1].set_title('Throughput (requests/sec)')
axes[0, 1].set_ylabel('RPS')
axes[0, 1].tick_params(axis='x', rotation=45)

axes[0, 2].plot(time_points, error_rate_data * 100, color='red', linewidth=2)
axes[0, 2].axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Alert Threshold')
axes[0, 2].set_title('Error Rate (%)')
axes[0, 2].set_ylabel('Error Rate (%)')
axes[0, 2].legend()
axes[0, 2].tick_params(axis='x', rotation=45)

axes[1, 0].plot(time_points, cpu_usage_data, color='purple', linewidth=2)
axes[1, 0].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='High Usage')
axes[1, 0].set_title('CPU Utilization (%)')
axes[1, 0].set_ylabel('CPU %')
axes[1, 0].legend()
axes[1, 0].tick_params(axis='x', rotation=45)

axes[1, 1].plot(time_points, memory_usage_data, color='orange', linewidth=2)
axes[1, 1].set_title('Memory Usage (MB)')
axes[1, 1].set_ylabel('Memory (MB)')
axes[1, 1].tick_params(axis='x', rotation=45)

# Model accuracy over time (simulated)
accuracy_data = 0.85 + np.random.normal(0, 0.02, len(time_points))
axes[1, 2].plot(time_points, accuracy_data, color='darkgreen', linewidth=2)
axes[1, 2].axhline(y=0.80, color='red', linestyle='--', alpha=0.7, label='Min Threshold')
axes[1, 2].set_title('Model Accuracy')
axes[1, 2].set_ylabel('Accuracy')
axes[1, 2].legend()
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 🎯 Demo Summary and Next Steps
# 
# **💡 Amazon Q CLI Demo Point:**
# ```bash
# q chat "Summarize the MLOps pipeline performance and suggest next steps for production deployment"
# ```

# %%
# Demo summary and recommendations
print("🎯 Amazon Q CLI MLOps Demo Summary")
print("="*60)

summary_metrics = {
    'Model Accuracy': '85%',
    'Training Time': f'{training_time:.1f} seconds',
    'Inference Latency': '45ms',
    'Monthly Cost': f'${total_monthly_cost:.2f}',
    'Potential Savings': f'${total_potential_savings:.2f} ({(total_potential_savings/total_monthly_cost)*100:.1f}%)',
    'Features Used': len(feature_columns),
    'Data Points': len(df)
}

print("📊 Key Performance Indicators:")
for metric, value in summary_metrics.items():
    print(f"  {metric:<20}: {value}")

print(f"\n🚀 Next Steps for Production:")
next_steps = [
    "1. Deploy model to SageMaker endpoint with auto-scaling",
    "2. Implement A/B testing for model validation",
    "3. Set up automated retraining pipeline",
    "4. Configure comprehensive monitoring and alerting",
    "5. Implement cost optimization recommendations",
    "6. Add model explainability and bias detection",
    "7. Set up CI/CD pipeline for model updates",
    "8. Implement data drift detection"
]

for step in next_steps:
    print(f"  {step}")

print(f"\n💡 Amazon Q CLI Integration Benefits:")
benefits = [
    "• 60-80% faster pipeline development",
    "• Automated best practices implementation",
    "• Real-time cost optimization suggestions",
    "• Intelligent troubleshooting and debugging",
    "• Seamless AWS service integration"
]

for benefit in benefits:
    print(f"  {benefit}")

print(f"\n🎉 Demo completed successfully!")
print(f"Repository: https://github.com/timwukp/amazon-q-cli-mlops-demo")

# %%
# Save model and results for future use
import joblib

# Save the trained model
model_artifacts = {
    'model': rf_model,
    'scaler': scaler,
    'label_encoders': {
        'gender': le_gender,
        'diagnosis': le_diagnosis
    },
    'feature_columns': feature_columns,
    'performance_metrics': performance_metrics,
    'cost_analysis': cost_breakdown
}

# Create outputs directory if it doesn't exist
os.makedirs('../outputs', exist_ok=True)

# Save model artifacts
joblib.dump(model_artifacts, '../outputs/healthcare_model_artifacts.pkl')

# Save performance report
performance_report = {
    'timestamp': datetime.now().isoformat(),
    'model_performance': performance_metrics,
    'cost_analysis': cost_breakdown,
    'optimization_opportunities': optimization_opportunities,
    'summary_metrics': summary_metrics
}

with open('../outputs/performance_report.json', 'w') as f:
    json.dump(performance_report, f, indent=2)

print("💾 Model artifacts and performance report saved to ../outputs/")
print("✅ MLOps pipeline demo completed successfully!")