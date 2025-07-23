#!/bin/bash

# Amazon Q CLI Demo Commands
# This script contains all the Q CLI commands used in the live demo

echo "🤖 Amazon Q CLI MLOps Demo Commands"
echo "=================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

demo_command() {
    echo -e "\n${BLUE}Demo Command:${NC} $1"
    echo -e "${YELLOW}Expected Response:${NC} $2"
    echo "----------------------------------------"
}

echo -e "\n${GREEN}PART 1: SAGEMAKER PIPELINE & MLOPS WORKFLOWS${NC}"

demo_command \
"q chat 'Help me create a SageMaker pipeline for model training with preprocessing, training, and evaluation steps'" \
"Step-by-step pipeline definition with code generation and best practices"

demo_command \
"q chat 'Show me how to monitor my SageMaker pipeline execution and debug failed steps'" \
"CloudWatch integration setup and debugging strategies"

demo_command \
"q chat 'Write a Python script for a classification model using scikit-learn that can be deployed to SageMaker'" \
"Complete model training script with SageMaker-compatible structure"

demo_command \
"q chat 'Help me deploy a model to SageMaker endpoint with auto-scaling configuration'" \
"Endpoint configuration code with auto-scaling and monitoring setup"

echo -e "\n${GREEN}PART 2: MLOPS REVIEW & PERFORMANCE OPTIMIZATION${NC}"

demo_command \
"q chat 'Review my current MLOps pipeline and assess its maturity level against industry best practices'" \
"MLOps maturity scoring with gap analysis and improvement recommendations"

demo_command \
"q chat 'Analyze my SageMaker pipeline execution logs and identify performance bottlenecks'" \
"Execution time analysis with bottleneck identification and optimization strategies"

demo_command \
"q chat 'My SageMaker endpoint has high latency. Analyze the model and suggest optimization strategies'" \
"Latency analysis with model optimization techniques and infrastructure recommendations"

demo_command \
"q chat 'My PySpark job on EMR is running slowly. Analyze the code and suggest performance improvements'" \
"Spark job analysis with resource allocation optimization and code improvements"

demo_command \
"q chat 'Optimize my data preprocessing pipeline to reduce training time and improve data quality'" \
"Preprocessing pipeline analysis with parallelization opportunities and quality improvements"

echo -e "\n${GREEN}PART 3: COST OPTIMIZATION LIVE DEMO${NC}"

demo_command \
"q chat 'Analyze my SageMaker costs for the last 30 days and identify the top cost drivers'" \
"Detailed cost breakdown with trend analysis and optimization opportunities"

demo_command \
"q chat 'I'm running ml.m5.2xlarge instances for training. What are more cost-effective alternatives for my workload?'" \
"Instance type comparison matrix with performance vs cost analysis"

demo_command \
"q chat 'Help me configure SageMaker training jobs to use Spot instances and save costs'" \
"Spot instance setup code with interruption handling and cost savings calculation"

demo_command \
"q chat 'Show me how to optimize EMR costs by using Spot instances and right-sizing clusters'" \
"EMR cost breakdown with spot instance configuration and cluster optimization"

demo_command \
"q chat 'Compare costs between EMR Serverless and EMR on EC2 for my PySpark workloads running 4 hours daily'" \
"Detailed cost comparison with usage pattern analysis and migration recommendations"

demo_command \
"q chat 'Analyze my S3 storage costs for ML datasets and recommend lifecycle policies to reduce expenses'" \
"Storage class analysis with lifecycle policy recommendations and cost projections"

echo -e "\n${GREEN}PART 4: AMAZON Q DEVELOPER IDE EXTENSION${NC}"

demo_command \
"q chat 'Help me clone the aws-samples/amazon-sagemaker-architecting-for-ml-hcls repository and set up the development environment'" \
"Git clone commands with environment setup and dependency installation guidance"

echo "VSCode Q Developer Actions:"
echo "- Explain this SageMaker MLOps pipeline and suggest cost-optimized improvements"
echo "- Generate a cost-optimized SageMaker pipeline using Spot instances and efficient resource allocation"
echo "- Enhance this pipeline with MLOps best practices for monitoring and observability"
echo "- Generate CloudFormation template for this SageMaker pipeline with proper IAM roles"

echo -e "\n${GREEN}PART 5: ADVANCED INTEGRATION${NC}"

demo_command \
"q chat 'Set up comprehensive monitoring for my ML pipeline including data quality, model performance, and infrastructure metrics'" \
"Multi-layer monitoring architecture with automated alerting configuration"

demo_command \
"q chat 'Create automated performance tuning for my SageMaker hyperparameter optimization jobs'" \
"Automated hyperparameter tuning setup with performance-based optimization"

demo_command \
"q chat 'Design a multi-account MLOps architecture with centralized governance and distributed execution'" \
"Multi-account architecture design with cross-account IAM and centralized monitoring"

echo -e "\n${GREEN}ADDITIONAL DEMO COMMANDS${NC}"

demo_command \
"q chat 'Calculate potential savings by switching from on-demand to spot instances for my training workload'" \
"Cost savings calculation with detailed comparison and implementation guidance"

demo_command \
"q chat 'Find and help me clean up unused SageMaker endpoints and EMR clusters'" \
"Resource cleanup script with cost impact analysis"

demo_command \
"q chat 'Set up budget alerts for my ML workloads to prevent cost overruns'" \
"AWS Budgets configuration with CloudWatch alarms and automated responses"

demo_command \
"q chat 'Implement the top 5 cost optimization recommendations for my ML infrastructure'" \
"Prioritized optimization implementation with step-by-step guidance"

demo_command \
"q chat 'Create a predictive cost model for my ML workloads based on usage patterns'" \
"Predictive modeling setup with historical data analysis and forecasting"

echo -e "\n${GREEN}TROUBLESHOOTING COMMANDS${NC}"

demo_command \
"q chat 'Diagnose why my SageMaker training job is running 3x slower than expected'" \
"Performance diagnosis with bottleneck identification and resolution strategies"

demo_command \
"q chat 'My EMR cluster is underutilized. How can I optimize resource allocation?'" \
"Resource utilization analysis with optimization recommendations"

demo_command \
"q chat 'Find the optimal balance between performance and cost for my ML workload'" \
"Cost-performance trade-off analysis with optimization recommendations"

echo -e "\n${GREEN}DEMO PREPARATION COMMANDS${NC}"

echo "Pre-demo setup commands:"
echo "q chat 'Set up cost monitoring dashboard for my ML workloads'"
echo "q chat 'Create cost allocation tags for better expense tracking'"
echo "q chat 'Configure billing alerts for ML service spending thresholds'"

echo -e "\n${YELLOW}Note: Replace 'q chat' with the actual Amazon Q CLI command syntax when available${NC}"
echo -e "${YELLOW}These commands are examples for demonstration purposes${NC}"