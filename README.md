# Amazon Q CLI & Q Developer IDE Extension MLOps Demo

## 🎯 Overview

This repository contains comprehensive materials for a **60-minute live demonstration** of Amazon Q CLI and Q Developer IDE Extension, specifically designed for **AI/MLOps teams** using SageMaker, EMR, and other AWS AI services.

## 🚀 Demo Highlights

- **SageMaker Pipeline & MLOps Workflows** - Automated pipeline creation and monitoring
- **MLOps Review & Performance Optimization** - Live performance analysis and optimization
- **Cost Optimization** - Real-time cost analysis and savings recommendations  
- **EMR & PySpark Integration** - Serverless data processing optimization
- **Amazon Q Developer IDE Extension** - Intelligent code assistance and GitHub integration

## 📋 Demo Agenda (60 minutes)

### Part 1: SageMaker Pipeline & MLOps Workflows (10 min)
- Pipeline creation and monitoring
- Model development and deployment
- Troubleshooting and debugging

### Part 2: MLOps Review & Performance Optimization (18 min)
- MLOps maturity assessment
- Model performance optimization
- Data pipeline performance tuning
- Infrastructure optimization

### Part 3: Cost Optimization Live Demo (12 min)
- SageMaker cost analysis
- EMR cost optimization
- Storage and data transfer optimization

### Part 4: Amazon Q Developer IDE Extension (15 min)
- VSCode integration and setup
- Live coding with intelligent assistance
- GitHub integration and collaboration

### Part 5: Advanced Integration (5 min)
- Comprehensive monitoring setup
- Automated optimization workflows

## 📁 Repository Structure

```
├── README.md                          # This file
├── demo-agenda/                       # Detailed demo scripts
│   ├── 01-sagemaker-mlops.md         # SageMaker MLOps workflows
│   ├── 02-performance-optimization.md # Performance optimization guide
│   ├── 03-cost-optimization.md       # Cost optimization strategies
│   ├── 04-q-developer-ide.md         # Q Developer IDE extension demo
│   └── 05-advanced-integration.md    # Advanced scenarios
├── notebooks/                         # Sample Jupyter notebooks
│   ├── mlops-pipeline-demo.ipynb     # Main MLOps pipeline demo
│   ├── performance-optimization.ipynb # Performance optimization examples
│   ├── cost-analysis.ipynb           # Cost analysis and optimization
│   └── data-processing-emr.ipynb     # EMR/PySpark examples
├── scripts/                           # Demo scripts and utilities
│   ├── setup-demo-environment.sh     # Environment setup
│   ├── q-cli-commands.sh             # Sample Q CLI commands
│   └── cost-optimization-tools.py    # Cost optimization utilities
├── sample-data/                       # Sample datasets for demo
│   ├── healthcare-data.csv           # Healthcare sample data
│   └── model-performance-metrics.json # Sample metrics
├── infrastructure/                    # Infrastructure as Code
│   ├── cloudformation/               # CloudFormation templates
│   └── cdk/                          # AWS CDK examples
└── docs/                             # Additional documentation
    ├── setup-guide.md                # Setup instructions
    ├── troubleshooting.md            # Common issues and solutions
    └── best-practices.md             # MLOps best practices
```

## 🛠️ Prerequisites

### Software Requirements
- **Amazon Q CLI** - Latest version installed
- **VSCode** with Amazon Q Developer extension
- **AWS CLI** - Configured with appropriate permissions
- **Python 3.8+** with Jupyter notebook support
- **Git** for repository management

### AWS Services Access
- Amazon SageMaker (training, endpoints, pipelines)
- Amazon EMR (clusters, serverless)
- Amazon S3 (data storage)
- AWS Cost Explorer and Billing
- CloudWatch (monitoring and logs)

### Permissions Required
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:*",
        "emr:*",
        "s3:*",
        "ce:*",
        "cloudwatch:*",
        "iam:PassRole"
      ],
      "Resource": "*"
    }
  ]
}
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/timwukp/amazon-q-cli-mlops-demo.git
cd amazon-q-cli-mlops-demo
```

### 2. Setup Environment
```bash
# Run setup script
chmod +x scripts/setup-demo-environment.sh
./scripts/setup-demo-environment.sh

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Configure Amazon Q CLI
```bash
# Install Amazon Q CLI (if not already installed)
curl -sSL https://d2eo22ngex1n9g.cloudfront.net/Documentation/SDK/bedrock-python-sdk.tar.gz | tar -xz
pip install amazon-q-cli

# Configure Q CLI
q configure
```

### 4. Setup VSCode Extension
1. Install Amazon Q Developer extension from VSCode marketplace
2. Sign in with your AWS credentials
3. Configure GitHub integration

## 📊 Demo Scenarios

### Scenario 1: MLOps Pipeline Review
- Assess current MLOps maturity
- Identify performance bottlenecks
- Implement optimization recommendations

### Scenario 2: Cost Optimization Sprint
- Analyze current AWS spending
- Implement immediate cost savings
- Setup automated cost monitoring

### Scenario 3: Performance Tuning Workshop
- Profile model inference performance
- Optimize data processing pipelines
- Implement caching and parallelization

## 🎯 Expected Outcomes

### Performance Improvements
- **30-50%** reduction in model inference latency
- **20-40%** improvement in training job completion time
- **25-45%** increase in data processing throughput

### Cost Savings
- **20-40%** reduction in SageMaker training costs
- **15-30%** EMR cost savings through optimization
- **10-25%** S3 storage cost reduction

### MLOps Maturity
- Automated monitoring and alerting
- Continuous performance optimization
- Standardized deployment processes

## 🔧 Troubleshooting

### Common Issues
1. **Q CLI Authentication** - Ensure AWS credentials are properly configured
2. **Permission Errors** - Verify IAM roles have required permissions
3. **Resource Limits** - Check AWS service quotas and limits

### Support Resources
- [Amazon Q CLI Documentation](https://docs.aws.amazon.com/amazonq/)
- [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- [EMR Best Practices](https://docs.aws.amazon.com/emr/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Tim Wu** - *Sr. AI/ML Specialist Solutions Architect, AWS WWSO*
- LinkedIn: [tim-wu-7865a7b0](https://www.linkedin.com/in/tim-wu-7865a7b0/)

## 🏷️ Tags

`amazon-q` `mlops` `sagemaker` `emr` `cost-optimization` `performance-optimization` `ai` `machine-learning` `aws` `demo`

---

**Ready to transform your MLOps workflow with Amazon Q?** 🚀

Start with the [Setup Guide](docs/setup-guide.md) and follow the [Demo Agenda](demo-agenda/) for a comprehensive experience!