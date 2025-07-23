# Setup Guide for Amazon Q CLI MLOps Demo

## 🎯 Overview

This guide will help you set up the complete environment for the Amazon Q CLI MLOps demo, including all necessary tools, dependencies, and configurations.

## 📋 Prerequisites

### System Requirements
- **Operating System**: macOS 10.15+ or Linux (Ubuntu 18.04+)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space
- **Network**: Stable internet connection for AWS services

### Required Accounts
- **AWS Account** with appropriate permissions
- **GitHub Account** for repository access
- **VSCode** or preferred IDE

## 🛠️ Installation Steps

### Step 1: Clone the Repository
```bash
git clone https://github.com/timwukp/amazon-q-cli-mlops-demo.git
cd amazon-q-cli-mlops-demo
```

### Step 2: Run Setup Script
```bash
chmod +x scripts/setup-demo-environment.sh
./scripts/setup-demo-environment.sh
```

### Step 3: Activate Virtual Environment
```bash
source venv/bin/activate
```

### Step 4: Configure AWS Credentials
```bash
aws configure
```

Enter your AWS credentials:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., us-east-1)
- Default output format (json)

### Step 5: Update Environment Variables
Edit the `.env` file with your specific AWS details:
```bash
# Update these values with your AWS account details
AWS_DEFAULT_REGION=us-east-1
SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole
S3_BUCKET=your-unique-demo-bucket
```

### Step 6: Install Amazon Q CLI
```bash
# Note: Replace with actual installation command when available
# This is a placeholder for the actual Q CLI installation
pip install amazon-q-cli  # Placeholder command
```

### Step 7: Setup VSCode Extension
1. Open VSCode
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "Amazon Q Developer"
4. Install the extension
5. Sign in with your AWS credentials

## 🔧 AWS Services Setup

### SageMaker Execution Role
Create a SageMaker execution role with the following policies:
- AmazonSageMakerFullAccess
- AmazonS3FullAccess
- CloudWatchFullAccess

```bash
# Create SageMaker execution role (using AWS CLI)
aws iam create-role --role-name SageMakerExecutionRole --assume-role-policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}'

# Attach policies
aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### S3 Bucket Setup
```bash
# Create S3 bucket for demo data
aws s3 mb s3://your-unique-demo-bucket --region us-east-1

# Upload sample data
aws s3 cp sample-data/ s3://your-unique-demo-bucket/data/ --recursive
```

### EMR Setup (Optional)
If you plan to use EMR features:
```bash
# Create EMR service role
aws emr create-default-roles
```

## 📊 Verification Steps

### Test AWS Connection
```bash
aws sts get-caller-identity
```

### Test SageMaker Access
```bash
aws sagemaker list-training-jobs --max-items 5
```

### Test S3 Access
```bash
aws s3 ls s3://your-unique-demo-bucket/
```

### Test Python Environment
```bash
python -c "import sagemaker, boto3, pandas, numpy; print('All packages imported successfully')"
```

## 🚀 Launch Demo Environment

### Start Jupyter Lab
```bash
jupyter lab
```

### Open VSCode
```bash
code .
```

### Verify Q CLI Installation
```bash
q --version  # Placeholder command
```

## 🔍 Troubleshooting

### Common Issues

#### 1. AWS Credentials Not Found
**Error**: `Unable to locate credentials`
**Solution**: 
```bash
aws configure
# Or set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

#### 2. Permission Denied Errors
**Error**: `Access Denied` when accessing AWS services
**Solution**: Verify IAM permissions and role attachments

#### 3. Python Package Conflicts
**Error**: Package version conflicts
**Solution**: 
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

#### 4. SageMaker Role Issues
**Error**: `Cannot assume role`
**Solution**: Verify the SageMaker execution role exists and has correct trust policy

#### 5. S3 Bucket Access Issues
**Error**: `Bucket does not exist` or `Access Denied`
**Solution**: 
```bash
# Check bucket exists
aws s3 ls s3://your-bucket-name/
# Verify bucket policy and permissions
```

### Getting Help

#### AWS Support Resources
- [AWS Documentation](https://docs.aws.amazon.com/)
- [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- [AWS CLI User Guide](https://docs.aws.amazon.com/cli/)

#### Community Resources
- [AWS Forums](https://forums.aws.amazon.com/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/amazon-web-services)
- [GitHub Issues](https://github.com/timwukp/amazon-q-cli-mlops-demo/issues)

## 📝 Environment Validation Checklist

- [ ] Repository cloned successfully
- [ ] Virtual environment activated
- [ ] Python packages installed
- [ ] AWS credentials configured
- [ ] SageMaker execution role created
- [ ] S3 bucket created and accessible
- [ ] Sample data uploaded
- [ ] Jupyter Lab launches successfully
- [ ] VSCode opens with Q Developer extension
- [ ] All verification tests pass

## 🎯 Next Steps

Once setup is complete:

1. **Review Demo Agenda**: Read through the [demo agenda](../demo-agenda/) files
2. **Explore Notebooks**: Open the sample notebooks in the `notebooks/` directory
3. **Practice Commands**: Try the Q CLI commands from `scripts/q-cli-commands.sh`
4. **Customize Environment**: Modify configurations for your specific use case

## 📞 Support

If you encounter issues during setup:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Review the [FAQ](../README.md#troubleshooting)
3. Create an issue on [GitHub](https://github.com/timwukp/amazon-q-cli-mlops-demo/issues)
4. Contact the demo author: [Tim Wu](https://www.linkedin.com/in/tim-wu-7865a7b0/)

---

**Ready to start the demo?** 🚀 

Proceed to the [Demo Agenda](../demo-agenda/01-sagemaker-mlops.md) to begin your Amazon Q CLI MLOps journey!