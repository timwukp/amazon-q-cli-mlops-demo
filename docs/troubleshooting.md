# Troubleshooting Guide

## Common Issues and Solutions

### 1. AWS Credentials Issues

**Problem**: `Unable to locate credentials` or `Access Denied` errors

**Solutions**:
```bash
# Check current credentials
aws sts get-caller-identity

# Configure credentials
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### 2. Python Package Issues

**Problem**: Import errors or package conflicts

**Solutions**:
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Jupyter Notebook Issues

**Problem**: Kernel not found or notebook won't start

**Solutions**:
```bash
# Install kernel
python -m ipykernel install --user --name=amazon-q-mlops-demo

# Start Jupyter Lab
jupyter lab
```

### 4. S3 Bucket Issues

**Problem**: Bucket doesn't exist or access denied

**Solutions**:
```bash
# Create bucket
aws s3 mb s3://your-unique-bucket-name

# Check bucket permissions
aws s3api get-bucket-policy --bucket your-bucket-name
```

### 5. SageMaker Role Issues

**Problem**: Cannot assume SageMaker execution role

**Solutions**:
```bash
# Create SageMaker execution role
aws iam create-role --role-name SageMakerExecutionRole --assume-role-policy-document file://trust-policy.json

# Attach required policies
aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

## Getting Help

- Check the [Setup Guide](setup-guide.md)
- Review the [README](../README.md)
- Create an issue on [GitHub](https://github.com/timwukp/amazon-q-cli-mlops-demo/issues)
