# Demo Notebooks

This directory contains Jupyter notebooks for the Amazon Q CLI MLOps demo.

## 📚 Notebook Overview

### 1. `mlops-pipeline-demo.ipynb`
**Main MLOps Pipeline Demonstration**
- Complete end-to-end MLOps pipeline
- SageMaker integration
- Model training and deployment
- Performance monitoring

### 2. `performance-optimization.ipynb`
**Performance Optimization Examples**
- Model inference optimization
- Data pipeline tuning
- Resource allocation strategies
- Benchmarking and profiling

### 3. `cost-analysis.ipynb`
**Cost Analysis and Optimization**
- AWS cost analysis
- Cost optimization strategies
- Spot instance configuration
- ROI calculations

### 4. `data-processing-emr.ipynb`
**EMR and PySpark Examples**
- EMR Serverless setup
- PySpark data processing
- Performance optimization
- Cost comparison

## 🚀 Getting Started

### Prerequisites
1. Complete the [setup guide](../docs/setup-guide.md)
2. Activate the virtual environment: `source venv/bin/activate`
3. Start Jupyter Lab: `jupyter lab`

### Running the Notebooks
1. Open Jupyter Lab in your browser
2. Navigate to the `notebooks/` directory
3. Select the notebook you want to run
4. Choose the "Amazon Q MLOps Demo" kernel
5. Run cells sequentially

## 🎯 Demo Flow

### Recommended Order:
1. **Start with**: `mlops-pipeline-demo.ipynb` - Get familiar with the basic concepts
2. **Continue with**: `performance-optimization.ipynb` - Learn optimization techniques
3. **Explore**: `cost-analysis.ipynb` - Understand cost implications
4. **Advanced**: `data-processing-emr.ipynb` - Scale with EMR and PySpark

## 🔧 Customization

### Adapting for Your Use Case:
- Update AWS account details in notebook variables
- Modify dataset paths and S3 bucket names
- Adjust model parameters and configurations
- Customize monitoring and alerting thresholds

### Environment Variables:
Each notebook reads from the `.env` file in the root directory. Update these values:
```
AWS_DEFAULT_REGION=us-east-1
SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole
S3_BUCKET=your-demo-bucket
```

## 📊 Expected Outputs

### Performance Metrics:
- Model training time: ~10-15 minutes
- Inference latency: <100ms
- Data processing throughput: Variable based on dataset size

### Cost Estimates:
- SageMaker training: $2-5 per run
- EMR processing: $1-3 per hour
- S3 storage: <$1 per month for demo data

## 🔍 Troubleshooting

### Common Issues:
1. **Kernel not found**: Select "Amazon Q MLOps Demo" kernel
2. **AWS credentials**: Ensure `aws configure` is completed
3. **Permission errors**: Verify IAM roles and policies
4. **Resource limits**: Check AWS service quotas

### Getting Help:
- Check the [troubleshooting guide](../docs/troubleshooting.md)
- Review notebook comments and markdown cells
- Create an issue on GitHub for specific problems

## 🎬 Demo Integration

### With Amazon Q CLI:
- Use Q CLI commands alongside notebook execution
- Compare Q CLI suggestions with notebook implementations
- Demonstrate real-time optimization recommendations

### With VSCode Q Developer:
- Open notebooks in VSCode with Q Developer extension
- Get intelligent code suggestions and explanations
- Generate additional code based on notebook context

## 📝 Notes

- Notebooks are designed for educational and demonstration purposes
- Production implementations may require additional security and error handling
- Cost estimates are approximate and may vary based on usage patterns
- Always clean up AWS resources after demo to avoid unnecessary charges

---

**Ready to explore?** Start with `mlops-pipeline-demo.ipynb` and follow the guided demo experience! 🚀