#!/usr/bin/env python3
"""
Amazon Q CLI MLOps Demo - Cost Optimization Tools

This script provides utility functions for cost analysis and optimization
demonstrations used in the Amazon Q CLI MLOps demo.
"""

import boto3
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CostOptimizationAnalyzer:
    """Cost optimization analyzer for AWS ML workloads."""
    
    def __init__(self, region: str = 'us-east-1'):
        """Initialize the cost analyzer."""
        self.region = region
        self.ce_client = boto3.client('ce', region_name=region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.emr_client = boto3.client('emr', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        
    def analyze_sagemaker_costs(self, days: int = 30) -> Dict:
        """Analyze SageMaker costs for the specified period."""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                ],
                Filter={
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon SageMaker']
                    }
                }
            )
            
            total_cost = 0
            daily_costs = []
            
            for result in response['ResultsByTime']:
                date = result['TimePeriod']['Start']
                cost = float(result['Groups'][0]['Metrics']['BlendedCost']['Amount']) if result['Groups'] else 0
                total_cost += cost
                daily_costs.append({'date': date, 'cost': cost})
            
            return {
                'total_cost': round(total_cost, 2),
                'average_daily_cost': round(total_cost / days, 2),
                'daily_breakdown': daily_costs,
                'optimization_potential': self._calculate_sagemaker_optimization_potential(total_cost)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing SageMaker costs: {str(e)}")
            return {'error': str(e)}
    
    def analyze_emr_costs(self, days: int = 30) -> Dict:
        """Analyze EMR costs for the specified period."""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                ],
                Filter={
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Elastic MapReduce']
                    }
                }
            )
            
            total_cost = 0
            daily_costs = []
            
            for result in response['ResultsByTime']:
                date = result['TimePeriod']['Start']
                cost = float(result['Groups'][0]['Metrics']['BlendedCost']['Amount']) if result['Groups'] else 0
                total_cost += cost
                daily_costs.append({'date': date, 'cost': cost})
            
            return {
                'total_cost': round(total_cost, 2),
                'average_daily_cost': round(total_cost / days, 2),
                'daily_breakdown': daily_costs,
                'optimization_potential': self._calculate_emr_optimization_potential(total_cost)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing EMR costs: {str(e)}")
            return {'error': str(e)}
    
    def get_spot_instance_savings(self, instance_type: str, hours_per_month: int = 720) -> Dict:
        """Calculate potential savings from using Spot instances."""
        # Sample pricing data (in practice, would use AWS Pricing API)
        on_demand_pricing = {
            'ml.m5.large': 0.115,
            'ml.m5.xlarge': 0.230,
            'ml.m5.2xlarge': 0.460,
            'ml.m5.4xlarge': 0.920,
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'm5.2xlarge': 0.384,
            'm5.4xlarge': 0.768
        }
        
        if instance_type not in on_demand_pricing:
            return {'error': f'Pricing data not available for {instance_type}'}
        
        on_demand_hourly = on_demand_pricing[instance_type]
        spot_hourly = on_demand_hourly * 0.3  # Assume 70% discount for demo
        
        monthly_on_demand = on_demand_hourly * hours_per_month
        monthly_spot = spot_hourly * hours_per_month
        monthly_savings = monthly_on_demand - monthly_spot
        
        return {
            'instance_type': instance_type,
            'on_demand_hourly': on_demand_hourly,
            'spot_hourly': spot_hourly,
            'monthly_on_demand_cost': round(monthly_on_demand, 2),
            'monthly_spot_cost': round(monthly_spot, 2),
            'monthly_savings': round(monthly_savings, 2),
            'savings_percentage': round((monthly_savings / monthly_on_demand) * 100, 1)
        }
    
    def analyze_s3_storage_costs(self, bucket_name: str) -> Dict:
        """Analyze S3 storage costs and optimization opportunities."""
        try:
            # Get bucket size and object count
            cloudwatch = boto3.client('cloudwatch', region_name=self.region)
            
            # Get storage metrics
            storage_response = cloudwatch.get_metric_statistics(
                Namespace='AWS/S3',
                MetricName='BucketSizeBytes',
                Dimensions=[
                    {'Name': 'BucketName', 'Value': bucket_name},
                    {'Name': 'StorageType', 'Value': 'StandardStorage'}
                ],
                StartTime=datetime.now() - timedelta(days=2),
                EndTime=datetime.now(),
                Period=86400,
                Statistics=['Average']
            )
            
            if not storage_response['Datapoints']:
                return {'error': 'No storage metrics available'}
            
            storage_bytes = storage_response['Datapoints'][-1]['Average']
            storage_gb = storage_bytes / (1024**3)
            
            # Calculate costs for different storage classes
            standard_cost = storage_gb * 0.023  # $0.023 per GB for Standard
            ia_cost = storage_gb * 0.0125      # $0.0125 per GB for IA
            glacier_cost = storage_gb * 0.004  # $0.004 per GB for Glacier
            
            return {
                'bucket_name': bucket_name,
                'storage_gb': round(storage_gb, 2),
                'current_monthly_cost': round(standard_cost, 2),
                'optimization_options': {
                    'intelligent_tiering': {
                        'monthly_cost': round(ia_cost, 2),
                        'savings': round(standard_cost - ia_cost, 2)
                    },
                    'glacier_archival': {
                        'monthly_cost': round(glacier_cost, 2),
                        'savings': round(standard_cost - glacier_cost, 2)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing S3 costs: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_sagemaker_optimization_potential(self, current_cost: float) -> Dict:
        """Calculate SageMaker optimization potential."""
        return {
            'spot_instances': {
                'potential_savings': round(current_cost * 0.7, 2),
                'percentage': 70
            },
            'right_sizing': {
                'potential_savings': round(current_cost * 0.3, 2),
                'percentage': 30
            },
            'scheduled_training': {
                'potential_savings': round(current_cost * 0.2, 2),
                'percentage': 20
            }
        }
    
    def _calculate_emr_optimization_potential(self, current_cost: float) -> Dict:
        """Calculate EMR optimization potential."""
        return {
            'serverless_migration': {
                'potential_savings': round(current_cost * 0.6, 2),
                'percentage': 60
            },
            'spot_instances': {
                'potential_savings': round(current_cost * 0.5, 2),
                'percentage': 50
            },
            'auto_scaling': {
                'potential_savings': round(current_cost * 0.3, 2),
                'percentage': 30
            }
        }
    
    def generate_cost_report(self, days: int = 30) -> Dict:
        """Generate comprehensive cost optimization report."""
        logger.info("Generating comprehensive cost optimization report...")
        
        report = {
            'report_date': datetime.now().isoformat(),
            'analysis_period_days': days,
            'sagemaker_analysis': self.analyze_sagemaker_costs(days),
            'emr_analysis': self.analyze_emr_costs(days),
            'recommendations': []
        }
        
        # Add recommendations based on analysis
        sagemaker_cost = report['sagemaker_analysis'].get('total_cost', 0)
        emr_cost = report['emr_analysis'].get('total_cost', 0)
        
        if sagemaker_cost > 100:
            report['recommendations'].append({
                'service': 'SageMaker',
                'recommendation': 'Switch to Spot instances for training jobs',
                'potential_savings': round(sagemaker_cost * 0.7, 2),
                'priority': 'High'
            })
        
        if emr_cost > 50:
            report['recommendations'].append({
                'service': 'EMR',
                'recommendation': 'Migrate to EMR Serverless for variable workloads',
                'potential_savings': round(emr_cost * 0.6, 2),
                'priority': 'High'
            })
        
        return report

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AWS ML Cost Optimization Analyzer')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--days', type=int, default=30, help='Analysis period in days')
    parser.add_argument('--service', choices=['sagemaker', 'emr', 'all'], default='all',
                       help='Service to analyze')
    parser.add_argument('--output', help='Output file for report (JSON format)')
    
    args = parser.parse_args()
    
    analyzer = CostOptimizationAnalyzer(region=args.region)
    
    if args.service == 'sagemaker':
        result = analyzer.analyze_sagemaker_costs(args.days)
    elif args.service == 'emr':
        result = analyzer.analyze_emr_costs(args.days)
    else:
        result = analyzer.generate_cost_report(args.days)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Report saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()