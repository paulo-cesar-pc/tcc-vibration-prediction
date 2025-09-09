#!/usr/bin/env python3
"""
Complete Pipeline Execution Script
==================================

Run the complete end-to-end vibration prediction pipeline.

Usage:
    python scripts/run_complete_pipeline.py [data_path]

Example:
    python scripts/run_complete_pipeline.py --output-dir results/
    python scripts/run_complete_pipeline.py custom_data/ --output-dir results/
"""

import sys
import argparse
import logging
from pathlib import Path
import json

# Add src to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline.orchestrator import run_complete_pipeline
from config.settings import get_settings
from utils.helpers import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run complete vibration prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --output-dir results/
  %(prog)s custom_data/ --models random_forest xgboost --test-size 0.3
        """
    )
    
    # Optional arguments (data_path now has default)
    parser.add_argument('data_path', 
                       nargs='?',
                       default='full_data/',
                       help='Path to input data directory (default: full_data/)')
    
    # Optional arguments
    parser.add_argument('--output-dir', '-o',
                       help='Output directory for results (default: pipeline_outputs)')
    
    parser.add_argument('--target', '-t',
                       default='CM2_PV_VRM01_VIBRATION',
                       help='Target column name (default: CM2_PV_VRM01_VIBRATION)')
    
    parser.add_argument('--test-size',
                       type=float,
                       default=0.2,
                       help='Test set size fraction (default: 0.2)')
    
    parser.add_argument('--selection-strategy',
                       choices=['balanced', 'best_performance', 'minimal'],
                       default='balanced',
                       help='Feature selection strategy (default: balanced)')
    
    parser.add_argument('--models',
                       nargs='+',
                       help='Specific models to train (default: all available)')
    
    parser.add_argument('--equipment-count',
                       type=int,
                       default=10,
                       help='Number of equipment units for business analysis (default: 10)')
    
    parser.add_argument('--maintenance-budget',
                       type=float,
                       default=500000.0,
                       help='Annual maintenance budget for ROI analysis (default: 500000.0)')
    
    parser.add_argument('--config',
                       help='Path to configuration JSON file')
    
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level (default: INFO)')
    
    parser.add_argument('--quiet', '-q',
                       action='store_true',
                       help='Suppress console output (log to file only)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file."""
    if not config_path:
        return {}
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        return json.load(f)


def validate_inputs(args):
    """Validate input arguments."""
    # Check data file exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    # Validate test size
    if not 0 < args.test_size < 1:
        raise ValueError(f"Test size must be between 0 and 1, got: {args.test_size}")
    
    # Validate equipment count
    if args.equipment_count <= 0:
        raise ValueError(f"Equipment count must be positive, got: {args.equipment_count}")
    
    # Validate maintenance budget
    if args.maintenance_budget <= 0:
        raise ValueError(f"Maintenance budget must be positive, got: {args.maintenance_budget}")


def print_execution_summary(results):
    """Print execution summary to console."""
    print("\n" + "="*60)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*60)
    
    # Metadata
    metadata = results.get('workflow_metadata', {}).get('execution_metadata', {})
    duration = metadata.get('total_duration', 0)
    print(f"Execution time: {duration:.1f} seconds")
    print(f"Status: {metadata.get('status', 'Unknown')}")
    
    # Data pipeline
    data_results = results.get('data_pipeline_results', {})
    summary = data_results.get('summary', {})
    print(f"\nData processing: {summary.get('steps_completed', 0)}/{summary.get('total_steps', 0)} steps")
    
    selected_features = data_results.get('selected_features', [])
    if selected_features:
        print(f"Features selected: {len(selected_features)}")
    
    # ML pipeline
    ml_results = results.get('ml_pipeline_results', {})
    ml_metadata = ml_results.get('pipeline_metadata', {})
    print(f"Models trained: {ml_metadata.get('models_trained', 0)}")
    
    # Best model
    recommendation = ml_results.get('recommendation', {})
    if recommendation:
        best_model = recommendation.get('recommended_model', 'Unknown')
        recommendation_summary = recommendation.get('recommendation_summary', {})
        
        print(f"\nBest model: {best_model}")
        print(f"Performance: {recommendation_summary.get('performance', 'N/A')}")
        print(f"Business impact: {recommendation_summary.get('business_impact', 'N/A')}")
        print(f"Deployment status: {recommendation_summary.get('deployment_readiness', 'N/A')}")
    
    # Executive report
    executive_report = results.get('executive_report', {})
    if executive_report:
        key_results = executive_report.get('key_results', {})
        business_impact = key_results.get('business_impact', {})
        
        if business_impact:
            print(f"\nBusiness Impact:")
            print(f"  Annual savings: {business_impact.get('annual_savings', 'N/A')}")
            print(f"  ROI: {business_impact.get('roi_percentage', 'N/A')}")
            print(f"  Payback period: {business_impact.get('payback_period', 'N/A')}")
    
    print("="*60)


def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(level=args.log_level, quiet=args.quiet)
        logger = logging.getLogger(__name__)
        
        # Validate inputs
        validate_inputs(args)
        
        # Load configuration
        config = load_config(args.config)
        
        # Log startup information
        logger.info("="*60)
        logger.info("STARTING VIBRATION PREDICTION PIPELINE")
        logger.info("="*60)
        logger.info(f"Data path: {args.data_path}")
        logger.info(f"Target column: {args.target}")
        logger.info(f"Output directory: {args.output_dir or 'pipeline_outputs'}")
        logger.info(f"Test size: {args.test_size}")
        logger.info(f"Feature selection: {args.selection_strategy}")
        if args.models:
            logger.info(f"Models: {', '.join(args.models)}")
        logger.info(f"Equipment count: {args.equipment_count}")
        logger.info(f"Maintenance budget: ${args.maintenance_budget:,.0f}")
        
        # Run complete pipeline
        results = run_complete_pipeline(
            data_path=args.data_path,
            output_dir=args.output_dir,
            config=config,
            target_column=args.target,
            test_size=args.test_size,
            selection_strategy=args.selection_strategy,
            model_names=args.models,
            annual_equipment_count=args.equipment_count,
            annual_maintenance_budget=args.maintenance_budget
        )
        
        # Print summary if not quiet
        if not args.quiet:
            print_execution_summary(results)
        
        # Log completion
        logger.info("Pipeline completed successfully")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if not args.quiet:
            print(f"\nERROR: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())