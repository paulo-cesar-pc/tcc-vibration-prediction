#!/usr/bin/env python3
"""
Data Pipeline Execution Script
==============================

Run only the data processing pipeline for vibration prediction.

Usage:
    python scripts/run_data_pipeline.py path/to/data.csv

Example:
    python scripts/run_data_pipeline.py data/vibration_data.csv --output-dir data_outputs/
"""

import sys
import argparse
import logging
from pathlib import Path
import json

# Add src to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline.data_pipeline import create_data_pipeline
from config.settings import get_settings
from utils.helpers import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run data processing pipeline for vibration prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/vibration_data.csv
  %(prog)s data/vibration_data.csv --output-dir data_outputs/ --target vibration
  %(prog)s data/vibration_data.csv --selection balanced --test-size 0.3
        """
    )
    
    # Required arguments
    parser.add_argument('data_path', 
                       help='Path to input data CSV file')
    
    # Optional arguments
    parser.add_argument('--output-dir', '-o',
                       help='Output directory for results (default: data_pipeline_outputs)')
    
    parser.add_argument('--target', '-t',
                       default='vibration',
                       help='Target column name (default: vibration)')
    
    parser.add_argument('--test-size',
                       type=float,
                       default=0.2,
                       help='Test set size fraction (default: 0.2)')
    
    parser.add_argument('--selection-strategy',
                       choices=['balanced', 'best_performance', 'minimal'],
                       default='balanced',
                       help='Feature selection strategy (default: balanced)')
    
    parser.add_argument('--config',
                       help='Path to configuration JSON file')
    
    parser.add_argument('--save-data',
                       action='store_true',
                       help='Save processed data to CSV files')
    
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


def save_processed_data(pipeline, output_dir, target_column='vibration'):
    """Save processed data to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw data (if available)
    if pipeline.raw_data is not None:
        pipeline.raw_data.to_csv(output_dir / 'raw_data.csv', index=False)
        print(f"Raw data saved: {output_dir / 'raw_data.csv'}")
    
    # Save clean data (if available)
    if pipeline.clean_data is not None:
        pipeline.clean_data.to_csv(output_dir / 'clean_data.csv', index=False)
        print(f"Clean data saved: {output_dir / 'clean_data.csv'}")
    
    # Save engineered features (if available)
    if pipeline.engineered_features is not None:
        pipeline.engineered_features.to_csv(output_dir / 'engineered_features.csv', index=False)
        print(f"Engineered features saved: {output_dir / 'engineered_features.csv'}")
    
    # Save final processed data
    try:
        X_train, X_test, y_train, y_test = pipeline.get_final_data(target_column)
        
        # Save training data
        train_data = X_train.copy()
        train_data[target_column] = y_train
        train_data.to_csv(output_dir / 'train_data.csv', index=False)
        print(f"Training data saved: {output_dir / 'train_data.csv'}")
        
        # Save test data
        test_data = X_test.copy()
        test_data[target_column] = y_test
        test_data.to_csv(output_dir / 'test_data.csv', index=False)
        print(f"Test data saved: {output_dir / 'test_data.csv'}")
        
    except Exception as e:
        print(f"Could not save final data: {e}")


def print_pipeline_summary(pipeline):
    """Print pipeline execution summary."""
    summary = pipeline.get_pipeline_summary()
    
    print("\n" + "="*50)
    print("DATA PIPELINE SUMMARY")
    print("="*50)
    
    print(f"Steps completed: {summary.get('steps_completed', 0)}/{summary.get('total_steps', 0)}")
    
    if 'execution_summary' in summary:
        exec_summary = summary['execution_summary']
        
        # Execution times
        if 'execution_times' in exec_summary:
            print(f"\nExecution times:")
            for step, time_taken in exec_summary['execution_times'].items():
                print(f"  {step}: {time_taken:.2f}s")
        
        # Data shapes
        if 'data_shapes' in exec_summary:
            print(f"\nData shapes:")
            for step, shape in exec_summary['data_shapes'].items():
                print(f"  {step}: {shape}")
    
    # Selected features
    if pipeline.selected_features:
        print(f"\nSelected features ({len(pipeline.selected_features)}):")
        for i, feature in enumerate(pipeline.selected_features[:10], 1):
            print(f"  {i}. {feature}")
        if len(pipeline.selected_features) > 10:
            print(f"  ... and {len(pipeline.selected_features) - 10} more")
    
    # Final data info
    if hasattr(pipeline, 'train_test_split') and pipeline.train_test_split:
        print(f"\nFinal data split:")
        print(f"  Training samples: {len(pipeline.train_test_split['X_train']):,}")
        print(f"  Test samples: {len(pipeline.train_test_split['X_test']):,}")
        print(f"  Selected features: {len(pipeline.selected_features or [])}")
    
    print("="*50)


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
        
        # Set output directory
        output_dir = args.output_dir or 'data_pipeline_outputs'
        
        # Log startup information
        logger.info("="*50)
        logger.info("STARTING DATA PROCESSING PIPELINE")
        logger.info("="*50)
        logger.info(f"Data path: {args.data_path}")
        logger.info(f"Target column: {args.target}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Test size: {args.test_size}")
        logger.info(f"Feature selection: {args.selection_strategy}")
        
        # Create and run data pipeline
        pipeline = create_data_pipeline(config=config)
        
        final_data = pipeline.run_complete_pipeline(
            data_path=args.data_path,
            target_column=args.target,
            test_size=args.test_size,
            selection_strategy=args.selection_strategy
        )
        
        # Save pipeline state
        pipeline.save_pipeline_state(Path(output_dir) / 'pipeline_state.pkl')
        
        # Save pipeline summary
        summary = pipeline.get_pipeline_summary()
        with open(Path(output_dir) / 'pipeline_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save selected features
        if pipeline.selected_features:
            with open(Path(output_dir) / 'selected_features.json', 'w') as f:
                json.dump({
                    'features': pipeline.selected_features,
                    'feature_count': len(pipeline.selected_features),
                    'selection_strategy': args.selection_strategy
                }, f, indent=2)
        
        # Save processed data if requested
        if args.save_data:
            save_processed_data(pipeline, output_dir, args.target)
        
        # Print summary if not quiet
        if not args.quiet:
            print_pipeline_summary(pipeline)
            
            X_train, X_test, y_train, y_test = final_data
            print(f"\nFinal processed data ready for modeling:")
            print(f"  Training set: {X_train.shape}")
            print(f"  Test set: {X_test.shape}")
            print(f"  Features: {X_train.shape[1]}")
        
        # Log completion
        logger.info("Data pipeline completed successfully")
        logger.info(f"Results saved to: {output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
        if not args.quiet:
            print(f"\nERROR: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())