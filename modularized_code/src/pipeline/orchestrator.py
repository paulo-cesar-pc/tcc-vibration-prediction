"""
Pipeline Orchestrator
=====================

Complete workflow orchestration for the industrial vibration prediction system.
Combines data processing and ML pipelines into a unified end-to-end solution.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime

from config.settings import get_settings
from utils.helpers import Timer
from .data_pipeline import DataPipeline
from .ml_pipeline import MLPipeline

logger = logging.getLogger(__name__)
settings = get_settings()


class PipelineOrchestrator:
    """
    Complete workflow orchestrator for vibration prediction.
    
    This class manages the complete end-to-end workflow from raw data
    to trained models and business insights.
    """
    
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 random_state: int = 42,
                 output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize pipeline orchestrator.
        
        Parameters:
        -----------
        config : Optional[Dict[str, Any]], default=None
            Configuration for all pipeline components
        random_state : int, default=42
            Random state for reproducibility
        output_dir : Optional[Union[str, Path]], default=None
            Directory for saving outputs
        """
        self.config = config or {}
        self.random_state = random_state
        self.output_dir = Path(output_dir) if output_dir else Path("pipeline_outputs")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipelines
        self.data_pipeline = DataPipeline(
            config=self.config.get('data_pipeline', {}),
            random_state=random_state
        )
        
        self.ml_pipeline = MLPipeline(
            config=self.config.get('ml_pipeline', {}),
            random_state=random_state
        )
        
        # Orchestration state
        self.execution_log = []
        self.complete_results = {}
        self.execution_metadata = {
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'status': 'initialized'
        }
        
        logger.info(f"PipelineOrchestrator initialized with output dir: {self.output_dir}")
    
    def run_complete_workflow(self,
                             data_path: Union[str, Path],
                             target_column: str = 'vibration',
                             test_size: float = 0.2,
                             selection_strategy: str = 'balanced',
                             model_names: Optional[List[str]] = None,
                             custom_model_params: Optional[Dict[str, Dict[str, Any]]] = None,
                             annual_equipment_count: int = 10,
                             annual_maintenance_budget: float = 500000.0,
                             save_intermediate_results: bool = True) -> Dict[str, Any]:
        """
        Execute the complete end-to-end workflow.
        
        Parameters:
        -----------
        data_path : Union[str, Path]
            Path to raw data file
        target_column : str, default='vibration'
            Name of target column
        test_size : float, default=0.2
            Test set size fraction
        selection_strategy : str, default='balanced'
            Feature selection strategy
        model_names : Optional[List[str]], default=None
            Models to train
        custom_model_params : Optional[Dict[str, Dict[str, Any]]], default=None
            Custom model parameters
        annual_equipment_count : int, default=10
            Equipment count for business analysis
        annual_maintenance_budget : float, default=500000.0
            Maintenance budget for business analysis
        save_intermediate_results : bool, default=True
            Whether to save intermediate results
            
        Returns:
        --------
        Dict[str, Any]
            Complete workflow results
        """
        logger.info("="*60)
        logger.info("STARTING COMPLETE VIBRATION PREDICTION WORKFLOW")
        logger.info("="*60)
        
        self.execution_metadata['start_time'] = datetime.now()
        self.execution_metadata['status'] = 'running'
        
        total_timer = Timer("Complete workflow")
        total_timer.__enter__()
        
        try:
            # Phase 1: Data Processing Pipeline
            logger.info("PHASE 1: DATA PROCESSING PIPELINE")
            logger.info("-" * 40)
            
            self._log_step("Starting data processing pipeline")
            
            # Run complete data pipeline
            X_train, X_test, y_train, y_test = self.data_pipeline.run_complete_pipeline(
                data_path=data_path,
                target_column=target_column,
                test_size=test_size,
                selection_strategy=selection_strategy
            )
            
            self._log_step("Data processing pipeline completed")
            
            # Save intermediate results if requested
            if save_intermediate_results:
                self._save_data_pipeline_results()
            
            # Phase 2: Machine Learning Pipeline
            logger.info("\nPHASE 2: MACHINE LEARNING PIPELINE")
            logger.info("-" * 40)
            
            self._log_step("Starting ML pipeline")
            
            # Run complete ML pipeline
            ml_results = self.ml_pipeline.run_complete_pipeline(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_names=model_names,
                custom_params=custom_model_params,
                annual_equipment_count=annual_equipment_count,
                annual_maintenance_budget=annual_maintenance_budget
            )
            
            self._log_step("ML pipeline completed")
            
            # Save intermediate results if requested
            if save_intermediate_results:
                self._save_ml_pipeline_results()
            
            # Phase 3: Compile Complete Results
            logger.info("\nPHASE 3: COMPILING RESULTS")
            logger.info("-" * 40)
            
            self._log_step("Compiling complete results")
            
            # Compile comprehensive results
            self.complete_results = self._compile_complete_results(ml_results)
            
            # Generate executive report
            executive_report = self._generate_executive_report()
            self.complete_results['executive_report'] = executive_report
            
            self._log_step("Results compilation completed")
            
            # Phase 4: Save Final Results
            logger.info("\nPHASE 4: SAVING RESULTS")
            logger.info("-" * 40)
            
            self._log_step("Saving final results")
            self._save_complete_results()
            self._log_step("All results saved")
            
            # Update execution metadata
            self.execution_metadata['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            self.execution_metadata['status'] = 'failed'
            self.execution_metadata['error'] = str(e)
            raise
        
        finally:
            total_timer.__exit__(None, None, None)
            self.execution_metadata['end_time'] = datetime.now()
            self.execution_metadata['total_duration'] = total_timer.elapsed_time
        
        # Final summary
        self._print_final_summary()
        
        logger.info("="*60)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        return self.complete_results
    
    def _log_step(self, message: str):
        """Log a workflow step with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.execution_log.append(log_entry)
        logger.info(log_entry)
    
    def _save_data_pipeline_results(self):
        """Save data pipeline intermediate results."""
        data_output_dir = self.output_dir / "data_pipeline"
        data_output_dir.mkdir(exist_ok=True)
        
        # Save pipeline state
        self.data_pipeline.save_pipeline_state(data_output_dir / "pipeline_state.pkl")
        
        # Save pipeline summary
        summary = self.data_pipeline.get_pipeline_summary()
        with open(data_output_dir / "pipeline_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save selected features
        if self.data_pipeline.selected_features:
            with open(data_output_dir / "selected_features.json", 'w') as f:
                json.dump({
                    'features': self.data_pipeline.selected_features,
                    'feature_count': len(self.data_pipeline.selected_features)
                }, f, indent=2)
        
        logger.info(f"Data pipeline results saved to {data_output_dir}")
    
    def _save_ml_pipeline_results(self):
        """Save ML pipeline intermediate results."""
        ml_output_dir = self.output_dir / "ml_pipeline"
        self.ml_pipeline.save_pipeline_results(ml_output_dir)
    
    def _compile_complete_results(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive results from both pipelines."""
        return {
            'workflow_metadata': {
                'execution_metadata': self.execution_metadata,
                'execution_log': self.execution_log,
                'configuration': self.config
            },
            'data_pipeline_results': {
                'summary': self.data_pipeline.get_pipeline_summary(),
                'selected_features': self.data_pipeline.selected_features
            },
            'ml_pipeline_results': ml_results,
            'data_characteristics': self._analyze_data_characteristics(),
            'model_comparison': self._create_model_comparison_summary(ml_results)
        }
    
    def _analyze_data_characteristics(self) -> Dict[str, Any]:
        """Analyze characteristics of the processed data."""
        characteristics = {}
        
        if self.data_pipeline.raw_data is not None:
            characteristics['raw_data'] = {
                'shape': self.data_pipeline.raw_data.shape,
                'columns': list(self.data_pipeline.raw_data.columns),
                'memory_usage_mb': self.data_pipeline.raw_data.memory_usage(deep=True).sum() / 1024**2
            }
        
        if self.data_pipeline.cleaned_data is not None:
            characteristics['clean_data'] = {
                'shape': self.data_pipeline.cleaned_data.shape,
                'data_quality_score': self._calculate_data_quality_score(self.data_pipeline.cleaned_data)
            }
        
        if self.data_pipeline.engineered_features is not None:
            characteristics['engineered_features'] = {
                'shape': self.data_pipeline.engineered_features.shape,
                'feature_types': self._categorize_features(self.data_pipeline.engineered_features.columns)
            }
        
        return characteristics
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate a data quality score."""
        # Simple quality score based on completeness
        completeness = (1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        return round(completeness, 2)
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, int]:
        """Categorize features by type."""
        categories = {
            'rolling_mean': 0,
            'rolling_std': 0,
            'temporal': 0,
            'lag': 0,
            'original': 0,
            'other': 0
        }
        
        for feature in feature_names:
            if '_rolling_mean_' in feature:
                categories['rolling_mean'] += 1
            elif '_rolling_std_' in feature:
                categories['rolling_std'] += 1
            elif feature in ['hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos']:
                categories['temporal'] += 1
            elif '_lag_' in feature:
                categories['lag'] += 1
            elif any(keyword in feature for keyword in ['temperature', 'pressure', 'speed', 'load']):
                categories['original'] += 1
            else:
                categories['other'] += 1
        
        return categories
    
    def _create_model_comparison_summary(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary comparison of all models."""
        recommendation = ml_results.get('recommendation', {})
        
        if not recommendation:
            return {}
        
        model_scores = recommendation.get('model_scores', {})
        
        # Create ranking
        sorted_models = sorted(
            model_scores.items(), 
            key=lambda x: x[1]['composite_score'], 
            reverse=True
        )
        
        comparison_summary = {
            'total_models_trained': len(model_scores),
            'model_ranking': [
                {
                    'rank': i + 1,
                    'model_name': model_name,
                    'composite_score': scores['composite_score'],
                    'test_r2': scores['test_r2'],
                    'roi_percentage': scores['roi_percentage'],
                    'deployment_status': scores['deployment_status']
                }
                for i, (model_name, scores) in enumerate(sorted_models)
            ],
            'recommended_model': recommendation.get('recommended_model'),
            'performance_categories': self._categorize_model_performance(model_scores)
        }
        
        return comparison_summary
    
    def _categorize_model_performance(self, model_scores: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Categorize models by performance level."""
        categories = {
            'excellent': [],
            'good': [],
            'fair': [],
            'poor': []
        }
        
        for model_name, scores in model_scores.items():
            r2_score = scores.get('test_r2', 0)
            
            if r2_score >= 0.9:
                categories['excellent'].append(model_name)
            elif r2_score >= 0.8:
                categories['good'].append(model_name)
            elif r2_score >= 0.6:
                categories['fair'].append(model_name)
            else:
                categories['poor'].append(model_name)
        
        return categories
    
    def _generate_executive_report(self) -> Dict[str, Any]:
        """Generate executive summary report."""
        if not self.complete_results.get('ml_pipeline_results'):
            return {}
        
        ml_results = self.complete_results['ml_pipeline_results']
        recommendation = ml_results.get('recommendation', {})
        
        if not recommendation:
            return {}
        
        best_model = recommendation.get('recommended_model', 'Unknown')
        model_scores = recommendation.get('model_scores', {})
        
        executive_report = {
            'project_title': 'Industrial Vibration Prediction System',
            'execution_date': self.execution_metadata['start_time'].strftime("%Y-%m-%d"),
            'total_execution_time': f"{self.execution_metadata['total_duration']:.1f} seconds",
            
            # Key Results
            'key_results': {
                'recommended_model': best_model,
                'model_performance': self._get_model_performance_summary(best_model, model_scores),
                'business_impact': self._get_business_impact_summary(ml_results),
                'deployment_readiness': self._get_deployment_readiness_summary(ml_results)
            },
            
            # Technical Summary
            'technical_summary': {
                'data_processing': self._get_data_processing_summary(),
                'model_comparison': self._get_model_comparison_summary(),
                'feature_engineering': self._get_feature_engineering_summary()
            },
            
            # Recommendations
            'recommendations': self._get_executive_recommendations(ml_results)
        }
        
        return executive_report
    
    def _get_model_performance_summary(self, model_name: str, model_scores: Dict) -> Dict[str, Any]:
        """Get performance summary for the best model."""
        if model_name not in model_scores:
            return {}
        
        scores = model_scores[model_name]
        return {
            'accuracy': f"{scores.get('test_r2', 0)*100:.1f}%",
            'error_rate': f"{scores.get('test_rmse', 0):.4f}",
            'overall_score': f"{scores.get('composite_score', 0):.1f}/100"
        }
    
    def _get_business_impact_summary(self, ml_results: Dict) -> Dict[str, Any]:
        """Get business impact summary."""
        recommendation = ml_results.get('recommendation', {})
        detailed_analysis = recommendation.get('detailed_analysis', {})
        business_insights = detailed_analysis.get('business_insights', {})
        
        if not business_insights:
            return {}
        
        business_impact = business_insights.get('business_impact', {})
        roi_info = business_impact.get('roi', {})
        cost_savings = business_impact.get('cost_savings', {})
        
        return {
            'annual_savings': f"${cost_savings.get('total_annual_savings', 0):,.0f}",
            'roi_percentage': f"{roi_info.get('roi_percentage', 0):.1f}%",
            'payback_period': f"{roi_info.get('payback_period_months', 0):.1f} months"
        }
    
    def _get_deployment_readiness_summary(self, ml_results: Dict) -> Dict[str, str]:
        """Get deployment readiness summary."""
        recommendation = ml_results.get('recommendation', {})
        detailed_analysis = recommendation.get('detailed_analysis', {})
        validation = detailed_analysis.get('validation', {})
        
        if not validation:
            return {'status': 'Unknown', 'message': 'Validation not completed'}
        
        overall_assessment = validation.get('overall_assessment', {})
        return {
            'status': overall_assessment.get('readiness_status', 'Unknown'),
            'message': overall_assessment.get('readiness_message', 'No assessment available')
        }
    
    def _get_data_processing_summary(self) -> Dict[str, Any]:
        """Get data processing summary."""
        summary = self.data_pipeline.get_pipeline_summary()
        return {
            'steps_completed': f"{summary.get('steps_completed', 0)}/{summary.get('total_steps', 0)}",
            'features_engineered': summary.get('final_feature_count', 0),
            'data_quality': 'High'  # Could be calculated based on actual metrics
        }
    
    def _get_model_comparison_summary(self) -> Dict[str, Any]:
        """Get model comparison summary."""
        comparison = self.complete_results.get('model_comparison', {})
        return {
            'models_trained': comparison.get('total_models_trained', 0),
            'performance_distribution': comparison.get('performance_categories', {}),
            'best_performing_category': self._get_best_category(comparison.get('performance_categories', {}))
        }
    
    def _get_feature_engineering_summary(self) -> Dict[str, Any]:
        """Get feature engineering summary."""
        characteristics = self.complete_results.get('data_characteristics', {})
        engineered = characteristics.get('engineered_features', {})
        
        return {
            'total_features': engineered.get('shape', [0, 0])[1],
            'feature_types': engineered.get('feature_types', {}),
            'selected_features': len(self.data_pipeline.selected_features or [])
        }
    
    def _get_best_category(self, categories: Dict[str, List]) -> str:
        """Get the category with the most models."""
        if not categories:
            return 'Unknown'
        
        return max(categories.keys(), key=lambda x: len(categories[x]))
    
    def _get_executive_recommendations(self, ml_results: Dict) -> List[str]:
        """Get executive recommendations."""
        recommendation = ml_results.get('recommendation', {})
        detailed_analysis = recommendation.get('detailed_analysis', {})
        business_insights = detailed_analysis.get('business_insights', {})
        
        if not business_insights:
            return []
        
        deployment_recs = business_insights.get('deployment_recommendations', {})
        return deployment_recs.get('recommendations', [])[:5]  # Top 5 recommendations
    
    def _save_complete_results(self):
        """Save all complete results to files."""
        # Save complete results as JSON
        with open(self.output_dir / "complete_results.json", 'w') as f:
            json.dump(self._serialize_for_json(self.complete_results), f, indent=2, default=str)
        
        # Save executive report separately
        if 'executive_report' in self.complete_results:
            with open(self.output_dir / "executive_report.json", 'w') as f:
                json.dump(
                    self._serialize_for_json(self.complete_results['executive_report']), 
                    f, indent=2, default=str
                )
        
        # Save execution log
        with open(self.output_dir / "execution_log.txt", 'w') as f:
            f.write("\n".join(self.execution_log))
        
        logger.info(f"Complete results saved to {self.output_dir}")
    
    def _serialize_for_json(self, obj: Any) -> Any:
        """Convert complex types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: self._serialize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_) or (hasattr(np, 'bool_') and isinstance(obj, type(np.bool_(True)))):
            return bool(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return obj
    
    def _print_final_summary(self):
        """Print final execution summary."""
        logger.info("\n" + "="*60)
        logger.info("FINAL EXECUTION SUMMARY")
        logger.info("="*60)
        
        # Timing information
        duration = self.execution_metadata.get('total_duration', 0)
        logger.info(f"Total execution time: {duration:.1f} seconds")
        
        # Data pipeline results
        data_summary = self.data_pipeline.get_pipeline_summary()
        logger.info(f"Data pipeline steps: {data_summary.get('steps_completed', 0)}/{data_summary.get('total_steps', 0)}")
        
        if self.data_pipeline.selected_features:
            logger.info(f"Features selected: {len(self.data_pipeline.selected_features)}")
        
        # ML pipeline results
        ml_summary = self.ml_pipeline.get_pipeline_summary()
        logger.info(f"ML pipeline steps: {ml_summary.get('steps_completed', 0)}/{ml_summary.get('total_steps', 0)}")
        logger.info(f"Models trained: {ml_summary.get('models_trained', 0)}")
        
        if ml_summary.get('best_model'):
            best_r2 = ml_summary.get('best_model_metrics', {}).get('test_r2', 0)
            logger.info(f"Best model: {ml_summary['best_model']} (R² = {best_r2:.4f})")
        
        # Output location
        logger.info(f"Results saved to: {self.output_dir}")
        
        logger.info("="*60)


def run_complete_pipeline(data_path: Union[str, Path],
                         output_dir: Optional[Union[str, Path]] = None,
                         config: Optional[Dict[str, Any]] = None,
                         target_column: str = 'vibration',
                         test_size: float = 0.2,
                         selection_strategy: str = 'balanced',
                         model_names: Optional[List[str]] = None,
                         annual_equipment_count: int = 10,
                         annual_maintenance_budget: float = 500000.0) -> Dict[str, Any]:
    """
    Run the complete pipeline from data to business insights.
    
    Parameters:
    -----------
    data_path : Union[str, Path]
        Path to raw data file
    output_dir : Optional[Union[str, Path]], default=None
        Output directory for results
    config : Optional[Dict[str, Any]], default=None
        Pipeline configuration
    target_column : str, default='vibration'
        Target column name
    test_size : float, default=0.2
        Test set size fraction
    selection_strategy : str, default='balanced'
        Feature selection strategy
    model_names : Optional[List[str]], default=None
        Models to train
    annual_equipment_count : int, default=10
        Equipment count for business analysis
    annual_maintenance_budget : float, default=500000.0
        Maintenance budget for business analysis
        
    Returns:
    --------
    Dict[str, Any]
        Complete pipeline results
    """
    orchestrator = PipelineOrchestrator(
        config=config,
        output_dir=output_dir
    )
    
    return orchestrator.run_complete_workflow(
        data_path=data_path,
        target_column=target_column,
        test_size=test_size,
        selection_strategy=selection_strategy,
        model_names=model_names,
        annual_equipment_count=annual_equipment_count,
        annual_maintenance_budget=annual_maintenance_budget
    )