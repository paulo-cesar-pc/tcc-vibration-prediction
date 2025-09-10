"""
Machine Learning Pipeline
=========================

Complete ML pipeline for model training, evaluation, and deployment
preparation for industrial vibration prediction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import pickle
import json

from config.settings import get_settings
from models.trainer import ModelTrainer
from models.registry import ModelRegistry, get_registry
from evaluation.metrics import evaluate_model_performance
from evaluation.validator import validate_model_performance
from evaluation.visualizer import create_evaluation_plots
from evaluation.analyzer import generate_business_insights
from utils.helpers import Timer

logger = logging.getLogger(__name__)
settings = get_settings()


class MLPipeline:
    """
    Complete machine learning pipeline for vibration prediction.
    
    This class orchestrates model training, evaluation, validation,
    and deployment preparation.
    """
    
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 random_state: int = 42,
                 registry: Optional[ModelRegistry] = None):
        """
        Initialize ML pipeline.
        
        Parameters:
        -----------
        config : Optional[Dict[str, Any]], default=None
            Configuration dictionary for pipeline
        random_state : int, default=42
            Random state for reproducibility
        registry : Optional[ModelRegistry], default=None
            Model registry to use
        """
        self.config = config or {}
        self.random_state = random_state
        self.registry = registry or get_registry()
        
        # Pipeline components
        self.model_trainer = ModelTrainer(registry=self.registry, random_state=random_state)
        self.trained_models = {}
        self.evaluation_results = {}
        self.validation_results = {}
        self.business_insights = {}
        
        # Pipeline state
        self.data_loaded = False
        self.models_trained = False
        self.models_evaluated = False
        self.best_model_name = None
        
        # Metadata
        self.pipeline_metadata = {
            'steps_completed': [],
            'execution_times': {},
            'model_count': 0,
            'best_model_metrics': {}
        }
        
        logger.info("MLPipeline initialized")
    
    def setup_data(self,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> 'MLPipeline':
        """
        Setup training and test data for the pipeline.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature matrix
        y_train : pd.Series
            Training target variable
        X_test : pd.DataFrame
            Test feature matrix
        y_test : pd.Series
            Test target variable
            
        Returns:
        --------
        MLPipeline
            Self for method chaining
        """
        logger.info("Setting up ML pipeline data")
        
        with Timer("Data setup") as timer:
            self.model_trainer.setup_data(X_train, y_train, X_test, y_test)
            self.data_loaded = True
        
        self._update_metadata("setup_data", timer.elapsed_time)
        
        logger.info(f"ML pipeline data setup complete:")
        logger.info(f"  • Training samples: {len(X_train):,}")
        logger.info(f"  • Test samples: {len(X_test):,}")
        logger.info(f"  • Features: {X_train.shape[1]}")
        
        return self
    
    def train_models(self,
                    model_names: Optional[List[str]] = None,
                    custom_params: Optional[Dict[str, Dict[str, Any]]] = None,
                    include_scaling_variants: bool = True) -> Dict[str, Any]:
        """
        Train multiple models and compare performance.
        
        Parameters:
        -----------
        model_names : Optional[List[str]], default=None
            List of model names to train
        custom_params : Optional[Dict[str, Dict[str, Any]]], default=None
            Custom parameters for specific models
        include_scaling_variants : bool, default=True
            Whether to include scaled variants
            
        Returns:
        --------
        Dict[str, Any]
            Training results summary
        """
        if not self.data_loaded:
            raise ValueError("Data not loaded. Call setup_data() first.")
        
        logger.info("Training models")
        
        with Timer("Model training") as timer:
            self.trained_models = self.model_trainer.train_multiple_models(
                model_names=model_names,
                custom_params=custom_params,
                include_scaling_variants=include_scaling_variants
            )
            
            self.models_trained = True
        
        # Update metadata
        self._update_metadata("train_models", timer.elapsed_time)
        self.pipeline_metadata['model_count'] = len(self.trained_models)
        
        # Get training results
        training_results = self.model_trainer.get_training_results(as_dataframe=True)
        
        if not training_results.empty:
            best_model_row = training_results.iloc[0]
            self.best_model_name = best_model_row['model_alias']
            self.pipeline_metadata['best_model_metrics'] = {
                'name': self.best_model_name,
                'test_r2': best_model_row.get('test_r2', 0),
                'test_rmse': best_model_row.get('test_rmse', float('inf'))
            }
        
        logger.info(f"Model training complete:")
        logger.info(f"  • Models trained: {len(self.trained_models)}")
        logger.info(f"  • Best model: {self.best_model_name}")
        if self.best_model_name:
            logger.info(f"  • Best R²: {self.pipeline_metadata['best_model_metrics']['test_r2']:.4f}")
        
        return training_results.to_dict('records') if not training_results.empty else {}
    
    def evaluate_models(self,
                       accuracy_thresholds: Optional[List[float]] = None,
                       create_plots: bool = True) -> Dict[str, Any]:
        """
        Evaluate all trained models comprehensively.
        
        Parameters:
        -----------
        accuracy_thresholds : Optional[List[float]], default=None
            Accuracy thresholds for evaluation
        create_plots : bool, default=True
            Whether to create evaluation plots
            
        Returns:
        --------
        Dict[str, Any]
            Evaluation results for all models
        """
        if not self.models_trained:
            raise ValueError("Models not trained. Call train_models() first.")
        
        logger.info("Evaluating models")
        
        accuracy_thresholds = accuracy_thresholds or [0.001, 0.002, 0.005, 0.01]
        
        with Timer("Model evaluation") as timer:
            # Get data from trainer
            X_train = self.model_trainer.X_train_
            y_train = self.model_trainer.y_train_
            X_test = self.model_trainer.X_test_
            y_test = self.model_trainer.y_test_
            
            # Evaluate each model
            for model_name in self.trained_models.keys():
                try:
                    model = self.registry.get_model(model_name)
                    
                    # Choose appropriate data based on scaling requirements
                    if model.requires_scaling:
                        X_train_data = self.model_trainer.X_train_scaled_
                        X_test_data = self.model_trainer.X_test_scaled_
                    else:
                        X_train_data = X_train
                        X_test_data = X_test
                    
                    # Generate predictions
                    train_pred = model.predict(X_train_data)
                    test_pred = model.predict(X_test_data)
                    
                    # Comprehensive evaluation
                    evaluation = evaluate_model_performance(
                        y_train, train_pred, y_test, test_pred, accuracy_thresholds
                    )
                    
                    # Store evaluation results
                    self.evaluation_results[model_name] = evaluation
                    
                    # Create plots if requested
                    if create_plots:
                        feature_importance = model.get_feature_importance()
                        time_index = getattr(X_test, 'index', None)
                        
                        plots = create_evaluation_plots(
                            y_train, train_pred, y_test, test_pred,
                            model_name=model_name,
                            feature_importance=feature_importance,
                            time_index=time_index,
                            save_plots=self.config.get('save_plots', False),
                            output_dir=self.config.get('plots_output_dir')
                        )
                        
                        # Store plot references
                        if 'plots' not in self.evaluation_results[model_name]:
                            self.evaluation_results[model_name]['plots'] = {}
                        self.evaluation_results[model_name]['plots'] = list(plots.keys())
                    
                    logger.info(f"Evaluated model: {model_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate model {model_name}: {e}")
                    continue
            
            self.models_evaluated = True
        
        self._update_metadata("evaluate_models", timer.elapsed_time)
        
        logger.info(f"Model evaluation complete for {len(self.evaluation_results)} models")
        
        return self.evaluation_results
    
    def validate_models(self,
                       validation_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Validate models for deployment readiness.
        
        Parameters:
        -----------
        validation_thresholds : Optional[Dict[str, float]], default=None
            Custom validation thresholds
            
        Returns:
        --------
        Dict[str, Any]
            Validation results for all models
        """
        if not self.models_evaluated:
            raise ValueError("Models not evaluated. Call evaluate_models() first.")
        
        logger.info("Validating models for deployment")
        
        with Timer("Model validation") as timer:
            # Get data from trainer
            y_train = self.model_trainer.y_train_
            y_test = self.model_trainer.y_test_
            feature_count = self.model_trainer.X_train_.shape[1]
            
            # Validate each evaluated model
            for model_name in self.evaluation_results.keys():
                try:
                    model = self.registry.get_model(model_name)
                    
                    # Choose appropriate data
                    if model.requires_scaling:
                        X_train_data = self.model_trainer.X_train_scaled_
                        X_test_data = self.model_trainer.X_test_scaled_
                    else:
                        X_train_data = self.model_trainer.X_train_
                        X_test_data = self.model_trainer.X_test_
                    
                    # Generate predictions
                    train_pred = model.predict(X_train_data)
                    test_pred = model.predict(X_test_data)
                    
                    # Validate model
                    validation = validate_model_performance(
                        y_train, train_pred, y_test, test_pred,
                        model_name=model_name,
                        feature_count=feature_count,
                        validation_thresholds=validation_thresholds
                    )
                    
                    # Store validation results
                    self.validation_results[model_name] = validation
                    
                    logger.info(f"Validated model: {model_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to validate model {model_name}: {e}")
                    continue
        
        self._update_metadata("validate_models", timer.elapsed_time)
        
        logger.info(f"Model validation complete for {len(self.validation_results)} models")
        
        return self.validation_results
    
    def generate_business_analysis(self,
                                  annual_equipment_count: int = 10,
                                  annual_maintenance_budget: float = 500000.0) -> Dict[str, Any]:
        """
        Generate business insights and recommendations.
        
        Parameters:
        -----------
        annual_equipment_count : int, default=10
            Number of equipment units monitored
        annual_maintenance_budget : float, default=500000.0
            Annual maintenance budget
            
        Returns:
        --------
        Dict[str, Any]
            Business analysis for all models
        """
        if not self.models_evaluated:
            raise ValueError("Models not evaluated. Call evaluate_models() first.")
        
        logger.info("Generating business analysis")
        
        with Timer("Business analysis") as timer:
            # Get data from trainer
            y_train = self.model_trainer.y_train_
            y_test = self.model_trainer.y_test_
            feature_count = self.model_trainer.X_train_.shape[1]
            
            # Generate business insights for each model
            for model_name in self.evaluation_results.keys():
                try:
                    model = self.registry.get_model(model_name)
                    
                    # Choose appropriate data
                    if model.requires_scaling:
                        X_train_data = self.model_trainer.X_train_scaled_
                        X_test_data = self.model_trainer.X_test_scaled_
                    else:
                        X_train_data = self.model_trainer.X_train_
                        X_test_data = self.model_trainer.X_test_
                    
                    # Generate predictions
                    train_pred = model.predict(X_train_data)
                    test_pred = model.predict(X_test_data)
                    
                    # Generate business insights
                    business_analysis = generate_business_insights(
                        y_train, train_pred, y_test, test_pred,
                        model_name=model_name,
                        feature_count=feature_count,
                        annual_equipment_count=annual_equipment_count,
                        annual_maintenance_budget=annual_maintenance_budget
                    )
                    
                    # Store business insights
                    self.business_insights[model_name] = business_analysis
                    
                    logger.info(f"Generated business analysis for: {model_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate business analysis for {model_name}: {e}")
                    continue
        
        self._update_metadata("generate_business_analysis", timer.elapsed_time)
        
        logger.info(f"Business analysis complete for {len(self.business_insights)} models")
        
        return self.business_insights
    
    def get_best_model_recommendation(self) -> Dict[str, Any]:
        """
        Get recommendation for the best model based on comprehensive analysis.
        
        Returns:
        --------
        Dict[str, Any]
            Best model recommendation with detailed analysis
        """
        if not self.business_insights:
            raise ValueError("Business analysis not completed. Call generate_business_analysis() first.")
        
        logger.info("Determining best model recommendation")
        
        # Score models based on multiple criteria
        model_scores = {}
        
        for model_name in self.business_insights.keys():
            try:
                # Get key metrics
                evaluation = self.evaluation_results[model_name]
                validation = self.validation_results.get(model_name, {})
                business = self.business_insights[model_name]
                
                # Scoring criteria
                test_r2 = evaluation['test_metrics'].get('test_r2', 0)
                test_rmse = evaluation['test_metrics'].get('test_rmse', float('inf'))
                overfitting = evaluation['overfitting_metrics'].get('overfitting_r2', 1)
                roi_percentage = business['business_impact']['roi']['roi_percentage']
                
                # Deployment readiness
                deployment_status = validation.get('overall_assessment', {}).get('readiness_status', 'NOT_READY')
                deployment_score = {'READY': 100, 'CONDITIONAL': 75, 'NOT_READY': 0}.get(deployment_status, 0)
                
                # Calculate composite score
                performance_score = min(test_r2 * 100, 100)  # R² as percentage, capped at 100
                robustness_score = max(0, 100 - (overfitting * 500))  # Penalty for overfitting
                business_score = min(max(roi_percentage, 0), 200) / 2  # ROI capped at 200%, scaled to 0-100
                
                composite_score = (
                    performance_score * 0.4 +
                    robustness_score * 0.2 +
                    business_score * 0.3 +
                    deployment_score * 0.1
                )
                
                model_scores[model_name] = {
                    'composite_score': composite_score,
                    'performance_score': performance_score,
                    'robustness_score': robustness_score,
                    'business_score': business_score,
                    'deployment_score': deployment_score,
                    'test_r2': test_r2,
                    'test_rmse': test_rmse,
                    'roi_percentage': roi_percentage,
                    'deployment_status': deployment_status
                }
                
            except Exception as e:
                logger.warning(f"Failed to score model {model_name}: {e}")
                continue
        
        if not model_scores:
            raise ValueError("No models could be scored for recommendation")
        
        # Find best model
        best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['composite_score'])
        best_model_info = model_scores[best_model_name]
        
        recommendation = {
            'recommended_model': best_model_name,
            'recommendation_score': best_model_info['composite_score'],
            'model_scores': model_scores,
            'detailed_analysis': {
                'evaluation': self.evaluation_results[best_model_name],
                'validation': self.validation_results.get(best_model_name, {}),
                'business_insights': self.business_insights[best_model_name]
            },
            'recommendation_summary': {
                'performance': f"R² = {best_model_info['test_r2']:.4f}, RMSE = {best_model_info['test_rmse']:.4f}",
                'business_impact': f"ROI = {best_model_info['roi_percentage']:.1f}%",
                'deployment_readiness': best_model_info['deployment_status'],
                'overall_assessment': self._get_recommendation_message(best_model_info)
            }
        }
        
        logger.info(f"Best model recommendation: {best_model_name}")
        logger.info(f"Composite score: {best_model_info['composite_score']:.1f}/100")
        
        return recommendation
    
    def run_complete_pipeline(self,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_test: pd.DataFrame,
                             y_test: pd.Series,
                             model_names: Optional[List[str]] = None,
                             custom_params: Optional[Dict[str, Dict[str, Any]]] = None,
                             annual_equipment_count: int = 10,
                             annual_maintenance_budget: float = 500000.0) -> Dict[str, Any]:
        """
        Run the complete ML pipeline in one call.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature matrix
        y_train : pd.Series
            Training target variable
        X_test : pd.DataFrame
            Test feature matrix
        y_test : pd.Series
            Test target variable
        model_names : Optional[List[str]], default=None
            Models to train
        custom_params : Optional[Dict[str, Dict[str, Any]]], default=None
            Custom model parameters
        annual_equipment_count : int, default=10
            Equipment count for business analysis
        annual_maintenance_budget : float, default=500000.0
            Maintenance budget for business analysis
            
        Returns:
        --------
        Dict[str, Any]
            Complete pipeline results including best model recommendation
        """
        logger.info("Starting complete ML pipeline")
        
        pipeline_start_time = Timer("Complete ML pipeline")
        pipeline_start_time.__enter__()
        
        try:
            # Step 1: Setup data
            self.setup_data(X_train, y_train, X_test, y_test)
            
            # Step 2: Train models
            training_results = self.train_models(
                model_names=model_names,
                custom_params=custom_params
            )
            
            # Step 3: Evaluate models
            evaluation_results = self.evaluate_models()
            
            # Step 4: Validate models
            validation_results = self.validate_models()
            
            # Step 5: Generate business analysis
            business_results = self.generate_business_analysis(
                annual_equipment_count=annual_equipment_count,
                annual_maintenance_budget=annual_maintenance_budget
            )
            
            # Step 6: Get best model recommendation
            recommendation = self.get_best_model_recommendation()
            
            # Step 7: Generate enhanced visualization plots
            enhanced_plots = None
            try:
                from evaluation.enhanced_plotter import create_enhanced_evaluation_plots
                
                logger.info("Creating enhanced evaluation plots")
                enhanced_plots = create_enhanced_evaluation_plots(
                    evaluation_results=evaluation_results,
                    y_train=y_train,
                    y_test=y_test,
                    X_train=X_train,
                    X_test=X_test,
                    model_registry=self.registry,
                    output_dir=None,  # Will use default plots directory
                    top_n=5,
                    save_plots=True
                )
                logger.info("Enhanced evaluation plots created successfully")
            except Exception as e:
                logger.warning(f"Failed to create enhanced plots: {e}")
                enhanced_plots = {'error': str(e)}
            
            # Compile complete results
            complete_results = {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'validation_results': validation_results,
                'business_analysis': business_results,
                'recommendation': recommendation,
                'enhanced_plots': enhanced_plots,
                'pipeline_metadata': self.get_pipeline_summary()
            }
            
        except Exception as e:
            logger.error(f"ML Pipeline failed: {e}")
            raise
        
        finally:
            pipeline_start_time.__exit__(None, None, None)
            total_time = pipeline_start_time.elapsed_time
            self.pipeline_metadata['total_execution_time'] = total_time
        
        logger.info(f"Complete ML pipeline finished in {total_time:.2f}s")
        
        return complete_results
    
    def save_pipeline_results(self, output_dir: Union[str, Path]) -> None:
        """
        Save complete pipeline results to disk.
        
        Parameters:
        -----------
        output_dir : Union[str, Path]
            Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save evaluation results
        if self.evaluation_results:
            with open(output_dir / "evaluation_results.json", 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json.dump(self._serialize_for_json(self.evaluation_results), f, indent=2)
        
        # Save validation results
        if self.validation_results:
            with open(output_dir / "validation_results.json", 'w') as f:
                json.dump(self._serialize_for_json(self.validation_results), f, indent=2)
        
        # Save business insights
        if self.business_insights:
            with open(output_dir / "business_insights.json", 'w') as f:
                json.dump(self._serialize_for_json(self.business_insights), f, indent=2)
        
        # Save pipeline metadata
        with open(output_dir / "pipeline_metadata.json", 'w') as f:
            json.dump(self._serialize_for_json(self.pipeline_metadata), f, indent=2)
        
        # Save best model recommendation if available
        if self.models_evaluated and self.business_insights:
            try:
                recommendation = self.get_best_model_recommendation()
                with open(output_dir / "model_recommendation.json", 'w') as f:
                    json.dump(self._serialize_for_json(recommendation), f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save model recommendation: {e}")
        
        logger.info(f"Pipeline results saved to {output_dir}")
    
    def _serialize_for_json(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable types."""
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
    
    def _get_recommendation_message(self, model_info: Dict[str, Any]) -> str:
        """Generate recommendation message based on model performance."""
        score = model_info['composite_score']
        
        if score >= 85:
            return "Excellent model - highly recommended for production deployment"
        elif score >= 70:
            return "Good model - suitable for production with monitoring"
        elif score >= 50:
            return "Fair model - consider improvements before deployment"
        else:
            return "Poor model - significant improvements required"
    
    def _update_metadata(self, step_name: str, execution_time: float):
        """Update pipeline metadata."""
        self.pipeline_metadata['steps_completed'].append(step_name)
        self.pipeline_metadata['execution_times'][step_name] = execution_time
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline execution summary.
        
        Returns:
        --------
        Dict[str, Any]
            Pipeline execution summary
        """
        summary = {
            'steps_completed': len(self.pipeline_metadata['steps_completed']),
            'total_steps': 6,  # setup, train, evaluate, validate, analyze, recommend
            'execution_summary': self.pipeline_metadata.copy(),
            'models_trained': len(self.trained_models),
            'models_evaluated': len(self.evaluation_results),
            'models_validated': len(self.validation_results),
            'business_analyses_completed': len(self.business_insights)
        }
        
        if self.best_model_name:
            summary['best_model'] = self.best_model_name
            summary['best_model_metrics'] = self.pipeline_metadata['best_model_metrics']
        
        return summary


def create_ml_pipeline(config: Optional[Dict[str, Any]] = None,
                      random_state: int = 42,
                      registry: Optional[ModelRegistry] = None) -> MLPipeline:
    """
    Create an ML pipeline with optional configuration.
    
    Parameters:
    -----------
    config : Optional[Dict[str, Any]], default=None
        Configuration for pipeline
    random_state : int, default=42
        Random state for reproducibility
    registry : Optional[ModelRegistry], default=None
        Model registry to use
        
    Returns:
    --------
    MLPipeline
        Configured ML pipeline instance
    """
    return MLPipeline(config=config, random_state=random_state, registry=registry)