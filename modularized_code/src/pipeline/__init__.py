"""
Pipeline Integration Module
===========================

End-to-end pipeline orchestration for vibration prediction.
Provides complete data processing and ML pipeline integration.

Key Components:
--------------
- DataPipeline: Complete data processing pipeline
- MLPipeline: End-to-end machine learning pipeline
- PipelineOrchestrator: Orchestrates complete workflow

Features:
---------
- Data loading and preprocessing automation
- Feature engineering pipeline
- Model training and evaluation automation
- Result persistence and reporting
- Error handling and recovery
- Progress tracking and logging
"""

from .data_pipeline import DataPipeline, create_data_pipeline
from .ml_pipeline import MLPipeline, create_ml_pipeline
from .orchestrator import PipelineOrchestrator, run_complete_pipeline

__all__ = [
    # Classes
    "DataPipeline",
    "MLPipeline",
    "PipelineOrchestrator",
    
    # Convenience functions
    "create_data_pipeline",
    "create_ml_pipeline", 
    "run_complete_pipeline",
]