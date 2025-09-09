"""
Business Analysis for Model Evaluation
======================================

Generate business insights, recommendations, and deployment assessments
from model evaluation results.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from config.settings import get_settings
from .metrics import ModelMetrics

logger = logging.getLogger(__name__)
settings = get_settings()


class BusinessAnalyzer:
    """
    Generate business insights and recommendations from model evaluation.
    
    This class analyzes model performance from a business perspective,
    providing deployment recommendations and operational insights.
    """
    
    def __init__(self, 
                 business_thresholds: Optional[Dict[str, float]] = None,
                 industry_context: str = "industrial_vibration"):
        """
        Initialize business analyzer.
        
        Parameters:
        -----------
        business_thresholds : Optional[Dict[str, float]], default=None
            Business-specific thresholds for evaluation
        industry_context : str, default="industrial_vibration"
            Industry context for recommendations
        """
        self.industry_context = industry_context
        
        # Default business thresholds for industrial vibration monitoring
        self.business_thresholds = business_thresholds or {
            'excellent_r2': 0.90,              # Excellent performance threshold
            'good_r2': 0.80,                   # Good performance threshold
            'acceptable_r2': 0.60,             # Minimum acceptable performance
            'max_production_rmse': 0.005,      # Maximum RMSE for production
            'max_acceptable_overfitting': 0.15, # Maximum overfitting for deployment
            'min_accuracy_within_tolerance': 75.0, # Minimum accuracy within tolerance
            'cost_per_false_alarm': 100.0,     # Cost of false alarm ($)
            'cost_per_missed_detection': 1000.0, # Cost of missed detection ($)
            'maintenance_cost_savings': 0.15,   # Potential maintenance cost savings (15%)
            'equipment_downtime_cost_per_hour': 5000.0 # Downtime cost ($/hour)
        }
        
        logger.debug("BusinessAnalyzer initialized")
    
    def assess_performance_level(self, 
                                test_r2: float,
                                test_rmse: float) -> Dict[str, Any]:
        """
        Assess model performance level from business perspective.
        
        Parameters:
        -----------
        test_r2 : float
            Test R² score
        test_rmse : float
            Test RMSE
            
        Returns:
        --------
        Dict[str, Any]
            Performance assessment results
        """
        # Determine performance level
        if test_r2 >= self.business_thresholds['excellent_r2']:
            performance_level = "Excellent"
            business_confidence = "High confidence for production deployment"
            deployment_recommendation = "DEPLOY_IMMEDIATELY"
        elif test_r2 >= self.business_thresholds['good_r2']:
            performance_level = "Good"
            business_confidence = "Suitable for production with monitoring"
            deployment_recommendation = "DEPLOY_WITH_MONITORING"
        elif test_r2 >= self.business_thresholds['acceptable_r2']:
            performance_level = "Fair"
            business_confidence = "Requires improvement before production"
            deployment_recommendation = "IMPROVE_BEFORE_DEPLOY"
        else:
            performance_level = "Poor"
            business_confidence = "Not suitable for production deployment"
            deployment_recommendation = "NOT_SUITABLE"
        
        # RMSE assessment
        rmse_acceptable = test_rmse <= self.business_thresholds['max_production_rmse']
        
        assessment = {
            'performance_level': performance_level,
            'business_confidence': business_confidence,
            'deployment_recommendation': deployment_recommendation,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'rmse_acceptable': rmse_acceptable,
            'rmse_threshold': self.business_thresholds['max_production_rmse']
        }
        
        return assessment
    
    def calculate_business_impact(self,
                                 test_metrics: Dict[str, float],
                                 baseline_metrics: Optional[Dict[str, float]] = None,
                                 annual_equipment_count: int = 10,
                                 annual_maintenance_budget: float = 500000.0) -> Dict[str, Any]:
        """
        Calculate business impact and ROI of the model.
        
        Parameters:
        -----------
        test_metrics : Dict[str, float]
            Test performance metrics
        baseline_metrics : Optional[Dict[str, float]], default=None
            Baseline model metrics for comparison
        annual_equipment_count : int, default=10
            Number of equipment units monitored annually
        annual_maintenance_budget : float, default=500000.0
            Annual maintenance budget ($)
            
        Returns:
        --------
        Dict[str, Any]
            Business impact analysis
        """
        logger.info("Calculating business impact and ROI")
        
        r2_score = test_metrics.get('test_r2', 0)
        rmse = test_metrics.get('test_rmse', float('inf'))
        
        # Calculate accuracy-based savings
        accuracy_improvement = r2_score  # Assume improvement over no-prediction baseline
        if baseline_metrics and 'test_r2' in baseline_metrics:
            accuracy_improvement = max(0, r2_score - baseline_metrics['test_r2'])
        
        # Estimate maintenance cost savings
        potential_savings_rate = min(accuracy_improvement * self.business_thresholds['maintenance_cost_savings'], 0.3)
        annual_maintenance_savings = annual_maintenance_budget * potential_savings_rate
        
        # Estimate downtime reduction
        # Better predictions lead to better maintenance scheduling
        estimated_downtime_reduction_hours = accuracy_improvement * 24 * annual_equipment_count  # Hours per year
        downtime_cost_savings = (estimated_downtime_reduction_hours * 
                                self.business_thresholds['equipment_downtime_cost_per_hour'])
        
        # False alarm and missed detection costs
        # Estimate based on RMSE - lower RMSE means fewer false alarms
        false_alarm_rate = min(rmse / 0.01, 0.5)  # Higher RMSE leads to more false alarms
        missed_detection_rate = max(0.1 - r2_score, 0.01)  # Lower R² leads to more missed detections
        
        annual_false_alarm_cost = (false_alarm_rate * annual_equipment_count * 52 * 
                                  self.business_thresholds['cost_per_false_alarm'])
        annual_missed_detection_cost = (missed_detection_rate * annual_equipment_count * 12 * 
                                       self.business_thresholds['cost_per_missed_detection'])
        
        # Total annual savings
        total_annual_savings = (annual_maintenance_savings + 
                               downtime_cost_savings - 
                               annual_false_alarm_cost - 
                               annual_missed_detection_cost)
        
        # Implementation costs (estimate)
        estimated_implementation_cost = 150000.0  # Initial setup and training
        estimated_annual_operating_cost = 30000.0  # Ongoing costs
        
        # ROI calculation
        net_annual_benefit = total_annual_savings - estimated_annual_operating_cost
        roi_percentage = (net_annual_benefit / estimated_implementation_cost) * 100 if estimated_implementation_cost > 0 else 0
        
        business_impact = {
            'performance_metrics': {
                'r2_score': r2_score,
                'rmse': rmse,
                'accuracy_improvement': accuracy_improvement
            },
            'cost_savings': {
                'annual_maintenance_savings': annual_maintenance_savings,
                'downtime_cost_savings': downtime_cost_savings,
                'total_annual_savings': total_annual_savings
            },
            'cost_avoidance': {
                'false_alarm_cost': annual_false_alarm_cost,
                'missed_detection_cost': annual_missed_detection_cost,
                'total_cost_avoidance': annual_false_alarm_cost + annual_missed_detection_cost
            },
            'investment': {
                'implementation_cost': estimated_implementation_cost,
                'annual_operating_cost': estimated_annual_operating_cost,
                'net_annual_benefit': net_annual_benefit
            },
            'roi': {
                'roi_percentage': roi_percentage,
                'payback_period_months': (estimated_implementation_cost / net_annual_benefit * 12) 
                                       if net_annual_benefit > 0 else float('inf')
            }
        }
        
        return business_impact
    
    def generate_deployment_recommendations(self,
                                          performance_assessment: Dict[str, Any],
                                          business_impact: Dict[str, Any],
                                          model_name: str = "Model") -> Dict[str, Any]:
        """
        Generate comprehensive deployment recommendations.
        
        Parameters:
        -----------
        performance_assessment : Dict[str, Any]
            Performance assessment results
        business_impact : Dict[str, Any]
            Business impact analysis
        model_name : str, default="Model"
            Name of the model
            
        Returns:
        --------
        Dict[str, Any]
            Deployment recommendations
        """
        logger.info(f"Generating deployment recommendations for {model_name}")
        
        deployment_status = performance_assessment['deployment_recommendation']
        roi_percentage = business_impact['roi']['roi_percentage']
        payback_months = business_impact['roi']['payback_period_months']
        
        recommendations = {
            'model_name': model_name,
            'deployment_status': deployment_status,
            'recommendations': []
        }
        
        # Primary deployment decision
        if deployment_status == "DEPLOY_IMMEDIATELY":
            recommendations['recommendations'].extend([
                "✅ Model ready for immediate production deployment",
                f"Expected ROI: {roi_percentage:.1f}% annually",
                f"Payback period: {payback_months:.1f} months",
                "Implement real-time monitoring dashboard",
                "Set up automated alerts for prediction drift"
            ])
        
        elif deployment_status == "DEPLOY_WITH_MONITORING":
            recommendations['recommendations'].extend([
                "⚠️ Model suitable for production with enhanced monitoring",
                f"Expected ROI: {roi_percentage:.1f}% annually",
                "Implement gradual rollout with pilot equipment",
                "Establish performance monitoring thresholds",
                "Plan for model retraining every 3-6 months"
            ])
        
        elif deployment_status == "IMPROVE_BEFORE_DEPLOY":
            recommendations['recommendations'].extend([
                "🔄 Model requires improvement before production deployment",
                f"Current ROI: {roi_percentage:.1f}% (target: >100%)",
                "Collect additional training data",
                "Explore advanced feature engineering",
                "Consider ensemble methods or hyperparameter optimization"
            ])
        
        else:  # NOT_SUITABLE
            recommendations['recommendations'].extend([
                "❌ Model not suitable for production deployment",
                f"Insufficient ROI: {roi_percentage:.1f}%",
                "Fundamental model architecture review required",
                "Consider alternative modeling approaches",
                "Reassess data quality and feature engineering"
            ])
        
        # Additional recommendations based on specific metrics
        r2_score = performance_assessment['test_r2']
        rmse = performance_assessment['test_rmse']
        
        if rmse > self.business_thresholds['max_production_rmse']:
            recommendations['recommendations'].append(
                f"🎯 Improve RMSE from {rmse:.4f} to below {self.business_thresholds['max_production_rmse']:.3f}"
            )
        
        if r2_score < self.business_thresholds['good_r2']:
            recommendations['recommendations'].append(
                f"📊 Target R² improvement from {r2_score:.3f} to above {self.business_thresholds['good_r2']:.2f}"
            )
        
        # Operational recommendations
        operational_recommendations = [
            f"Monitor key performance indicators: R² > {self.business_thresholds['acceptable_r2']:.2f}, RMSE < {self.business_thresholds['max_production_rmse']:.3f}",
            "Establish automated retraining pipeline",
            "Set up data quality monitoring",
            "Document model assumptions and limitations"
        ]
        
        recommendations['operational_recommendations'] = operational_recommendations
        
        return recommendations
    
    def create_executive_summary(self,
                                performance_assessment: Dict[str, Any],
                                business_impact: Dict[str, Any],
                                feature_count: int,
                                training_samples: int,
                                model_name: str = "Vibration Prediction Model") -> Dict[str, Any]:
        """
        Create executive summary for business stakeholders.
        
        Parameters:
        -----------
        performance_assessment : Dict[str, Any]
            Performance assessment results
        business_impact : Dict[str, Any]
            Business impact analysis
        feature_count : int
            Number of features used
        training_samples : int
            Number of training samples
        model_name : str, default="Vibration Prediction Model"
            Name of the model
            
        Returns:
        --------
        Dict[str, Any]
            Executive summary
        """
        logger.info("Creating executive summary for business stakeholders")
        
        r2_score = performance_assessment['test_r2']
        performance_level = performance_assessment['performance_level']
        annual_savings = business_impact['cost_savings']['total_annual_savings']
        roi_percentage = business_impact['roi']['roi_percentage']
        payback_months = business_impact['roi']['payback_period_months']
        
        executive_summary = {
            'report_title': f"{model_name} - Business Impact Assessment",
            'generated_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': model_name,
            
            # Key Performance Indicators
            'key_metrics': {
                'prediction_accuracy': f"{r2_score*100:.1f}%",
                'performance_level': performance_level,
                'features_analyzed': feature_count,
                'data_samples_used': f"{training_samples:,}",
            },
            
            # Business Impact
            'business_impact': {
                'annual_cost_savings': f"${annual_savings:,.0f}",
                'roi_percentage': f"{roi_percentage:.1f}%",
                'payback_period': f"{payback_months:.1f} months" if payback_months != float('inf') else "Not achievable",
                'net_present_value_3yr': f"${(business_impact['investment']['net_annual_benefit'] * 3 - business_impact['investment']['implementation_cost']):,.0f}"
            },
            
            # Risk Assessment
            'risk_assessment': {
                'technical_risk': self._assess_technical_risk(performance_assessment),
                'business_risk': self._assess_business_risk(business_impact),
                'implementation_complexity': self._assess_implementation_complexity(feature_count)
            },
            
            # Strategic Recommendation
            'strategic_recommendation': self._get_strategic_recommendation(
                performance_assessment['deployment_recommendation'],
                roi_percentage
            )
        }
        
        return executive_summary
    
    def _assess_technical_risk(self, performance_assessment: Dict[str, Any]) -> str:
        """Assess technical risk level."""
        r2_score = performance_assessment['test_r2']
        
        if r2_score >= 0.9:
            return "Low - High model accuracy and reliability"
        elif r2_score >= 0.8:
            return "Medium - Good accuracy with monitoring required"
        elif r2_score >= 0.6:
            return "High - Model needs improvement before deployment"
        else:
            return "Very High - Model not suitable for production"
    
    def _assess_business_risk(self, business_impact: Dict[str, Any]) -> str:
        """Assess business risk level."""
        roi = business_impact['roi']['roi_percentage']
        
        if roi >= 200:
            return "Low - Strong financial case with high returns"
        elif roi >= 100:
            return "Medium - Positive ROI with acceptable returns"
        elif roi >= 50:
            return "High - Marginal returns, careful evaluation needed"
        else:
            return "Very High - Negative or insufficient returns"
    
    def _assess_implementation_complexity(self, feature_count: int) -> str:
        """Assess implementation complexity."""
        if feature_count <= 20:
            return "Low - Simple feature set, easy to implement"
        elif feature_count <= 50:
            return "Medium - Moderate complexity, standard implementation"
        else:
            return "High - Complex feature engineering, careful implementation needed"
    
    def _get_strategic_recommendation(self, deployment_status: str, roi_percentage: float) -> Dict[str, Any]:
        """Get strategic recommendation based on analysis."""
        if deployment_status == "DEPLOY_IMMEDIATELY" and roi_percentage >= 150:
            return {
                'decision': 'STRONGLY RECOMMEND',
                'reasoning': 'High accuracy model with strong financial returns',
                'timeline': 'Implement within 3 months',
                'investment_priority': 'High'
            }
        elif deployment_status in ["DEPLOY_IMMEDIATELY", "DEPLOY_WITH_MONITORING"] and roi_percentage >= 100:
            return {
                'decision': 'RECOMMEND',
                'reasoning': 'Good model performance with positive ROI',
                'timeline': 'Implement within 6 months',
                'investment_priority': 'Medium-High'
            }
        elif roi_percentage >= 50:
            return {
                'decision': 'CONDITIONAL APPROVAL',
                'reasoning': 'Marginal returns, proceed with caution',
                'timeline': 'Pilot project in 6-12 months',
                'investment_priority': 'Medium'
            }
        else:
            return {
                'decision': 'NOT RECOMMENDED',
                'reasoning': 'Insufficient returns or poor model performance',
                'timeline': 'Reassess in 12 months',
                'investment_priority': 'Low'
            }


def generate_business_insights(y_train: Union[pd.Series, np.ndarray],
                              train_pred: np.ndarray,
                              y_test: Union[pd.Series, np.ndarray],
                              test_pred: np.ndarray,
                              model_name: str = "Model",
                              feature_count: Optional[int] = None,
                              annual_equipment_count: int = 10,
                              annual_maintenance_budget: float = 500000.0) -> Dict[str, Any]:
    """
    Generate comprehensive business insights and recommendations.
    
    Parameters:
    -----------
    y_train : Union[pd.Series, np.ndarray]
        True training values
    train_pred : np.ndarray
        Training predictions
    y_test : Union[pd.Series, np.ndarray]
        True test values
    test_pred : np.ndarray
        Test predictions
    model_name : str, default="Model"
        Name of the model
    feature_count : Optional[int], default=None
        Number of features used
    annual_equipment_count : int, default=10
        Number of equipment units
    annual_maintenance_budget : float, default=500000.0
        Annual maintenance budget
        
    Returns:
    --------
    Dict[str, Any]
        Complete business analysis
    """
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_squared_error
    
    test_metrics = {
        'test_r2': r2_score(y_test, test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
    }
    
    # Initialize analyzer
    analyzer = BusinessAnalyzer()
    
    # Perform analysis
    performance_assessment = analyzer.assess_performance_level(
        test_metrics['test_r2'], test_metrics['test_rmse']
    )
    
    business_impact = analyzer.calculate_business_impact(
        test_metrics, 
        annual_equipment_count=annual_equipment_count,
        annual_maintenance_budget=annual_maintenance_budget
    )
    
    deployment_recommendations = analyzer.generate_deployment_recommendations(
        performance_assessment, business_impact, model_name
    )
    
    executive_summary = analyzer.create_executive_summary(
        performance_assessment, 
        business_impact,
        feature_count or 0,
        len(y_train),
        model_name
    )
    
    # Compile complete analysis
    business_analysis = {
        'performance_assessment': performance_assessment,
        'business_impact': business_impact,
        'deployment_recommendations': deployment_recommendations,
        'executive_summary': executive_summary,
        'analysis_metadata': {
            'model_name': model_name,
            'analysis_date': datetime.now().isoformat(),
            'training_samples': len(y_train),
            'test_samples': len(y_test),
            'feature_count': feature_count
        }
    }
    
    logger.info(f"Business analysis complete for {model_name}")
    logger.info(f"Performance: {performance_assessment['performance_level']}")
    logger.info(f"ROI: {business_impact['roi']['roi_percentage']:.1f}%")
    
    return business_analysis