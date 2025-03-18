"""
Model monitoring module for the ML pipeline.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from ..utils.logger import get_logger

logger = get_logger(__name__)

class ModelMonitor:
    """Handles model monitoring and drift detection."""
    
    def __init__(self, config: Dict):
        """
        Initialize ModelMonitor with configuration.
        
        Args:
            config: Configuration dictionary containing monitoring parameters
        """
        self.config = config
        self.baseline_stats = None
        self.current_stats = None
        
    def compute_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute statistical measures for numerical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing statistics for each feature
        """
        stats_dict = {}
        
        for column in df.columns:
            column_stats = {
                'mean': df[column].mean(),
                'std': df[column].std(),
                'median': df[column].median(),
                'q1': df[column].quantile(0.25),
                'q3': df[column].quantile(0.75),
                'min': df[column].min(),
                'max': df[column].max(),
                'missing_pct': (df[column].isnull().sum() / len(df)) * 100
            }
            stats_dict[column] = column_stats
            
        return stats_dict
        
    def load_baseline(self, baseline_path: str) -> None:
        """
        Load and compute statistics for baseline data.
        
        Args:
            baseline_path: Path to baseline data
        """
        try:
            baseline_data = pd.read_csv(baseline_path)
            self.baseline_stats = self.compute_statistics(baseline_data)
            logger.info(f"Loaded baseline statistics from {baseline_path}")
            
        except Exception as e:
            logger.error(f"Error loading baseline data: {str(e)}")
            raise
            
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.05
    ) -> Tuple[Dict, List[str]]:
        """
        Detect data drift between baseline and current data.
        
        Args:
            current_data: Current data to compare against baseline
            threshold: P-value threshold for drift detection
            
        Returns:
            Tuple of (drift statistics, list of drifted features)
        """
        try:
            if self.baseline_stats is None:
                raise ValueError("Baseline statistics not loaded")
                
            self.current_stats = self.compute_statistics(current_data)
            drift_stats = {}
            drifted_features = []
            
            for feature in self.baseline_stats.keys():
                if feature not in current_data.columns:
                    continue
                    
                # Get baseline and current values for comparison
                baseline_values = pd.Series([
                    self.baseline_stats[feature]['mean'],
                    self.baseline_stats[feature]['median'],
                    self.baseline_stats[feature]['q1'],
                    self.baseline_stats[feature]['q3']
                ])
                
                current_values = pd.Series([
                    self.current_stats[feature]['mean'],
                    self.current_stats[feature]['median'],
                    self.current_stats[feature]['q1'],
                    self.current_stats[feature]['q3']
                ])
                
                # Perform Kolmogorov-Smirnov test
                ks_statistic, p_value = stats.ks_2samp(
                    current_data[feature].dropna(),
                    baseline_values
                )
                
                # Calculate percentage change in key statistics
                pct_change_mean = abs(
                    (self.current_stats[feature]['mean'] - self.baseline_stats[feature]['mean'])
                    / self.baseline_stats[feature]['mean']
                ) * 100
                
                pct_change_std = abs(
                    (self.current_stats[feature]['std'] - self.baseline_stats[feature]['std'])
                    / self.baseline_stats[feature]['std']
                ) * 100
                
                # Check for drift based on statistical test and significant changes
                is_drifted = (
                    p_value < threshold or 
                    pct_change_mean > 50 or    # More significant mean change threshold
                    pct_change_std > 100       # More significant std change threshold
                )
                
                drift_stats[feature] = {
                    'ks_statistic': float(ks_statistic),
                    'p_value': float(p_value),
                    'pct_change_mean': float(pct_change_mean),
                    'pct_change_std': float(pct_change_std),
                    'is_drifted': is_drifted
                }
                
                if is_drifted:
                    drifted_features.append(feature)
                    logger.warning(
                        f"Detected drift in feature {feature} "
                        f"(p-value: {p_value:.4f}, "
                        f"mean_change: {pct_change_mean:.2f}%, "
                        f"std_change: {pct_change_std:.2f}%)"
                    )
                    
            return drift_stats, drifted_features
            
        except Exception as e:
            logger.error(f"Error detecting drift: {str(e)}")
            raise
            
    def monitor_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.75
    ) -> Dict:
        """
        Monitor model performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            threshold: Performance threshold
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            auc_score = roc_auc_score(y_true, y_pred)
            
            performance_metrics = {
                'auc_score': auc_score,
                'threshold': threshold,
                'below_threshold': auc_score < threshold
            }
            
            if auc_score < threshold:
                logger.warning(
                    f"Model performance below threshold: {auc_score:.4f} "
                    f"(threshold: {threshold})"
                )
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error monitoring performance: {str(e)}")
            raise
            
    def save_monitoring_report(
        self,
        drift_stats: Dict,
        performance_metrics: Dict,
        output_dir: str
    ) -> None:
        """
        Save monitoring results to a report file.
        
        Args:
            drift_stats: Dictionary containing drift statistics
            performance_metrics: Dictionary containing performance metrics
            output_dir: Directory to save the report
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy types to Python native types
            def convert_to_native(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_native(x) for x in obj]
                return obj

            report = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'drift_statistics': convert_to_native(drift_stats),
                'performance_metrics': convert_to_native(performance_metrics),
                'needs_retraining': bool(
                    len([f for f in drift_stats.values() if f['is_drifted']]) > 0
                    or performance_metrics['below_threshold']
                )
            }
            
            report_path = output_dir / 'monitoring_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Saved monitoring report to {report_path}")
            
            if report['needs_retraining']:
                logger.warning(
                    "Model retraining recommended due to data drift "
                    "or performance degradation"
                )
                
        except Exception as e:
            logger.error(f"Error saving monitoring report: {str(e)}")
            raise
