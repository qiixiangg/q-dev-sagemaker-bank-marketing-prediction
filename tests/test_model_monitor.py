"""
Tests for the model monitoring module.
"""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.monitoring.model_monitor import ModelMonitor

@pytest.fixture
def sample_config():
    return {
        'model_registry': {
            'metadata': {
                'threshold': 0.75
            }
        }
    }

@pytest.fixture
def sample_baseline_data():
    """Create synthetic baseline data."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(5, 2, n_samples),
        'feature_3': np.random.exponential(2, n_samples)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_drift_data():
    """Create synthetic data with drift."""
    np.random.seed(43)
    n_samples = 1000
    
    data = {
        'feature_1': np.random.normal(0.5, 1.2, n_samples),  # Shifted mean and variance
        'feature_2': np.random.normal(5, 2, n_samples),      # No drift
        'feature_3': np.random.exponential(3, n_samples)     # Different rate parameter
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def model_monitor(sample_config):
    return ModelMonitor(sample_config)

def test_model_monitor_initialization(model_monitor):
    """Test ModelMonitor initialization."""
    assert model_monitor.config is not None
    assert model_monitor.baseline_stats is None
    assert model_monitor.current_stats is None

def test_compute_statistics(model_monitor, sample_baseline_data):
    """Test statistics computation."""
    stats = model_monitor.compute_statistics(sample_baseline_data)
    
    # Check that stats are computed for each feature
    for column in sample_baseline_data.columns:
        assert column in stats
        
        # Check that all required statistics are present
        required_stats = ['mean', 'std', 'median', 'q1', 'q3', 'min', 'max', 'missing_pct']
        for stat in required_stats:
            assert stat in stats[column]
            assert isinstance(stats[column][stat], (int, float))

def test_load_baseline(model_monitor, sample_baseline_data, tmp_path):
    """Test baseline data loading."""
    # Save baseline data
    baseline_path = tmp_path / "baseline.csv"
    sample_baseline_data.to_csv(baseline_path, index=False)
    
    # Load baseline
    model_monitor.load_baseline(baseline_path)
    
    # Check that baseline statistics are computed
    assert model_monitor.baseline_stats is not None
    assert isinstance(model_monitor.baseline_stats, dict)
    assert all(col in model_monitor.baseline_stats for col in sample_baseline_data.columns)

def test_detect_drift(model_monitor, sample_baseline_data, sample_drift_data):
    """Test drift detection."""
    # Compute baseline stats
    model_monitor.baseline_stats = model_monitor.compute_statistics(sample_baseline_data)
    
    # Detect drift
    drift_stats, drifted_features = model_monitor.detect_drift(
        sample_drift_data,
        threshold=0.05
    )
    
    # Check drift statistics format
    assert isinstance(drift_stats, dict)
    assert isinstance(drifted_features, list)
    
    # Check that drift is detected in features with significant changes
    assert 'feature_1' in drifted_features  # Should detect drift due to shifted distribution
    assert 'feature_2' not in drifted_features  # Should not detect drift in unchanged feature
    assert 'feature_3' in drifted_features  # Should detect drift due to different rate

def test_monitor_performance(model_monitor):
    """Test performance monitoring."""
    # Create synthetic predictions
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.random(n_samples)  # Random probabilities
    
    # Monitor performance
    performance_metrics = model_monitor.monitor_performance(
        y_true,
        y_pred,
        threshold=0.75
    )
    
    # Check metrics format
    assert isinstance(performance_metrics, dict)
    assert 'auc_score' in performance_metrics
    assert 'threshold' in performance_metrics
    assert 'below_threshold' in performance_metrics
    
    # Check metric values
    assert 0 <= performance_metrics['auc_score'] <= 1
    assert isinstance(performance_metrics['below_threshold'], bool)

def test_save_monitoring_report(model_monitor, tmp_path):
    """Test monitoring report saving."""
    # Create sample monitoring results
    drift_stats = {
        'feature_1': {'is_drifted': True, 'p_value': 0.01},
        'feature_2': {'is_drifted': False, 'p_value': 0.8}
    }
    
    performance_metrics = {
        'auc_score': 0.85,
        'threshold': 0.75,
        'below_threshold': False
    }
    
    # Save report
    model_monitor.save_monitoring_report(
        drift_stats,
        performance_metrics,
        tmp_path
    )
    
    # Check that report file is created
    report_path = tmp_path / 'monitoring_report.json'
    assert report_path.exists()
    
    # Check report content
    with open(report_path) as f:
        report = json.load(f)
        assert 'timestamp' in report
        assert 'drift_statistics' in report
        assert 'performance_metrics' in report
        assert 'needs_retraining' in report
        
        # Check that needs_retraining is correctly determined
        assert report['needs_retraining'] == True  # Should be True due to drift in feature_1
