"""
Tests for the data processing module.
"""
import pandas as pd
import pytest
from pathlib import Path

from src.data.data_processor import DataProcessor

@pytest.fixture
def sample_config():
    return {
        'features': {
            'target': 'y',
            'numeric': ['age', 'campaign', 'pdays', 'previous'],
            'categorical': [
                'job', 'marital', 'education', 'default', 'housing',
                'loan', 'contact', 'month', 'poutcome'
            ],
            'drop': [
                'duration', 'emp.var.rate', 'cons.price.idx',
                'cons.conf.idx', 'euribor3m', 'nr.employed'
            ]
        }
    }

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'age': [25, 35, 45],
        'job': ['admin.', 'blue-collar', 'technician'],
        'marital': ['married', 'single', 'divorced'],
        'education': ['secondary', 'primary', 'tertiary'],
        'default': ['no', 'no', 'yes'],
        'housing': ['yes', 'no', 'yes'],
        'loan': ['no', 'yes', 'no'],
        'contact': ['telephone', 'cellular', 'telephone'],
        'month': ['may', 'jun', 'jul'],
        'campaign': [1, 2, 3],
        'pdays': [999, 6, 999],
        'previous': [0, 1, 2],
        'poutcome': ['unknown', 'success', 'failure'],
        'emp.var.rate': [-1.8, 1.1, 1.4],
        'cons.price.idx': [92.893, 93.994, 93.444],
        'cons.conf.idx': [-46.2, -36.4, -41.8],
        'euribor3m': [1.313, 4.857, 4.962],
        'nr.employed': [5099.1, 5191.0, 5228.1],
        'y': ['no', 'yes', 'no']
    })

@pytest.fixture
def data_processor(sample_config):
    return DataProcessor(sample_config)

def test_data_processor_initialization(data_processor, sample_config):
    """Test DataProcessor initialization."""
    assert data_processor.target == sample_config['features']['target']
    assert data_processor.numeric_features == sample_config['features']['numeric']
    assert data_processor.categorical_features == sample_config['features']['categorical']
    assert data_processor.drop_features == sample_config['features']['drop']

def test_process_data(data_processor, sample_data):
    """Test data processing functionality."""
    processed_data = data_processor.process_data(sample_data)
    
    # Check that target is first column
    assert processed_data.columns[0] == 'y'
    
    # Check that dropped columns are not present
    for col in data_processor.drop_features:
        assert col not in processed_data.columns
    
    # Check that derived features are present
    assert 'no_previous_contact' in processed_data.columns
    assert 'not_working' in processed_data.columns
    
    # Check that age ranges are created
    age_range_cols = [col for col in processed_data.columns if col.startswith('age_')]
    assert len(age_range_cols) > 0
    
    # Check that categorical variables are one-hot encoded
    for cat_feature in data_processor.categorical_features:
        encoded_cols = [col for col in processed_data.columns if col.startswith(f"{cat_feature}_")]
        assert len(encoded_cols) > 0

def test_split_data(data_processor, sample_data):
    """Test data splitting functionality."""
    processed_data = data_processor.process_data(sample_data)
    train_data, val_data, test_data = data_processor.split_data(
        processed_data,
        train_size=0.7,
        val_size=0.2
    )
    
    # Check split sizes
    total_samples = len(processed_data)
    assert len(train_data) == pytest.approx(total_samples * 0.7, abs=1)
    assert len(val_data) == pytest.approx(total_samples * 0.2, abs=1)
    assert len(test_data) == pytest.approx(total_samples * 0.1, abs=1)
    
    # Check that splits have same columns
    assert all(train_data.columns == val_data.columns)
    assert all(train_data.columns == test_data.columns)

def test_save_data(data_processor, sample_data, tmp_path):
    """Test data saving functionality."""
    processed_data = data_processor.process_data(sample_data)
    train_data, val_data, test_data = data_processor.split_data(processed_data)
    
    data_processor.save_data(
        train_data,
        val_data,
        test_data,
        tmp_path
    )
    
    # Check that files are created
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "validation.csv").exists()
    assert (tmp_path / "test.csv").exists()
    assert (tmp_path / "baseline.csv").exists()
    
    # Check that saved data can be loaded
    saved_train = pd.read_csv(tmp_path / "train.csv")
    assert all(saved_train.columns == train_data.columns)
    assert len(saved_train) == len(train_data)
