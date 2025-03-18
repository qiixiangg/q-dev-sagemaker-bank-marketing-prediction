"""
Tests for the model training module.
"""
import json
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from pathlib import Path

from src.training.model_trainer import ModelTrainer

@pytest.fixture
def sample_config():
    return {
        'xgboost': {
            'params': {
                'max_depth': 3,
                'eta': 0.1,
                'objective': 'binary:logistic',
                'eval_metric': 'auc'
            },
            'training': {
                'num_boost_round': 10,
                'early_stopping_rounds': 5,
                'verbose_eval': 0
            },
            'cv': {
                'nfold': 3,
                'stratified': True,
                'seed': 42
            }
        },
        'model_registry': {
            'model_name': 'test_model',
            'version': 'v1',
            'metadata': {
                'description': 'Test model',
                'target_metric': 'auc',
                'threshold': 0.75
            }
        }
    }

@pytest.fixture
def sample_data():
    # Create synthetic binary classification data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

@pytest.fixture
def model_trainer(sample_config):
    return ModelTrainer(sample_config)

def test_model_trainer_initialization(model_trainer, sample_config):
    """Test ModelTrainer initialization."""
    assert model_trainer.model_params == sample_config['xgboost']['params']
    assert model_trainer.training_params == sample_config['xgboost']['training']
    assert model_trainer.cv_params == sample_config['xgboost']['cv']
    assert model_trainer.model is None

def test_prepare_data(model_trainer, sample_data):
    """Test data preparation for XGBoost."""
    # Split data into features and target
    X = sample_data.drop('target', axis=1)
    y = sample_data['target']
    
    # Create train/val/test splits
    train_idx = int(len(sample_data) * 0.7)
    val_idx = int(len(sample_data) * 0.9)
    
    train_data = sample_data.iloc[:train_idx]
    val_data = sample_data.iloc[train_idx:val_idx]
    test_data = sample_data.iloc[val_idx:]
    
    # Prepare data
    dtrain, dval, dtest = model_trainer.prepare_data(
        train_data,
        val_data,
        test_data,
        target_col='target'
    )
    
    # Check that DMatrix objects are created correctly
    assert isinstance(dtrain, xgb.DMatrix)
    assert isinstance(dval, xgb.DMatrix)
    assert isinstance(dtest, xgb.DMatrix)
    
    # Check dimensions
    assert dtrain.num_row() == len(train_data)
    assert dval.num_row() == len(val_data)
    assert dtest.num_row() == len(test_data)

def test_train_with_cv(model_trainer, sample_data):
    """Test cross-validation training."""
    # Prepare data
    X = sample_data.drop('target', axis=1)
    y = sample_data['target']
    dtrain = xgb.DMatrix(X, label=y)
    
    # Run cross-validation
    cv_results = model_trainer.train_with_cv(dtrain)
    
    # Check results format
    assert isinstance(cv_results, pd.DataFrame)
    assert 'train-auc-mean' in cv_results.columns
    assert 'test-auc-mean' in cv_results.columns
    assert len(cv_results) > 0

def test_train_and_evaluate(model_trainer, sample_data):
    """Test model training and evaluation."""
    # Prepare data
    train_data = sample_data.iloc[:80]
    test_data = sample_data.iloc[80:]
    
    dtrain, _, dtest = model_trainer.prepare_data(
        train_data,
        test_data=test_data,
        target_col='target'
    )
    
    # Train model
    model = model_trainer.train(dtrain)
    assert model is not None
    assert isinstance(model, xgb.Booster)
    
    # Evaluate model
    auc_score, predictions, cm = model_trainer.evaluate(
        dtest,
        test_data['target']
    )
    
    # Check evaluation results
    assert isinstance(auc_score, float)
    assert 0 <= auc_score <= 1
    assert len(predictions) == len(test_data)
    assert cm.shape == (2, 2)  # Binary classification

def test_save_model(model_trainer, sample_data, tmp_path):
    """Test model saving functionality."""
    # Prepare and train model
    dtrain, _, _ = model_trainer.prepare_data(
        sample_data,
        target_col='target'
    )
    model = model_trainer.train(dtrain)
    
    # Save model
    model_trainer.save_model(tmp_path)
    
    # Check that files are created
    model_path = tmp_path / f"{model_trainer.config['model_registry']['model_name']}.json"
    metadata_path = tmp_path / 'metadata.json'
    
    assert model_path.exists()
    assert metadata_path.exists()
    
    # Check metadata content
    with open(metadata_path) as f:
        metadata = json.load(f)
        assert metadata['model_name'] == model_trainer.config['model_registry']['model_name']
        assert metadata['version'] == model_trainer.config['model_registry']['version']
        assert metadata['params'] == model_trainer.model_params
