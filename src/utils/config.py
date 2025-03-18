"""
Configuration settings for the ML pipeline.
"""
from pathlib import Path

# Data Configuration
DATA_CONFIG = {
    "raw_data": {
        "url": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
        "local_path": "data/raw",
        "filename": "bank-additional-full.csv"
    },
    "processed_data": {
        "path": "data/processed",
        "train_file": "train.csv",
        "val_file": "validation.csv",
        "test_file": "test.csv",
        "baseline_file": "baseline.csv"
    },
    "features": {
        "target": "y",
        "numeric": [
            "age",
            "campaign",
            "pdays",
            "previous"
        ],
        "categorical": [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "poutcome"
        ],
        "drop": [
            "duration",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed"
        ]
    },
    "split": {
        "train_size": 0.7,
        "val_size": 0.2,
        "test_size": 0.1,
        "random_state": 1729
    }
}

# Model Configuration
MODEL_CONFIG = {
    "xgboost": {
        "params": {
            "max_depth": 5,
            "eta": 0.5,
            "alpha": 2.5,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "tree_method": "auto",
            "predictor": "auto"
        },
        "training": {
            "num_boost_round": 150,
            "early_stopping_rounds": 10,
            "verbose_eval": 10
        },
        "cv": {
            "nfold": 3,
            "stratified": True,
            "shuffle": True,
            "seed": 42
        }
    },
    "model_registry": {
        "save_path": "models",
        "model_name": "xgboost_model",
        "version": "v1",
        "metadata": {
            "description": "XGBoost model for bank marketing prediction",
            "target_metric": "auc",
            "threshold": 0.75
        }
    },
    "inference": {
        "batch_size": 1000,
        "prediction_threshold": 0.5
    }
}

def get_data_config():
    """Get data configuration."""
    return DATA_CONFIG

def get_model_config():
    """Get model configuration."""
    return MODEL_CONFIG

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_CONFIG["raw_data"]["local_path"],
        DATA_CONFIG["processed_data"]["path"],
        MODEL_CONFIG["model_registry"]["save_path"]
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
