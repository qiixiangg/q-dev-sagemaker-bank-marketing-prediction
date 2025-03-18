"""
Model training module for the ML pipeline.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve

from ..utils.logger import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    """Handles model training and evaluation operations."""
    
    def __init__(self, config: Dict):
        """
        Initialize ModelTrainer with configuration.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.model_params = config['xgboost']['params']
        self.training_params = config['xgboost']['training']
        self.cv_params = config['xgboost']['cv']
        self.model = None
        
    def prepare_data(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
        target_col: str = 'y'
    ) -> Tuple[xgb.DMatrix, ...]:
        """
        Prepare data for XGBoost training.
        
        Args:
            train_data: Training DataFrame
            val_data: Validation DataFrame
            test_data: Test DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of DMatrix objects
        """
        try:
            # Prepare training data
            X_train = train_data.drop(target_col, axis=1)
            y_train = train_data[target_col]
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            # Prepare validation data if provided
            dval = None
            if val_data is not None:
                X_val = val_data.drop(target_col, axis=1)
                y_val = val_data[target_col]
                dval = xgb.DMatrix(X_val, label=y_val)
            
            # Prepare test data if provided
            dtest = None
            if test_data is not None:
                X_test = test_data.drop(target_col, axis=1)
                y_test = test_data[target_col]
                dtest = xgb.DMatrix(X_test, label=y_test)
            
            return dtrain, dval, dtest
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
            
    def train_with_cv(self, dtrain: xgb.DMatrix) -> Dict:
        """
        Train model with cross-validation.
        
        Args:
            dtrain: Training data as DMatrix
            
        Returns:
            Cross-validation results
        """
        try:
            cv_results = xgb.cv(
                params=self.model_params,
                dtrain=dtrain,
                num_boost_round=self.training_params['num_boost_round'],
                nfold=self.cv_params['nfold'],
                stratified=self.cv_params['stratified'],
                early_stopping_rounds=self.training_params['early_stopping_rounds'],
                metrics=['auc'],
                seed=self.cv_params['seed']
            )
            
            logger.info(
                f"Cross-validation results - "
                f"Train AUC: {cv_results.iloc[-1]['train-auc-mean']:.4f} "
                f"(±{cv_results.iloc[-1]['train-auc-std']:.4f}), "
                f"Test AUC: {cv_results.iloc[-1]['test-auc-mean']:.4f} "
                f"(±{cv_results.iloc[-1]['test-auc-std']:.4f})"
            )
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise
            
    def train(
        self,
        dtrain: xgb.DMatrix,
        dval: Optional[xgb.DMatrix] = None
    ) -> xgb.Booster:
        """
        Train the final model.
        
        Args:
            dtrain: Training data as DMatrix
            dval: Validation data as DMatrix
            
        Returns:
            Trained XGBoost model
        """
        try:
            # Prepare evaluation list
            evals = [(dtrain, 'train')]
            if dval is not None:
                evals.append((dval, 'validation'))
            
            # Train model
            self.model = xgb.train(
                params=self.model_params,
                dtrain=dtrain,
                num_boost_round=self.training_params['num_boost_round'],
                early_stopping_rounds=self.training_params['early_stopping_rounds'],
                evals=evals,
                verbose_eval=self.training_params['verbose_eval']
            )
            
            logger.info("Successfully trained model")
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def evaluate(
        self,
        dtest: xgb.DMatrix,
        y_true: pd.Series
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model performance.
        
        Args:
            dtest: Test data as DMatrix
            y_true: True labels
            
        Returns:
            Tuple of (AUC score, predictions, confusion matrix)
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
            
            # Get predictions
            predictions = self.model.predict(dtest)
            
            # Calculate metrics
            auc_score = roc_auc_score(y_true, predictions)
            cm = confusion_matrix(y_true, np.round(predictions))
            
            logger.info(f"Test AUC: {auc_score:.4f}")
            
            return auc_score, predictions, cm
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
            
    def save_model(self, output_dir: str) -> None:
        """
        Save the trained model and its metadata.
        
        Args:
            output_dir: Directory to save the model
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet")
            
            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = output_dir / f"{self.config['model_registry']['model_name']}.json"
            self.model.save_model(str(model_path))
            
            # Save metadata
            metadata = {
                'model_name': self.config['model_registry']['model_name'],
                'version': self.config['model_registry']['version'],
                'params': self.model_params,
                'metadata': self.config['model_registry']['metadata']
            }
            
            metadata_path = output_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved model and metadata to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
