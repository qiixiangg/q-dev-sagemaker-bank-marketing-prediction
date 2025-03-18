"""
Data processing module for the ML pipeline.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..utils.logger import get_logger

logger = get_logger(__name__)

class DataProcessor:
    """Handles all data processing operations."""
    
    def __init__(self, config: Dict):
        """
        Initialize DataProcessor with configuration.
        
        Args:
            config: Configuration dictionary containing data processing parameters
        """
        self.config = config
        self.target = config['features']['target']
        self.numeric_features = config['features']['numeric']
        self.categorical_features = config['features']['categorical']
        self.drop_features = config['features']['drop']
        self.scaler = MinMaxScaler()
        
    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Read CSV with proper handling of quotes and separators
            df = pd.read_csv(
                filepath,
                sep=';',
                quotechar='"',
                doublequote=True,
                encoding='utf-8'
            )
            
            # Clean column names (remove quotes if present)
            df.columns = df.columns.str.strip('"')
            
            logger.info(f"Successfully loaded data from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {str(e)}")
            raise
            
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the data with feature engineering steps.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        try:
            # Convert target to binary
            df['y'] = (df['y'] == 'yes').astype(int)
            
            # Create derived features
            df["no_previous_contact"] = np.where(df["pdays"] == 999, 1, 0)
            df["not_working"] = np.where(
                df["job"].isin(["student", "retired", "unemployed"]), 1, 0
            )
            
            # Drop unnecessary columns
            df_processed = df.drop(self.drop_features, axis=1)
            
            # Process age into ranges
            bins = [18, 30, 40, 50, 60, 70, 90]
            labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-plus']
            df_processed['age_range'] = pd.cut(
                df_processed['age'].astype(float), 
                bins, 
                labels=labels, 
                include_lowest=True
            )
            
            # Create age range dummies and drop original columns
            age_dummies = pd.get_dummies(df_processed['age_range'], prefix='age', dtype=int)
            df_processed = pd.concat([df_processed, age_dummies], axis=1)
            df_processed.drop(['age', 'age_range'], axis=1, inplace=True)
            
            # Scale remaining numeric features
            numeric_features = [f for f in self.numeric_features if f != 'age']
            if numeric_features:
                df_processed[numeric_features] = self.scaler.fit_transform(
                    df_processed[numeric_features]
                )
            
            # Convert categorical variables to indicators
            categorical_cols = df_processed.select_dtypes(include=['object']).columns
            df_processed = pd.get_dummies(
                df_processed,
                columns=categorical_cols,
                dtype=int
            )
            
            # Ensure target column is first
            cols = df_processed.columns.tolist()
            cols.remove('y')
            df_processed = df_processed[['y'] + cols]
            
            logger.info("Successfully processed data")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
            
    def split_data(
        self, 
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation and test sets.
        
        Args:
            df: Input DataFrame
            train_size: Proportion of data for training
            val_size: Proportion of data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        try:
            # Shuffle data
            df_shuffled = df.sample(frac=1, random_state=random_state)
            
            # Calculate split indices
            train_end = int(len(df) * train_size)
            val_end = int(len(df) * (train_size + val_size))
            
            # Split data
            train_data = df_shuffled.iloc[:train_end]
            val_data = df_shuffled.iloc[train_end:val_end]
            test_data = df_shuffled.iloc[val_end:]
            
            logger.info(
                f"Split data into train:{len(train_data)}, "
                f"validation:{len(val_data)}, test:{len(test_data)}"
            )
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
            
    def save_data(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        output_dir: Union[str, Path]
    ) -> None:
        """
        Save processed datasets to CSV files.
        
        Args:
            train_data: Training DataFrame
            val_data: Validation DataFrame
            test_data: Test DataFrame
            output_dir: Directory to save the files
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            train_data.to_csv(output_dir / "train.csv", index=False)
            val_data.to_csv(output_dir / "validation.csv", index=False)
            test_data.to_csv(output_dir / "test.csv", index=False)
            
            # Save baseline data for monitoring
            baseline_data = pd.concat([train_data, val_data]).drop(self.target, axis=1)
            baseline_data.to_csv(output_dir / "baseline.csv", index=False)
            
            logger.info(f"Successfully saved processed data to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
