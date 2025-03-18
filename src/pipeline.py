"""
Main pipeline script that orchestrates the ML workflow.
"""
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from src.data.data_processor import DataProcessor
from src.training.model_trainer import ModelTrainer
from src.monitoring.model_monitor import ModelMonitor
from src.utils.logger import get_logger
from src.utils.config import get_data_config, get_model_config, ensure_directories

logger = get_logger(__name__)

def run_pipeline() -> None:
    """
    Run the complete ML pipeline.
    """
    try:
        logger.info("Starting ML pipeline")
        
        # Get configurations
        data_config = get_data_config()
        model_config = get_model_config()
        
        # Ensure directories exist
        ensure_directories()
        
        # Initialize components
        data_processor = DataProcessor(data_config)
        model_trainer = ModelTrainer(model_config)
        model_monitor = ModelMonitor(model_config)
        
        # Data Processing
        logger.info("Starting data processing")
        raw_data = data_processor.load_data(
            Path(data_config["raw_data"]["local_path"]) / data_config["raw_data"]["filename"]
        )
        processed_data = data_processor.process_data(raw_data)
        
        # Split data
        train_data, val_data, test_data = data_processor.split_data(
            processed_data,
            train_size=data_config["split"]["train_size"],
            val_size=data_config["split"]["val_size"],
            random_state=data_config["split"]["random_state"]
        )
        
        # Save processed datasets
        data_processor.save_data(
            train_data,
            val_data,
            test_data,
            data_config["processed_data"]["path"]
        )
        logger.info("Completed data processing")
        
        # Model Training
        logger.info("Starting model training")
        dtrain, dval, dtest = model_trainer.prepare_data(
            train_data,
            val_data,
            test_data,
            target_col=data_config["features"]["target"]
        )
        
        # Cross-validation
        cv_results = model_trainer.train_with_cv(dtrain)
        
        # Train final model
        model = model_trainer.train(dtrain, dval)
        
        # Evaluate model
        test_auc, predictions, confusion_matrix = model_trainer.evaluate(
            dtest,
            test_data[data_config["features"]["target"]]
        )
        
        # Save model
        model_trainer.save_model(model_config["model_registry"]["save_path"])
        logger.info("Completed model training")
        
        # Model Monitoring
        logger.info("Starting model monitoring")
        model_monitor.load_baseline(
            Path(data_config["processed_data"]["path"]) / data_config["processed_data"]["baseline_file"]
        )
        
        # Detect data drift
        drift_stats, drifted_features = model_monitor.detect_drift(
            test_data.drop(data_config["features"]["target"], axis=1)
        )
        
        # Monitor performance
        performance_metrics = model_monitor.monitor_performance(
            test_data[data_config["features"]["target"]],
            predictions,
            threshold=model_config["model_registry"]["metadata"]["threshold"]
        )
        
        # Save monitoring report
        model_monitor.save_monitoring_report(
            drift_stats,
            performance_metrics,
            model_config["model_registry"]["save_path"]
        )
        logger.info("Completed model monitoring")
        
        logger.info("ML pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()
