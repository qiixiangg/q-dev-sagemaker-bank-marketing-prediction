"""
Main pipeline script that orchestrates the ML workflow.
"""
import hydra
from omegaconf import DictConfig
from pathlib import Path

from data.data_processor import DataProcessor
from training.model_trainer import ModelTrainer
from monitoring.model_monitor import ModelMonitor
from utils.logger import get_logger

logger = get_logger(__name__)

@hydra.main(config_path="../config", config_name="config")
def run_pipeline(config: DictConfig) -> None:
    """
    Run the complete ML pipeline.
    
    Args:
        config: Hydra configuration
    """
    try:
        logger.info("Starting ML pipeline")
        
        # Initialize components
        data_processor = DataProcessor(config.data)
        model_trainer = ModelTrainer(config.model)
        model_monitor = ModelMonitor(config.model)
        
        # Data Processing
        logger.info("Starting data processing")
        raw_data = data_processor.load_data(
            Path(config.data.raw_data.local_path) / config.data.raw_data.filename
        )
        processed_data = data_processor.process_data(raw_data)
        
        # Split data
        train_data, val_data, test_data = data_processor.split_data(
            processed_data,
            train_size=config.data.split.train_size,
            val_size=config.data.split.val_size,
            random_state=config.data.split.random_state
        )
        
        # Save processed datasets
        data_processor.save_data(
            train_data,
            val_data,
            test_data,
            config.data.processed_data.path
        )
        logger.info("Completed data processing")
        
        # Model Training
        logger.info("Starting model training")
        dtrain, dval, dtest = model_trainer.prepare_data(
            train_data,
            val_data,
            test_data,
            target_col=config.data.features.target
        )
        
        # Cross-validation
        cv_results = model_trainer.train_with_cv(dtrain)
        
        # Train final model
        model = model_trainer.train(dtrain, dval)
        
        # Evaluate model
        test_auc, predictions, confusion_matrix = model_trainer.evaluate(
            dtest,
            test_data[config.data.features.target]
        )
        
        # Save model
        model_trainer.save_model(config.model.model_registry.save_path)
        logger.info("Completed model training")
        
        # Model Monitoring
        logger.info("Starting model monitoring")
        model_monitor.load_baseline(
            Path(config.data.processed_data.path) / config.data.processed_data.baseline_file
        )
        
        # Detect data drift
        drift_stats, drifted_features = model_monitor.detect_drift(
            test_data.drop(config.data.features.target, axis=1)
        )
        
        # Monitor performance
        performance_metrics = model_monitor.monitor_performance(
            test_data[config.data.features.target],
            predictions,
            threshold=config.model.model_registry.metadata.threshold
        )
        
        # Save monitoring report
        model_monitor.save_monitoring_report(
            drift_stats,
            performance_metrics,
            config.model.model_registry.save_path
        )
        logger.info("Completed model monitoring")
        
        logger.info("ML pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()
