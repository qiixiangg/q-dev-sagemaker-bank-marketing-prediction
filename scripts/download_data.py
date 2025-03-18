"""
Script to download and prepare the bank marketing dataset.
"""
import os
import urllib.request
import zipfile
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.utils.logger import get_logger

logger = get_logger(__name__)

def download_file(url: str, filepath: Path) -> None:
    """
    Download a file from a URL.
    
    Args:
        url: URL to download from
        filepath: Path to save the file
    """
    try:
        logger.info(f"Downloading from {url} to {filepath}")
        urllib.request.urlretrieve(url, filepath)
        logger.info("Download completed")
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise

def extract_zip(zip_path: Path, extract_path: Path) -> None:
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to zip file
        extract_path: Path to extract to
    """
    try:
        logger.info(f"Extracting {zip_path} to {extract_path}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_path)
        logger.info("Extraction completed")
    except Exception as e:
        logger.error(f"Error extracting file: {str(e)}")
        raise

@hydra.main(config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    """
    Download and prepare the dataset.
    
    Args:
        config: Hydra configuration
    """
    try:
        logger.info("Starting data download")
        
        # Create data directories
        raw_data_path = Path(config.data.raw_data.local_path)
        raw_data_path.mkdir(parents=True, exist_ok=True)
        
        # Download data
        zip_path = raw_data_path / "bank-marketing.zip"
        if not zip_path.exists():
            download_file(config.data.raw_data.url, zip_path)
        
        # Extract data
        extract_zip(zip_path, raw_data_path)
        
        # Extract nested zip if it exists
        additional_zip = raw_data_path / "bank-additional.zip"
        if additional_zip.exists():
            extract_zip(additional_zip, raw_data_path)
        
        logger.info("Data preparation completed successfully")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
