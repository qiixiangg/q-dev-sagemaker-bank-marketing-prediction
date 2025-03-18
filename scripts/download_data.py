"""
Script to download and prepare the bank marketing dataset.
"""
import urllib.request
import zipfile
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config import get_data_config, ensure_directories

logger = get_logger(__name__)

def download_file(url: str, filepath: str) -> None:
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

def extract_zip(zip_path: str, extract_path: str) -> None:
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

def main() -> None:
    """Download and prepare the dataset."""
    try:
        logger.info("Starting data download")
        
        # Get configuration and ensure directories exist
        data_config = get_data_config()
        ensure_directories()
        
        # Set paths
        raw_data_path = Path(data_config["raw_data"]["local_path"])
        
        # Download data
        zip_path = raw_data_path / "bank-marketing.zip"
        if not zip_path.exists():
            download_file(data_config["raw_data"]["url"], str(zip_path))
        
        # Extract data
        extract_zip(str(zip_path), str(raw_data_path))
        
        # Extract nested zip if it exists
        additional_zip = raw_data_path / "bank-additional.zip"
        if additional_zip.exists():
            extract_zip(str(additional_zip), str(raw_data_path))
        
        logger.info("Data preparation completed successfully")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
