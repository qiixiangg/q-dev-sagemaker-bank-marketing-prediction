"""
Script to download and prepare the bank marketing dataset.
"""
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

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
        target_filename = data_config["raw_data"]["filename"]
        
        # If target file already exists, skip download
        if (raw_data_path / target_filename).exists():
            logger.info(f"Target file {target_filename} already exists, skipping download")
            return
        
        # Download data
        zip_path = raw_data_path / "bank-marketing.zip"
        if not zip_path.exists():
            download_file(data_config["raw_data"]["url"], str(zip_path))
        
        # Create a temporary directory for extraction
        temp_extract_path = raw_data_path / "temp"
        temp_extract_path.mkdir(exist_ok=True)
        
        # Extract main zip
        extract_zip(str(zip_path), str(temp_extract_path))
        
        # Look for and extract nested zip if it exists
        for nested_zip in temp_extract_path.glob("*.zip"):
            extract_zip(str(nested_zip), str(temp_extract_path))
        
        # Find and move the target file
        for file_path in temp_extract_path.rglob(target_filename):
            file_path.rename(raw_data_path / target_filename)
            break
        
        # Cleanup
        import shutil
        shutil.rmtree(str(temp_extract_path))
        zip_path.unlink()
        
        logger.info("Data preparation completed successfully")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
