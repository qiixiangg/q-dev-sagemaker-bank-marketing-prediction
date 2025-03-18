# Bank Marketing Prediction Pipeline

A production-ready machine learning pipeline for predicting bank marketing campaign success. This project implements a complete MLOps workflow with both local and containerized execution options.

## Project Structure

```
├── config/                 # Configuration files
│   ├── data/              # Data processing configs
│   └── model/             # Model training configs
├── src/                   # Source code
│   ├── data/              # Data processing modules
│   ├── training/          # Model training modules
│   ├── monitoring/        # Model monitoring modules
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── notebooks/             # Jupyter notebooks
└── scripts/               # Utility scripts
```

## Features

- Automated data preprocessing and feature engineering
- XGBoost model training with cross-validation
- Model performance monitoring and drift detection
- Comprehensive test suite
- CI/CD pipeline with GitHub Actions
- Docker containerization
- Simple configuration management
- Logging and experiment tracking

## Requirements

### Local Development
- Python 3.8+
- pip
- For macOS users:
  ```bash
  brew install libomp  # Required for XGBoost
  ```

### Docker Development
- Docker
- Docker Compose

## Quick Start

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/qiixiangg/q-dev-sagemaker-bank-marketing-prediction.git
cd q-dev-sagemaker-bank-marketing-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the data:
```bash
python scripts/download_data.py
```

5. Run the pipeline:
```bash
python src/pipeline.py
```

### Docker Development

1. Build and start the pipeline:
```bash
docker-compose up pipeline
```

## Development

### Testing Guide

#### Local Testing Setup

1. Install test dependencies:
```bash
pip install pytest pytest-cov black flake8
```

2. Run all tests with coverage:
```bash
# From project root
python -m pytest tests/ --cov=src --cov-report=term-missing
```

3. Run specific test files:
```bash
# Test data processing
python -m pytest tests/test_data_processor.py -v

# Test model training
python -m pytest tests/test_model_trainer.py -v

# Test model monitoring
python -m pytest tests/test_model_monitor.py -v
```

4. Run tests with detailed output:
```bash
python -m pytest -v --cov=src --cov-report=term-missing tests/
```

#### Code Quality Checks

1. Run code formatting:
```bash
# Check formatting
black --check src/ tests/

# Apply formatting
black src/ tests/
```

2. Run linting:
```bash
flake8 src/ tests/
```

#### Docker Testing

1. Run all tests in Docker:
```bash
docker-compose run test
```

2. Run specific test file in Docker:
```bash
docker-compose run test python -m pytest tests/test_data_processor.py -v
```

3. Run tests with linting:
```bash
docker-compose run test sh -c "flake8 src/ tests/ && pytest tests/"
```

#### Continuous Testing During Development

1. Install pytest-watch for continuous testing:
```bash
pip install pytest-watch
```

2. Run tests automatically on file changes:
```bash
ptw tests/ -- --cov=src --cov-report=term-missing
```

### Starting Jupyter Lab

Local:
```bash
jupyter lab
```

Docker:
```bash
docker-compose up jupyter
```

Then open http://localhost:8888 in your browser.

## Model Monitoring

Start the monitoring service:

Local:
```bash
python src/monitoring/model_monitor.py
```

Docker:
```bash
docker-compose up monitor
```

## Configuration

Configuration is managed through Python dictionaries in `src/utils/config.py`. The main configuration sections are:

- Data processing settings
- Model training parameters
- Monitoring thresholds and settings

## Docker Services

- `pipeline`: Main ML pipeline
- `test`: Run tests and linting
- `monitor`: Model monitoring service
- `jupyter`: Jupyter Lab environment

## GitHub Setup and CI/CD Pipeline

### Initial Setup

1. Fork and clone the repository
2. Set up GitHub Secrets:
   - Go to your repository's Settings > Secrets and variables > Actions
   - Add the following secrets:
     ```
     DOCKERHUB_USERNAME: Your Docker Hub username
     DOCKERHUB_TOKEN: Your Docker Hub access token
     ```

### CI/CD Pipeline

The project includes a GitHub Actions workflow that automatically:

1. Runs Python tests:
   - Installs system dependencies (libomp for XGBoost)
   - Installs Python dependencies
   - Downloads and prepares data
   - Runs pytest with coverage
   - Performs linting checks

2. Runs Docker tests:
   - Builds and tests Docker containers
   - Runs the full pipeline in Docker
   - Verifies container functionality

3. Builds and pushes Docker images (on main branch):
   - Builds optimized Docker images
   - Pushes to Docker Hub with version tags
   - Uses build caching for faster builds

4. Handles deployment (when configured)

### Development Workflow

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and test locally:
```bash
# Run tests
python -m pytest tests/

# Format code
black src/ tests/
flake8 src/ tests/

# Test the full pipeline
python scripts/download_data.py
python src/pipeline.py
```

3. Create a pull request:
   - The CI pipeline will automatically run tests
   - All checks must pass before merging
   - Docker images are built and pushed on merge to main

### Deployment

The pipeline is ready for deployment with:
- Docker image versioning
- Build caching
- Automated testing
- Directory structure management

To deploy to production:
1. Merge to main branch
2. CI/CD pipeline automatically:
   - Runs all tests
   - Builds and pushes Docker images
   - Handles deployment (when configured)

### Monitoring Deployment

Monitor your GitHub Actions:
1. Go to Actions tab in your repository
2. Check workflow runs for:
   - Test results
   - Build status
   - Deployment status
   - Coverage reports

## Production Deployment

Local:
```bash
python src/pipeline.py
```

Docker:
```bash
docker build -t bank-marketing:latest .
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models bank-marketing:latest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
