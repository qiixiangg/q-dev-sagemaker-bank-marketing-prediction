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

### Docker Development
- Docker
- Docker Compose

## Quick Start

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bank-marketing-prediction.git
cd bank-marketing-prediction
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

### Running Tests

Local:
```bash
python -m pytest tests/
```

Docker:
```bash
docker-compose run test
```

### Running Linting Checks

Local:
```bash
flake8 src/ tests/
```

Docker:
```bash
docker-compose run test lint
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

## CI/CD Pipeline

The project includes a GitHub Actions workflow that:

1. Runs tests
2. Performs code quality checks
3. Builds and pushes Docker images
4. Deploys to production (when configured)

## Development Workflow

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and run tests:
```bash
python -m pytest tests/
```

3. Format code:
```bash
black src/ tests/
```

4. Create a pull request

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
