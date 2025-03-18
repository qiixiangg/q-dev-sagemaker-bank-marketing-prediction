# ML Pipeline for Bank Marketing Prediction

A production-ready machine learning pipeline for predicting bank marketing campaign success. This project implements a complete MLOps workflow using Docker containers.

## Project Structure

```
ml_pipeline/
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
- Configurable with Hydra
- Logging and experiment tracking

## Requirements

- Docker
- Docker Compose

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml_pipeline.git
cd ml_pipeline
```

2. Start the pipeline:
```bash
docker-compose up pipeline
```

## Development

### Running Tests

```bash
docker-compose run test
```

### Running Linting Checks

```bash
docker-compose run test lint
```

### Starting Development Environment

```bash
docker-compose up dev
```

### Starting Jupyter Lab

```bash
docker-compose up jupyter
```

Then open http://localhost:8888 in your browser. The token will be shown in the console output.

## Model Monitoring

Start the monitoring service:

```bash
docker-compose up monitor
```

## Configuration

The project uses Hydra for configuration management. Main configuration files:

- `config/data/default.yaml`: Data processing settings
- `config/model/default.yaml`: Model training parameters
- `config/monitoring/default.yaml`: Monitoring thresholds and settings

Example of running with custom configuration:

```bash
docker-compose run pipeline python src/pipeline.py model.xgboost.params.max_depth=5
```

## Docker Services

- `pipeline`: Main ML pipeline
- `test`: Run tests and linting
- `monitor`: Model monitoring service
- `dev`: Development environment
- `jupyter`: Jupyter Lab environment

## CI/CD Pipeline

The project includes a GitHub Actions workflow that:

1. Runs tests in Docker containers
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
docker-compose run test
```

3. Format code:
```bash
docker-compose run test format
```

4. Create a pull request

## Production Deployment

Build the production image:

```bash
docker build -t ml-pipeline:latest .
```

Run the pipeline in production:

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models ml-pipeline:latest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
