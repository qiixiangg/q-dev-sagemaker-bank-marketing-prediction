#!/bin/bash
set -e

case "$1" in
    run)
        echo "Running ML pipeline..."
        poetry run python src/pipeline.py
        ;;
    test)
        echo "Running tests..."
        poetry run pytest tests/ -v --cov=src --cov-report=term-missing
        ;;
    monitor)
        echo "Running model monitoring..."
        poetry run python src/monitoring/model_monitor.py
        ;;
    dev)
        echo "Starting development environment..."
        poetry install
        poetry run python src/pipeline.py --config-dir=config --config-name=dev
        ;;
    jupyter)
        echo "Starting Jupyter Lab..."
        poetry install
        poetry run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
        ;;
    lint)
        echo "Running linting checks..."
        poetry run black . --check
        poetry run isort . --check
        poetry run flake8 .
        ;;
    format)
        echo "Formatting code..."
        poetry run black .
        poetry run isort .
        ;;
    *)
        exec "$@"
        ;;
esac
