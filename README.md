# Bank Marketing Prediction Project

This project demonstrates a real-world machine learning workflow for predicting bank marketing campaign success, focusing on two main roles:

1. Data Scientist (DS) workflow for model development
2. Machine Learning Engineer (MLE) workflow for model deployment

## Project Structure

```
├── data/                  # Data directory
│   └── raw/              # Raw data files
├── models/               # Saved models and artifacts
│   ├── xgboost_model.json       # Trained model
│   └── model_metadata.json      # Model metadata
├── notebooks/            # Jupyter notebooks for DS workflow
│   ├── model_development.ipynb  # Main DS notebook
│   ├── mlruns/              # MLflow experiment tracking
│   └── mlflow.db            # MLflow tracking database
├── src/                  # Source code
│   └── serving/         # Model serving (MLE)
│       ├── __init__.py
│       ├── app.py       # FastAPI application
│       └── model_serving.py  # Model serving logic
├── scripts/             # Utility scripts
│   └── run_api.py       # API startup script
└── requirements.txt     # Project dependencies
```

## Data Science Workflow

The Data Science workflow is documented in `notebooks/model_development.ipynb` and includes:

1. Data Collection & Loading
   - Automatic dataset download if not present
   - Data validation and initial inspection

2. Exploratory Data Analysis (EDA)
   - Data quality assessment
   - Feature distributions
   - Target analysis
   - Feature relationships

3. Data Preprocessing
   - Target encoding
   - Categorical feature encoding
   - Numeric feature scaling
   - Train/validation/test splitting

4. Model Development & Training
   - Cross-validation training
   - Hyperparameter tuning using GridSearchCV
   - Final model training with best parameters
   - MLflow experiment tracking

5. Model Evaluation
   - ROC-AUC score
   - Confusion matrix
   - Feature importance analysis
   - Precision-Recall curves

6. Model Export
   - Model saving
   - Metadata export with preprocessing parameters

### Getting Started with DS Workflow

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter:
```bash
jupyter notebook
```

4. Start the MLflow server:
```bash
cd notebooks
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```
   - MLflow UI will be accessible at http://localhost:5000
   - Experiments are tracked in SQLite database (mlflow.db)
   - Artifacts are stored in mlruns directory

5. Open `notebooks/model_development.ipynb` to start the DS workflow
   - The notebook will automatically:
     - Download the dataset if not present
     - Connect to the MLflow server
   - All preprocessing steps are performed in-memory

## Machine Learning Engineering Workflow

The MLE workflow focuses on deploying and serving the model in production. The workflow integrates with MLflow for model tracking and versioning:

1. Model Artifact Management
   - MLflow tracked experiments in `mlruns/` directory
   - Best model exported as `models/xgboost_model.json`
   - Model metadata stored in `models/model_metadata.json`:
     - Feature names and types
     - Label mappings for categorical features
     - Scaling parameters for numeric features
     - Model hyperparameters
     - Performance metrics
   - Experiment tracking via MLflow UI

2. Model Serving API
   - FastAPI-based REST API
   - In-memory preprocessing pipeline
     - No dependency on serialized preprocessing objects
     - Stateless transformation using stored parameters
   - Input validation with Pydantic
   - Comprehensive error handling
   - Logging system

### Starting the Model Serving API

1. Ensure virtual environment is activated and dependencies are installed

2. Run the API:
```bash
python scripts/run_api.py
```

3. API will be available at `http://localhost:8000`

### API Endpoints

- `POST /predict`
  - Accepts JSON payload with feature values
  - Returns prediction probability and binary prediction
  - Example:
    ```bash
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{
             "age": 41,
             "job": "management",
             "marital": "married",
             "education": "university.degree",
             "default": "no",
             "housing": "yes",
             "loan": "no",
             "contact": "cellular",
             "month": "may",
             "day_of_week": "mon",
             "duration": 240,
             "campaign": 1,
             "pdays": -1,
             "previous": 0,
             "poutcome": "nonexistent",
             "emp_var_rate": 1.1,
             "cons_price_idx": 93.994,
             "cons_conf_idx": -36.4,
             "euribor3m": 4.857,
             "nr_employed": 5191.0
         }'
    ```

- `GET /health`
  - Health check endpoint
  - Returns service status

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies

## Project Highlights

1. **Data Science Best Practices**
   - Automatic data acquisition
   - Comprehensive EDA
   - Cross-validation training
   - Hyperparameter optimization
   - MLflow experiment tracking and versioning
   - Feature importance analysis
   - Model evaluation metrics
   - Experiment comparison and visualization

2. **Production-Ready Model Serving**
   - RESTful API with FastAPI
   - Stateless preprocessing
   - Strong input validation
   - Error handling
   - Logging
   - Health monitoring

3. **Code Quality**
   - Modular design
   - Type hints
   - Documentation
   - Error handling
   - Logging
   - Testing

## License

This project is licensed under the MIT License - see the LICENSE file for details.
