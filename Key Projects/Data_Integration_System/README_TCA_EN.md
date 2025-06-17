
# TCA Challenge - Time Series

This repository contains the solution to the time series challenge for the **TCA** project. The architecture is organized into multiple modules to clearly separate backend logic, frontend interface, inference services, and deployment configuration.

## 📁 Project Structure

```bash
/
├── backend/             # API backend (FastAPI, PostgreSQL, business logic)
├── frontend/            # Dashboard interface (Dash)
├── inference_api/       # Inference service for the time series model
├── kubernetes_config/   # Kubernetes deployment configuration files
├── my-kedro-project/    # Data preprocessing and model training pipeline (Kedro)
├── .gitignore
├── README.md
```

## 📦 Backend 

Contains the backend server developed with FastAPI. Its main function is to expose the necessary endpoints for handling predictions and interacting with the database.  
This is the full version of the inference service.

Relevant internal structure:

```bash
backend/
├── .venv/               # Virtual environment (not in the repo but required)
├── app/
│   ├── core/            # Main configuration 
│   ├── db/              # Database session and SQLAlchemy models
│   ├── models/          # Pydantic models (table schemas)
│   ├── services/        # Predictions, queries
│   ├── utils/           # Helper functions
│   └── main.py          # FastAPI entry point
├── Dockerfile           # Backend image
├── .env                 # Environment variables (not in the repo for security reasons)
└── pyproject.toml       # Python dependencies
```

🔧 How to start the backend in development

```bash
cd backend
uv venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate     # Windows PowerShell
uv sync
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --reload
```

## 💻 frontend

Developed in Dash, this module displays the dashboard with results, allows data upload, and visualizes metrics interactively.

Relevant internal structure:

```bash
frontend/
├── .venv/               # Virtual environment (not in the repo but required)
├── app/
│   ├── assets/          # CSS, images, and static resources
│   ├── components/      # Reusable Dash components
│   ├── utils/           # Helper functions for loading and processing data
│   └── main.py          # Dash app entry point
├── Dockerfile           # Frontend image
├── .env                 # Environment variables (not in the repo for security reasons)
└── pyproject.toml       # Python dependencies
```

🔧 How to start the frontend in development

```bash
cd frontend
uv venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate     # Windows PowerShell
uv sync
uv run python -m app.main
# By default, it runs at http://127.0.0.1:8050
```

## 🔍 inference_api

Contains the scripts needed to expose the time series prediction model, process data, and serve predictions through a standalone endpoint.  
This is the initial version of the backend, focused on inference.

## ⚙️ kubernetes_config

Includes the YAML manifests to deploy services in a Kubernetes cluster.

## 📊 my-kedro-project

Contains the full Kedro pipeline for data preprocessing and model training, organized in nodes and pipelines. See the final version in `final-tca-pipeline-reto`.

## 🚀 Workflow

**Preprocessing and Training**

Run the Kedro pipeline (in `my-kedro-project/final-tca-pipeline-reto`) to generate models and artifacts.

**Main Backend**

Start the backend for the general API.

**Dashboard**

Start the frontend for interactive visualization.

**Deployment**

Apply the manifests from `kubernetes_config` after building Docker images using the Dockerfiles.

✅ General Requirements

Python >= 3.11
