# Reto TCA - Series de Tiempo

Este repositorio contiene la solución al reto de series de tiempo para el proyecto **TCA**. La arquitectura está organizada en múltiples módulos para separar claramente la lógica del backend, frontend, servicios de inferencia y configuración de despliegue.

## 📁 Estructura del Proyecto

```bash
/
├── backend/             # API backend (FastAPI, PostgreSQL, lógica de negocio)
├── frontend/            # Interfaz del dashboard (Dash)
├── inference_api/       # Servicio de inferencia para el modelo de series de tiempo
├── kubernetes_config/   # Archivos de configuración para despliegue en Kubernetes
├── my-kedro-project/    # Pipeline de preprocesamiento de datos y entrenamiento de modelos (Kedro)
├── .gitignore
├── README.md
```

## 📦 Backend 

Contiene el servidor backend desarrollado en FastAPI. Su función principal es exponer los endpoints necesarios para manejar predicciones y la interacción con la base de datos.
Es la versión completa del servicio de inferencia. 

Estructura interna relevante:

```bash
backend/
├── .venv/               # Entorno virtual (no está en el repositorio pero es necesario crearlo)
├── app/
│   ├── core/            # Configuración principal 
│   ├── db/              # Sesión con base de datos y modelos SQLAlchemy
│   ├── models/          # Pydantic models (esquemas de tablas)
│   ├── services/        # Predicciones, consultas
│   ├── utils/           # Funciones auxiliares
│   └── main.py          # Punto de entrada de la API FastAPI
├── Dockerfile           # Imagen del backend
├── .env                 # Variables de entorno (no está en el repo por cuestiones de seguridad)
└── pyproject.toml       # Dependencias Python
```

🔧 Cómo arrancar el backend en desarrollo
```bash
cd backend
uv venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell
uv sync
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --reload
```

## 💻 frontend

Desarrollado en Dash, este módulo muestra el dashboard con resultados, permite cargar datos y visualizar métricas de forma interactiva.

Estructura interna relevante:

```bash
frontend/
├── .venv/               # Entorno virtual (no está en el repositorio pero es necesario crearlo)
├── app/
│   ├── assets/          # CSS, imágenes y recursos estáticos
│   ├── components/      # Componentes Dash reutilizables
│   ├── utils/           # Funciones auxiliares para cargar y procesar datos
│   └── main.py          # Punto de entrada de la app Dash
├── Dockerfile           # Imagen del frontend
├── .env                 # Variables de entorno (no está en el repo por cuestiones de seguridad)
└── pyproject.toml       # Dependencias Python
```

🔧 Cómo arrancar el frontend en desarrollo

```bash
cd frontend
uv venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell
uv sync
uv run python -m app.main
# Por defecto corre en http://127.0.0.1:8050
```

## 🔍 inference_api
Contiene los scripts necesarios para exponer el modelo de predicción de series de tiempo, procesar datos y servir predicciones en un endpoint independiente. Es la versión inicial de backend, pero enfocada a inferencia.

## ⚙️ kubernetes_config
Incluye los manifiestos YAML para desplegar los servicios en un clúster de Kubernetes.

## 📊 my-kedro-project
Contiene el pipeline completo de preprocesamiento de datos y entrenamiento de modelos con Kedro, organizado en nodos y pipelines, revisar la versión final en final-tca-pipeline-reto.

## 🚀 Flujo de Trabajo
Preprocesamiento y Entrenamiento

  Ejecutar el pipeline Kedro (en my-kedro-project/final-tca-pipeline-reto) para generar modelos y artefactos.

Backend Principal

  Levantar backend para la API general.

Dashboard

  Levantar frontend para la visualización interactiva.

Despliegue

  Aplicar los manifiestos de kubernetes_config después de construir imágenes Docker con los Dockerfile.

✅ Requisitos Generales
Python >= 3.11
