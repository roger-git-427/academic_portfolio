# Usa una imagen ligera de Python 3.11
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /opt/

# Instalar herramientas necesarias y uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc curl && \
    pip install --upgrade pip && \
    pip install uv && \
    rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencias
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock

# Instalar dependencias sin el proyecto ni los dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --inexact

# Copiar el código fuente
COPY app app

# Instalar el proyecto (sin dev)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --inexact

# Asegura que el entorno virtual esté en el PATH
ENV PATH="/opt/.venv/bin:$PATH"

# Comando de inicio usando uvicorn directamente
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
