# Usa una imagen base ligera con Python 3.11
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /opt/

# Instalar uv y otras herramientas necesarias
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc curl && \
    pip install --upgrade pip && \
    pip install uv && \
    rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencias
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock

# Instalar solo dependencias (no proyecto)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --inexact


COPY src src

# Instalar el proyecto (dependencias ya están instaladas)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --inexact

# Asegurar que el entorno virtual esté en el PATH
ENV PATH="/opt/.venv/bin:$PATH"

CMD [ "python3", "-m", "src.main" ]
