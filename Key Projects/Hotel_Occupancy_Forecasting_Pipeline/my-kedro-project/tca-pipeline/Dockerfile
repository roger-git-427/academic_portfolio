#────────────────────────────────────────────────────
# 1) BASE IMAGE: use Python 3.11.9-slim
#────────────────────────────────────────────────────
FROM python:3.11.9-slim

#────────────────────────────────────────────────────
# 2) INSTALL SYSTEM TOOLS + uv
#────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && python -m pip install --upgrade pip setuptools wheel \
    && pip install uv \
    && rm -rf /var/lib/apt/lists/*

#────────────────────────────────────────────────────
# 3) INSTALL PYTHON DEPENDENCIES FROM uv.lock
#────────────────────────────────────────────────────
WORKDIR /app

# 3.1) Copy only the lockfile and manifest so Docker can cache this step
COPY pyproject.toml uv.lock ./

# 3.2) Sync exactly the production dependencies
#      • --frozen       → ensures pyproject.toml matches uv.lock
#      • --no-install-project → do not install the local project itself
#      • --no-dev       → skip dev-dependencies
#      • --inexact      → allow minor version flexibility if lockfile uses ranges
RUN uv sync --frozen --no-install-project --no-dev --inexact

#────────────────────────────────────────────────────
# 4) CREATE A NON-ROOT USER (uid=999, gid=999)
#────────────────────────────────────────────────────
RUN groupadd -g 999 kedro_group && \
    useradd -m -u 999 -g 999 -s /bin/bash kedro_docker

#────────────────────────────────────────────────────
# 5) COPY ENTIRE PROJECT AND FIX OWNERSHIP
#────────────────────────────────────────────────────
COPY . /home/kedro_docker
RUN chown -R 999:999 /home/kedro_docker

#────────────────────────────────────────────────────
# 6) SWITCH TO NON-ROOT USER AND SET WORKDIR
#────────────────────────────────────────────────────
USER kedro_docker
WORKDIR /home/kedro_docker

#────────────────────────────────────────────────────
# 7) EXPOSE PORT (e.g., for Jupyter or kedro-viz)
#────────────────────────────────────────────────────
EXPOSE 8888

#────────────────────────────────────────────────────
# 8) DEFAULT COMMAND
#────────────────────────────────────────────────────
CMD ["kedro", "run"]
