# Streamlit app image. Uses the app-only lock (requirements-app.txt) and the
# trained models versioned in app/model/.

# ---- build stage: install dependencies into a virtual environment ----
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements-app.txt .
RUN /opt/venv/bin/pip install -r requirements-app.txt \
    && find /opt/venv -type d \( -name __pycache__ -o -name tests -o -name test \) -prune -exec rm -rf {} +

# ---- runtime stage ----
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY src ./src
COPY app ./app

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
