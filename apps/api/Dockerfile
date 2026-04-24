FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY prompts /app/prompts
COPY apps/api /app/apps/api
COPY scripts /app/scripts

RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/src:/app/apps/api

WORKDIR /app/apps/api
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
