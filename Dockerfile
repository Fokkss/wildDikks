FROM python:3.12-slim

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

COPY tests/ ./tests/

COPY for_plot/ ./for_plot/