FROM python:3.11-slim

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

COPY tests/ ./tests/

COPY for_plot/ ./for_plot/

COPY new_model/ ./new_model/

COPY pytest.ini .



ENV PYTHONPATH=/app