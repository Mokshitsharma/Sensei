FROM python:3.11-slim

WORKDIR /app

# System deps some Python packages (scipy/sklearn/torch wheels) need at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PYTHONUNBUFFERED=1

# HF Spaces (Docker) expects the app on 7860; Render/Railway inject $PORT.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
