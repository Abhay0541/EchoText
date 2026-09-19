FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# Only the core (uncommented) dependencies install here — the heavy LoRA/
# integration extras are commented out in requirements.txt on purpose so the
# container stays small for the default (template backend) demo.
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
