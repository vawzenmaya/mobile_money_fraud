FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make run.sh executable
RUN chmod +x run.sh

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app.api:app"]
