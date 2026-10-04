# NOTE: training image for now. A separate, slimmer Dockerfile.serving
# will be added in Week 5 for the FastAPI inference app.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src.train", "--config", "configs/baseline.yaml"]
