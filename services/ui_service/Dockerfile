FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/ui_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ui_service/app.py .
COPY services/ui_service/.chainlit ./.chainlit
COPY services/ui_service/public ./public

EXPOSE 8080

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8080", "--headless"]
