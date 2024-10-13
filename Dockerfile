FROM python:3.12.7-bullseye

WORKDIR /app

COPY requirements.txt .

RUN pip install -U pip wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
