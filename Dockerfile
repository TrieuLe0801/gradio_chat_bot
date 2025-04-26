# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy code
COPY ./llm /app/llm
COPY ./ui /app/ui
COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 7860

