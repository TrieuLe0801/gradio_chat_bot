# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy code
COPY ./chainlit_chat/llm /app/llm
COPY ./chainlit_chat/ui /app/ui
COPY ./chainlit_chat/requirements.txt /app/requirements.txt
COPY ./.env /app/.env

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 7860

