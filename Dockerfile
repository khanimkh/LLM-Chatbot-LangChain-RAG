FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirement.txt /app/requirement.txt
RUN pip install --no-cache-dir -r /app/requirement.txt

COPY . /app

EXPOSE 8000

WORKDIR /app/backend

CMD ["uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000"]
