FROM python:3.12-slim

WORKDIR /app

COPY ui/requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY ui/ ./ui/

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]