FROM python:3.9.10-slim-buster

RUN apt-get update && apt-get install python-tk python3-tk tk-dev -y

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "main.py", "--server.port", "8080", "--server.enableXsrfProtection", "false"]