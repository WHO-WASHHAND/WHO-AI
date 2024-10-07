FROM python:3.11
LABEL authors="nguyenthanhvy"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

COPY . .

RUN pip install -r requirements.txt


ENTRYPOINT ["python3", "server.py"]