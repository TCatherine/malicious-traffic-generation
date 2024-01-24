FROM python:3.11

WORKDIR /app

COPY requirements.txt requirements.txt
RUN apt-get update && \
    apt-get install -y curl iputils-ping

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY aggregator ./aggregator
COPY generator ./generator
COPY analyzer ./analyzer
COPY *.py .

