FROM nvcr.io/nvidia/pytorch:25.02-py3

RUN apt-get update && apt-get install -y git

WORKDIR /app/

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt
RUN mkdir /app/data

ADD spectre /app/spectre
ADD scripts /app/scripts

WORKDIR /app
COPY .env /app/.env

ENV PYTHONPATH="${PYTHONPATH}:/app"