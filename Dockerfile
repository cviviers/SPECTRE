FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN apt-get update && apt-get install -y git dos2unix
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY spectre /app/spectre
COPY experiments /app/experiments
COPY scripts /app/scripts
RUN dos2unix /app/scripts/run_distributed.sh

WORKDIR /app
ENV PYTHONPATH="${PYTHONPATH}:/app"