FROM nvcr.io/nvidia/pytorch:25.10-py3

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app


ENV OUTPUT_DIR=/app/data
ENV CACHE_DIR=/app/data/hf_cache

ENTRYPOINT ["python","main.py"]
