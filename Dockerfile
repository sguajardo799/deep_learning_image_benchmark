FROM nvcr.io/nvidia/pytorch:25.10-py3

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt


ENV OUTPUT_DIR=/app/data
ENV CACHE_DIR=/app/data/hf_cache

ENTRYPOINT ["python","main.py"]
