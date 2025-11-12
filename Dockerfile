FROM nvcr.io/nvidia/pytorch:25.10-py3

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python","main.py"]
