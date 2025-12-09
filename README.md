# Repositorio de benchmark para tareas de clasificación y detección de objetos

## Uso
+ Compilar imagen ``` docker build -t [TAG] . ```
+ Correr contenedor con volumen para datasets 
``` 
mkdir -p ./data
docker run --rm -v $(pwd)/data:/app/data [TAG] --mode download_data --dataset_name [DATASET_NAME]
```
+ Entrenar modelo seleccionado en ``` config/default.yaml ```
```
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/config/default.yaml:/app/config/default.yaml:ro [TAG] --mode train_model  
```


+ Cuantizar modelo (ONNX/TensorRT)
```
docker run --rm --gpus all -v $(pwd)/output:/app/output --entrypoint python [TAG] quantize.py --model [MODEL] --task [TASK] --output_dir /app/output/quantized
```
Ejemplo:
```
docker run --rm --gpus all -v $(pwd)/output:/app/output --entrypoint python [TAG] quantize.py --model resnet18 --task classification
```
Ejemplo con checkpoint:
```
docker run --rm --gpus all -v $(pwd)/output:/app/output -v $(pwd)/checkpoints:/app/checkpoints --entrypoint python [TAG] quantize.py --model resnet18 --task classification --checkpoint /app/checkpoints/model.pth
```
