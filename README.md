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

