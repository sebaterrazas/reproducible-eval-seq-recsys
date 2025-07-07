# Evaluación de Modelos de Recomendación Secuencial

Este repositorio implementa una evaluación reproducible de modelos de recomendación secuencial en el dataset RetailRocket. Incluye preprocesamiento de datos, entrenamiento de modelos base y avanzados, análisis de sensibilidad, métricas de diversidad/novedad y evaluación de rendimiento integral.

## 📋 Tabla de Contenidos

- [Resumen](#resumen)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Modelos Implementados](#modelos-implementados)
- [Dataset](#dataset)
- [Métricas de Evaluación](#métricas-de-evaluación)
- [Uso](#uso)
- [Resultados](#resultados)
- [Requisitos](#requisitos)

## 🎯 Resumen

Este proyecto proporciona un framework de evaluación integral para sistemas de recomendación secuencial, enfocándose en:

- **Experimentos reproducibles** con preprocesamiento y evaluación estandarizados
- **Múltiples arquitecturas de modelos** (GRU4Rec y NextItNet)
- **Métricas comprehensivas** incluyendo precisión, diversidad y sesgo de popularidad
- **Análisis de sensibilidad** para optimización de hiperparámetros
- **Resultados pre-computados** para análisis y comparación inmediatos

## 📁 Estructura del Repositorio

```
repo/
├── scripts/                              # Notebooks de Jupyter con experimentos completos
│   ├── GRU4Rec_RetailRocket_Optimized.ipynb    # Entrenamiento y evaluación de GRU4Rec
│   ├── NextItNet_RetailRocket_Optimized.ipynb  # Entrenamiento y evaluación de NextItNet
|   ...
├── outputs/                              # Resultados experimentales pre-computados
│   ├── GRU_RetailRocket/
│   │   ├── gru4rec_results.json         # Métricas principales de evaluación
│   │   └── gru4rec_sensitivity_analysis.json  # Análisis de dimensión de embeddings
│   ├── NextItNet_RetailRocket/
│   |   ├── nextitnet_results.json       # Métricas principales de evaluación
│   |   └── nextitnet_sensitivity_analysis.json # Análisis de dimensión de embeddings
|   ...
├── .gitignore                           # Configuración de Git ignore
└── README.md                            # Este archivo
```

### 📓 Directorio Scripts

Cada notebook contiene un **experimento completo y autocontenido** con:

- **Preprocesamiento de datos** y creación de secuencias
- **Implementación del modelo** y entrenamiento
- **Evaluación comprehensiva** con múltiples métricas
- **Análisis de sensibilidad** para dimensiones de embeddings
- **Visualización** de resultados y tendencias de rendimiento
- **Celdas pre-ejecutadas** con salidas guardadas para inspección inmediata

### 📊 Directorio Outputs

Contiene **archivos JSON** con resultados pre-computados para cada combinación modelo-dataset:

- `*_results.json`: Métricas principales de evaluación (Recall, MRR, ILD, Sesgo de Popularidad) para k=[5,10,20]
- `*_sensitivity_analysis.json`: Análisis de dimensión de embeddings con Recall@20, tiempos de entrenamiento y tamaños de modelo

## 🧠 Modelos Implementados

### 1. GRU4Rec

- **Arquitectura**: Recomendación secuencial basada en Unidad Recurrente con Compuertas (GRU)
- **Características Clave**: Aprendizaje basado en sesiones, optimización de pérdida de ranking
- **Implementación**: Implementación personalizada en PyTorch con optimizaciones de memoria

### 2. NextItNet

- **Arquitectura**: Red neuronal convolucional con convoluciones causales dilatadas
- **Características Clave**: Arquitectura no recurrente, capacidad de procesamiento paralelo
- **Implementación**: Convolución dilatada multi-bloque con conexiones residuales

## 📊 Dataset

**Dataset de E-commerce RetailRocket**

- **Fuente**: Dataset RetailRocket de Kaggle
- **Eventos**: Solo interacciones de visualización
- **Preprocesamiento**:
  - Longitud de secuencia: 3-50 items
  - Frecuencia mínima de item: 5 apariciones
  - División entrenamiento/prueba: 80/20
- **Estadísticas Finales**: ~400K secuencias, ~63K items únicos

## 📈 Métricas de Evaluación

### Métricas de Precisión

- **Recall@K**: Proporción de items relevantes encontrados en las top-K recomendaciones
- **MRR@K**: Rango Recíproco Medio del primer item relevante

### Métricas de Diversidad y Sesgo

- **ILD@K**: Diversidad Intra-Lista usando similitud coseno de embeddings aprendidos
- **Sesgo de Popularidad@K**: Tendencia a recomendar items populares

### Análisis de Sensibilidad

- **Dimensiones de Embedding**: [32, 64, 128, 256]
- **Métrica de Evaluación**: Recall@20
- **Métricas Adicionales**: Tiempo de entrenamiento, tamaño del modelo, eficiencia de parámetros

## 🚀 Uso

### Opción 1: Ver Resultados Pre-computados

```bash
# Ver resultados principales de evaluación
cat outputs/GRU_RetailRocket/gru4rec_results.json
cat outputs/NextItNet_RetailRocket/nextitnet_results.json

# Ver análisis de sensibilidad
cat outputs/GRU_RetailRocket/gru4rec_sensitivity_analysis.json
cat outputs/NextItNet_RetailRocket/nextitnet_sensitivity_analysis.json
```

### Opción 2: Ejecutar Experimentos

```bash
# Abrir y ejecutar notebooks (celdas pre-ejecutadas)
jupyter notebook scripts/GRU4Rec_RetailRocket_Optimized.ipynb
jupyter notebook scripts/NextItNet_RetailRocket_Optimized.ipynb
```

### Opción 3: Cargar Resultados en Python

```python
import json

# Cargar resultados de GRU4Rec
with open('outputs/GRU_RetailRocket/gru4rec_results.json', 'r') as f:
    gru4rec_results = json.load(f)

# Cargar resultados de NextItNet
with open('outputs/NextItNet_RetailRocket/nextitnet_results.json', 'r') as f:
    nextitnet_results = json.load(f)

# Comparar modelos
print(f"GRU4Rec Recall@20: {gru4rec_results['recall_20']:.4f}")
print(f"NextItNet Recall@20: {nextitnet_results['recall_20']:.4f}")
```

## 📊 Resumen de Resultados

### Resultados Principales de Evaluación

| Modelo    | Recall@5 | Recall@10 | Recall@20 | MRR@5  | MRR@10 | MRR@20 |
| --------- | -------- | --------- | --------- | ------ | ------ | ------ |
| GRU4Rec   | 0.3641   | 0.4345    | 0.4965    | 0.2552 | 0.2647 | 0.2690 |
| NextItNet | 0.3628   | 0.4276    | 0.4850    | 0.2599 | 0.2686 | 0.2726 |

### Análisis de Diversidad y Sesgo

| Modelo    | ILD@10 | ILD@20 | Sesgo Pop.@10 | Sesgo Pop.@20 |
| --------- | ------ | ------ | ------------- | ------------- |
| GRU4Rec   | 0.6218 | 0.6680 | 0.0489        | 0.0451        |
| NextItNet | 0.7534 | 0.7885 | 0.0513        | 0.0482        |

### Hallazgos Clave

- **GRU4Rec** muestra métricas de precisión ligeramente mejores (Recall@20: 0.4965 vs 0.4850)
- **NextItNet** demuestra diversidad significativamente mayor (ILD@20: 0.7885 vs 0.6680)
- Ambos modelos muestran patrones similares de sesgo de popularidad
- **Dimensión de embedding 256** proporciona el mejor rendimiento para ambos modelos

## 🔧 Requisitos

```bash
# Dependencias principales
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tqdm>=4.62.0
matplotlib>=3.4.0
seaborn>=0.11.0
kagglehub>=0.2.0
psutil>=5.8.0

# Instalación
pip install torch numpy pandas scikit-learn tqdm matplotlib seaborn kagglehub psutil
```