# Evaluación de Modelos de Recomendación Secuencial

Este repositorio implementa una evaluación reproducible de modelos de recomendación secuencial en los datasets RetailRocket y Yoochoose. Incluye preprocesamiento de datos, entrenamiento de modelos base y avanzados, análisis de sensibilidad, métricas de diversidad/novedad y evaluación de rendimiento integral.

## 📋 Tabla de Contenidos

- [Resumen](#resumen)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Modelos Implementados](#modelos-implementados)
- [Datasets](#datasets)
- [Métricas de Evaluación](#métricas-de-evaluación)
- [Uso](#uso)
- [Resultados](#resultados)
- [Requisitos](#requisitos)

## 🎯 Resumen

Este proyecto proporciona un framework de evaluación integral para sistemas de recomendación secuencial, enfocándose en:

- **Experimentos reproducibles** con configuraciones estandarizadas
- **Múltiples arquitecturas** (GRU4Rec y NextItNet) en múltiples datasets
- **Métricas comprehensivas** incluyendo precisión, diversidad y sesgo de popularidad
- **Análisis de sensibilidad** para dimensiones de embeddings
- **Resultados pre-computados** para análisis posterior sin re-entrenamiento

## 📁 Estructura del Repositorio

```
repo/
├── scripts/                              # Notebooks de Jupyter con experimentos completos
│   ├── GRU4Rec_RetailRocket_Optimized.ipynb    # Entrenamiento y evaluación de GRU4Rec en RetailRocket
│   ├── GRU4Rec_YC_Optimized.ipynb              # Entrenamiento y evaluación de GRU4Rec en Yoochoose
│   ├── NextItNet_RetailRocket_Optimized.ipynb  # Entrenamiento y evaluación de NextItNet en RetailRocket
│   └── NextItNet_YC_Optimized.ipynb            # Entrenamiento y evaluación de NextItNet en Yoochoose
├── outputs/                              # Resultados experimentales pre-computados
│   ├── GRU_RetailRocket/
│   │   ├── gru4rec_results.json         # Métricas principales de evaluación
│   │   └── gru4rec_sensitivity_analysis.json  # Análisis de dimensión de embeddings
│   ├── GRU4Rec_Yoochoose/
│   │   ├── gru4rec_yc_results.json      # Métricas principales de evaluación
│   │   └── gru4rec_yc_sensitivity_analysis.json # Análisis de dimensión de embeddings
│   ├── NextItNet_RetailRocket/
│   │   ├── nextitnet_results.json       # Métricas principales de evaluación
│   │   └── nextitnet_sensitivity_analysis.json # Análisis de dimensión de embeddings
│   └── NextItNet_Yoochoose/
│       ├── nextitnet_yc_results.json      # Métricas principales de evaluación
│       └── nextitnet_yc_sensitivity_analysis.json # Análisis de dimensión de embeddings
├── models/                               # Modelos entrenados guardados
├── .gitignore                           # Archivos ignorados por Git
└── README.md                            # Este archivo
```

### 📝 Directorio Scripts

Cada notebook es **autocontenido** y puede ejecutarse independientemente:

- **GRU4Rec_RetailRocket_Optimized.ipynb**: Implementación completa de GRU4Rec en RetailRocket
- **GRU4Rec_YC_Optimized.ipynb**: Implementación completa de GRU4Rec en Yoochoose
- **NextItNet_RetailRocket_Optimized.ipynb**: Implementación completa de NextItNet en RetailRocket
- **NextItNet_YC_Optimized.ipynb**: Implementación completa de NextItNet en Yoochoose

Cada notebook incluye:

- Descarga y preprocesamiento de datos
- Implementación del modelo
- Entrenamiento optimizado
- Evaluación comprehensiva
- Análisis de sensibilidad
- Guardado de resultados en JSON

### 💾 Directorio Outputs

Contiene resultados **pre-computados** para cada combinación modelo-dataset:

- **Métricas principales**: Recall@K, MRR@K, ILD@K, Popularity Bias@K
- **Análisis de sensibilidad**: Rendimiento vs dimensión de embeddings
- **Formato JSON**: Fácil carga para análisis posterior y visualizaciones

## 🧠 Modelos Implementados

### 1. **GRU4Rec**

- **Arquitectura**: Unidad Recurrente con Compuertas (GRU)
- **Fortalezas**: Captura dependencias temporales, manejo eficiente de secuencias
- **Implementación**: Optimizada con técnicas de regularización y muestreo

### 2. **NextItNet**

- **Arquitectura**: Convoluciones causales dilatadas
- **Fortalezas**: Paralelización eficiente, captura patrones locales y globales
- **Implementación**: Bloques residuales con normalización por capas

## 📊 Datasets

### 1. **RetailRocket**

- **Tipo**: E-commerce (eventos de navegación)
- **Tamaño**: ~2.6M interacciones, ~1.4M usuarios, ~235K items
- **Características**: Sesiones de navegación en tienda online

### 2. **Yoochoose**

- **Tipo**: RecSys Challenge 2015 (sesiones de e-commerce)
- **Tamaño**: Variable según fracción utilizada
- **Características**: División temporal estándar para evaluación

## 📈 Métricas de Evaluación

### Precisión

- **Recall@K**: Proporción de items relevantes recuperados en top-K
- **MRR@K**: Rango recíproco medio de items relevantes

### Diversidad

- **ILD@K**: Diversidad Intra-Lista usando similaridad coseno de embeddings
- Implementación estándar académica para comparaciones justas

### Sesgo de Popularidad

- **Popularity Bias@K**: Tendencia a recomendar items populares
- Medida de justicia y novedad en recomendaciones

### Análisis de Sensibilidad

- **Embedding Dimensions**: 16, 32, 64, 128, 256
- **Métrica objetivo**: Recall@20 para análisis de trade-offs

## 🚀 Uso

### Ejecución Individual

Cada notebook puede ejecutarse independientemente:

```bash
# Ejecutar en Jupyter/Colab
jupyter notebook repo/scripts/GRU4Rec_RetailRocket_Optimized.ipynb
```

### Análisis de Resultados

Los resultados pre-computados permiten análisis inmediato:

```python
import json

# Cargar resultados
with open('repo/outputs/GRU_RetailRocket/gru4rec_results.json') as f:
    gru_results = json.load(f)

with open('repo/outputs/NextItNet_RetailRocket/nextitnet_results.json') as f:
    nextit_results = json.load(f)

# Comparar métricas
print(f"GRU4Rec Recall@20: {gru_results['recall_20']:.4f}")
print(f"NextItNet Recall@20: {nextit_results['recall_20']:.4f}")
```

## 📊 Resultados

Los resultados están organizados por:

- **Modelo**: GRU4Rec, NextItNet
- **Dataset**: RetailRocket, Yoochoose
- **Métricas**: Precisión, Diversidad, Sesgo de Popularidad
- **Configuraciones**: Análisis de sensibilidad de embeddings

### Archivos de Resultados

```json
{
  "recall_5": 0.xxxx,
  "recall_10": 0.xxxx,
  "recall_20": 0.xxxx,
  "mrr_5": 0.xxxx,
  "mrr_10": 0.xxxx,
  "mrr_20": 0.xxxx,
  "ild_5": 0.xxxx,
  "ild_10": 0.xxxx,
  "ild_20": 0.xxxx,
  "popularity_bias_5": 0.xxxx,
  "popularity_bias_10": 0.xxxx,
  "popularity_bias_20": 0.xxxx
}
```

## 🔧 Requisitos

```
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tqdm>=4.62.0
matplotlib>=3.4.0
seaborn>=0.11.0
psutil>=5.8.0
```

Para RetailRocket:

```
kagglehub>=0.1.0
```

## 🎯 Características Destacadas

- ✅ **Reproducibilidad completa** con seeds fijos
- ✅ **Implementaciones optimizadas** para memoria y velocidad
- ✅ **Métricas estandarizadas** siguiendo literatura académica
- ✅ **Resultados pre-computados** para análisis rápido
- ✅ **Documentación detallada** en cada notebook
- ✅ **Análisis de sensibilidad** sistemático
- ✅ **Múltiples datasets** para validación cruzada

---

**Nota**: Este repositorio está diseñado para investigación académica y comparación justa de modelos de recomendación secuencial. Todos los experimentos siguen protocolos estándar de la literatura.

### 📊 Notebooks Disponibles

1. **GRU4Rec_RetailRocket_Optimized.ipynb**

   - Implementación de GRU4Rec con dataset RetailRocket
   - Arquitectura: Unidades Recurrentes con Compuertas (GRU)
   - Salida: `outputs/GRU_RetailRocket/`

2. **GRU4Rec_YC_Optimized.ipynb**

   - Implementación de GRU4Rec con dataset Yoochoose
   - Arquitectura: Unidades Recurrentes con Compuertas (GRU)
   - Salida: `outputs/GRU4Rec_Yoochoose/`

3. **NextItNet_RetailRocket_Optimized.ipynb**

   - Implementación de NextItNet con dataset RetailRocket
   - Arquitectura: Convoluciones dilatadas causales
   - Salida: `outputs/NextItNet_RetailRocket/`

4. **NextItNet_YC_Optimized.ipynb**
   - Implementación de NextItNet con dataset Yoochoose
   - Arquitectura: Convoluciones dilatadas causales
   - Salida: `outputs/NextItNet_Yoochoose/`
