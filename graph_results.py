#!/usr/bin/env python3
"""
Script para generar gráficos comparativos de todos los modelos
Genera 4 gráficos: Recall@K y MRR@K para Yoochoose y RetailRocket
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



def load_results(outputs_dir="outputs"):
    """
    Carga todos los archivos JSON de resultados desde el directorio outputs
    
    Returns:
        dict: Diccionario con estructura {dataset: {model: results}}
    """
    results = {
        'Yoochoose': {},
        'RetailRocket': {}
    }
    
    # Mapeo de directorios a modelos y datasets
    directory_mapping = {
        'GRU4Rec_Yoochoose': ('Yoochoose', 'GRU4Rec'),
        'GRU_RetailRocket': ('RetailRocket', 'GRU4Rec'),
        'NextItNet_Yoochoose': ('Yoochoose', 'NextItNet'),
        'NextItNet_RetailRocket': ('RetailRocket', 'NextItNet'),
        'SASRec_Yoochoose': ('Yoochoose', 'SASRec'),
        'SASRec_RetailRocket': ('RetailRocket', 'SASRec'),
        'ItemKNN_Yoochoose': ('Yoochoose', 'ItemKNN'),
        'ItemKNN_RetailRocket': ('RetailRocket', 'ItemKNN'),
        'Popularity_Yoochoose': ('Yoochoose', 'Popularity'),
        'Popularity_RetailRocket': ('RetailRocket', 'Popularity'),
        'Random_Yoochoose': ('Yoochoose', 'Random'),
        'Random_RetailRocket': ('RetailRocket', 'Random')
    }
    
    # Mapeo de archivos de resultados
    file_mapping = {
        'GRU4Rec': 'gru4rec_results.json',
        'NextItNet': 'nextitnet_results.json',
        'SASRec': 'SASRec_results.json',
        'ItemKNN': 'itemknn_results.json',
        'Popularity': 'popularity_results.json',
        'Random': 'random_results.json'
    }
    
    # Archivos específicos para Yoochoose
    yoochoose_files = {
        'GRU4Rec': 'gru4rec_yc_results.json',
        'NextItNet': 'nextitnet_yc_results.json',
        'SASRec': 'SASRec_yc_results.json',
        'ItemKNN': 'itemknn_yc_results.json',
        'Popularity': 'popularity_yc_results.json',
        'Random': 'random_yc_results.json'
    }
    
    # Archivos específicos para RetailRocket
    retailrocket_files = {
        'GRU4Rec': 'gru4rec_results.json',
        'NextItNet': 'nextitnet_results.json',
        'SASRec': 'SASRec_results.json',
        'ItemKNN': 'itemknn_retailrocket_results.json',
        'Popularity': 'popularity_retailrocket_results.json',
        'Random': 'random_retailrocket_results.json'
    }
    
    outputs_path = Path(outputs_dir)
    
    for dir_name, (dataset, model) in directory_mapping.items():
        dir_path = outputs_path / dir_name
        
        if dir_path.exists():
            # Determinar el nombre del archivo
            if dataset == 'Yoochoose':
                filename = yoochoose_files.get(model, file_mapping.get(model))
            elif dataset == 'RetailRocket':
                filename = retailrocket_files.get(model, file_mapping.get(model))
            else:
                filename = file_mapping.get(model)
            
            if filename:
                file_path = dir_path / filename
                
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            results[dataset][model] = data
                            print(f"✅ Cargado: {dataset} - {model}")
                    except Exception as e:
                        print(f"❌ Error cargando {file_path}: {e}")
                else:
                    print(f"⚠️  Archivo no encontrado: {file_path}")
    
    return results

def create_comparison_plots(results):
    """
    Crea los 4 gráficos comparativos
    """
    # Configuración de estilo
    plt.style.use('default')
    
    # Colores para cada modelo
    colors = {
        'Random': '#FF6B6B',      # Rojo
        'Popularity': '#4ECDC4',  # Verde azulado
        'ItemKNN': '#6B73FF',     # Azul
        'GRU4Rec': '#FFB347',     # Naranja
        'NextItNet': '#DDA0DD',   # Púrpura claro
        'SASRec': '#000000'       # Negro
    }
    
    # Marcadores para cada modelo
    markers = {
        'Random': 'o',
        'Popularity': 's',
        'ItemKNN': '^',
        'GRU4Rec': 'D',
        'NextItNet': 'v',
        'SASRec': 'x'
    }
    
    k_values = [5, 10, 20]
    
    # Crear figura con 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Títulos para cada subplot
    titles = [
        'Recall@K por Modelo - Yoochoose',
        'MRR@K por Modelo - Yoochoose',
        'Recall@K por Modelo - RetailRocket',
        'MRR@K por Modelo - RetailRocket'
    ]
    
    # Métricas y datasets para cada subplot
    plot_configs = [
        ('Yoochoose', 'recall', axes[0, 0]),
        ('Yoochoose', 'mrr', axes[0, 1]),
        ('RetailRocket', 'recall', axes[1, 0]),
        ('RetailRocket', 'mrr', axes[1, 1])
    ]
    
    for i, (dataset, metric, ax) in enumerate(plot_configs):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.set_xlabel('K (Top-K)', fontsize=12)
        ax.set_ylabel(f'{metric.upper()}@K', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Plotear cada modelo
        for model in ['Random', 'Popularity', 'ItemKNN', 'GRU4Rec', 'NextItNet', 'SASRec']:
            if model in results[dataset]:
                model_data = results[dataset][model]
                
                # Extraer valores para los k especificados
                values = []
                for k in k_values:
                    key = f'{metric}_{k}'
                    if key in model_data:
                        values.append(model_data[key])
                    else:
                        values.append(0)  # Valor por defecto si no existe
                
                # Plotear la línea
                ax.plot(k_values, values, 
                       color=colors[model], 
                       marker=markers[model], 
                       linewidth=2.5, 
                       markersize=8, 
                       label=model,
                       markerfacecolor='white',
                       markeredgewidth=2)
        
        # Configurar leyenda
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Configurar límites del eje Y
        ax.set_ylim(0, None)
        
        # Configurar ticks del eje X
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])
    
    # Ajustar layout
    plt.tight_layout(pad=3.0)
    
    # Título general
    # fig.suptitle('Comparación de Modelos de Recomendación - Recall@K y MRR@K', 
    #              fontsize=16, fontweight='bold', y=0.98)
    
    # Guardar el gráfico
    plt.savefig('figures/model_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Gráficos generados exitosamente!")
    print("📊 Archivo guardado: figures/model_comparison_results.png")

def create_diversity_bias_plots(results):
    """
    Crea gráficos de ILD y Popularity Bias en imágenes separadas
    """
    # Configuración de estilo
    plt.style.use('default')
    
    # Colores para cada modelo (mismos que en los gráficos principales)
    colors = {
        'Random': '#FF6B6B',      # Rojo
        'Popularity': '#4ECDC4',  # Verde azulado
        'ItemKNN': '#6B73FF',     # Azul
        'GRU4Rec': '#FFB347',     # Naranja
        'NextItNet': '#DDA0DD',   # Púrpura claro
        'SASRec': '#000000'       # Negro
    }
    
    # Crear directorio si no existe
    os.makedirs('figures', exist_ok=True)
    
    # ========== GRÁFICO 1: ILD ==========
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    
    titles_ild = [
        'ILD para Top-10 Recomendaciones - Yoochoose',
        'ILD para Top-10 Recomendaciones - RetailRocket'
    ]
    
    datasets = ['Yoochoose', 'RetailRocket']
    
    for i, (dataset, ax) in enumerate(zip(datasets, axes1)):
        ax.set_title(titles_ild[i], fontsize=14, fontweight='bold')
        ax.set_ylabel('Intra-List Diversity (ILD)', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        
        # Recopilar datos para ILD
        models = []
        values = []
        model_colors = []
        
        for model in ['Random', 'Popularity', 'ItemKNN', 'GRU4Rec', 'NextItNet', 'SASRec']:
            if model in results[dataset]:
                model_data = results[dataset][model]
                if 'ild_10' in model_data:
                    models.append(model)
                    values.append(model_data['ild_10'])
                    model_colors.append(colors[model])
        
        if models:
            # Crear gráfico de barras
            bars = ax.bar(models, values, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Agregar valores encima de las barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # Rotar etiquetas del eje X
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, style='italic')
    
    # Layout automático con constrained_layout
    # plt.tight_layout(pad=2.5)  # No necesario con constrained_layout=True
    
    # Título general para ILD
    # fig1.suptitle('Análisis de Diversidad Intra-Lista (ILD)', 
    #              fontsize=16, fontweight='bold', y=0.95)
    
    # Guardar gráfico de ILD
    plt.savefig('figures/ild_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()
    
    print("✅ Gráfico de ILD generado: figures/ild_analysis.png")

    return
    
    # ========== GRÁFICO 2: POPULARITY BIAS ==========
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    
    titles_bias = [
        'Popularity Bias para Top-10 - Yoochoose',
        'Popularity Bias para Top-10 - RetailRocket'
    ]
    
    for i, (dataset, ax) in enumerate(zip(datasets, axes2)):
        ax.set_title(titles_bias[i], fontsize=14, fontweight='bold')
        ax.set_ylabel('Popularity Bias', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        
        # Recopilar datos para Popularity Bias
        models = []
        values = []
        model_colors = []
        
        for model in ['Random', 'Popularity', 'ItemKNN', 'GRU4Rec', 'NextItNet', 'SASRec']:
            if model in results[dataset]:
                model_data = results[dataset][model]
                if 'popularity_bias_10' in model_data:
                    models.append(model)
                    values.append(model_data['popularity_bias_10'])
                    model_colors.append(colors[model])
        
        if models:
            # Crear gráfico de barras
            bars = ax.bar(models, values, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Agregar valores encima de las barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # Rotar etiquetas del eje X
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, style='italic')
    
    # Ajustar manualmente los márgenes para evitar problemas de layout
    fig2.subplots_adjust(top=0.88, bottom=0.15, left=0.08, right=0.95, wspace=0.3)
    
    # Título general para Popularity Bias
    # fig2.suptitle('Análisis de Sesgo de Popularidad', 
    #              fontsize=14, fontweight='bold', y=0.95)
    
    # Guardar gráfico de Popularity Bias
    plt.savefig('figures/popularity_bias_analysis.png', dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()
    
    print("✅ Gráfico de Popularity Bias generado: figures/popularity_bias_analysis.png")
    print("✅ Gráficos de diversidad y sesgo generados exitosamente!")

def print_summary(results):
    """
    Imprime un resumen de los resultados cargados
    """
    print("\n" + "="*70)
    print("📋 RESUMEN DE RESULTADOS CARGADOS")
    print("="*70)
    
    for dataset in ['Yoochoose', 'RetailRocket']:
        print(f"\n🗂️  {dataset}:")
        if dataset in results and results[dataset]:
            for model in results[dataset]:
                data = results[dataset][model]
                recall_20 = data.get('recall_20', 0)
                mrr_20 = data.get('mrr_20', 0)
                print(f"   ✅ {model:<12} - Recall@20: {recall_20:.4f}, MRR@20: {mrr_20:.4f}")
        else:
            print(f"   ❌ No hay datos disponibles")
    
    print("\n" + "="*70)

def load_sensitivity_results(outputs_dir="outputs"):
    """
    Carga los archivos de análisis de sensibilidad para modelos neuronales
    
    Returns:
        dict: Diccionario con estructura {dataset: {model: sensitivity_data}}
    """
    sensitivity_results = {
        'Yoochoose': {},
        'RetailRocket': {}
    }
    
    # Mapeo de directorios y archivos de sensibilidad
    sensitivity_mapping = {
        'GRU4Rec_Yoochoose': ('Yoochoose', 'GRU4Rec', 'gru4rec_yc_sensitivity_analysis.json'),
        'GRU_RetailRocket': ('RetailRocket', 'GRU4Rec', 'gru4rec_sensitivity_analysis.json'),
        'NextItNet_Yoochoose': ('Yoochoose', 'NextItNet', 'nextitnet_yc_sensitivity_analysis.json'),
        'NextItNet_RetailRocket': ('RetailRocket', 'NextItNet', 'nextitnet_sensitivity_analysis.json'),
        'SASRec_Yoochoose': ('Yoochoose', 'SASRec', 'SASRec_yc_sensitivity_analysis.json'),
        'SASRec_RetailRocket': ('RetailRocket', 'SASRec', 'SASRec_sensitivity_analysis.json')
    }
    
    outputs_path = Path(outputs_dir)
    
    for dir_name, (dataset, model, filename) in sensitivity_mapping.items():
        dir_path = outputs_path / dir_name
        file_path = dir_path / filename
        
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    sensitivity_results[dataset][model] = data
                    print(f"✅ Sensibilidad cargada: {dataset} - {model}")
            except Exception as e:
                print(f"❌ Error cargando sensibilidad {file_path}: {e}")
        else:
            print(f"⚠️  Archivo de sensibilidad no encontrado: {file_path}")
    
    return sensitivity_results

def create_embedding_sensitivity_plots(sensitivity_results):
    """
    Crea gráficos de análisis de sensibilidad de embeddings (2 gráficos, uno por dataset)
    """
    # Configuración de estilo
    plt.style.use('default')
    
    # Colores para cada modelo
    colors = {
        'GRU4Rec': '#FFB347',     # Naranja
        'NextItNet': '#DDA0DD',   # Púrpura claro
        'SASRec': '#000000'       # Negro
    }
    
    # Marcadores para cada modelo
    markers = {
        'GRU4Rec': 'D',
        'NextItNet': 'v',
        'SASRec': 'x'
    }
    
    # Crear figura con 2 subplots (1x2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Títulos para cada subplot
    titles = [
        'Sensibilidad: Recall@20 vs Embedding Dimension - Yoochoose',
        'Sensibilidad: Recall@20 vs Embedding Dimension - RetailRocket'
    ]
    
    datasets = ['Yoochoose', 'RetailRocket']
    
    for i, (dataset, ax) in enumerate(zip(datasets, axes)):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.set_xlabel('Embedding Dimension', fontsize=12)
        ax.set_ylabel('Recall@20', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Plotear cada modelo
        for model in ['GRU4Rec', 'NextItNet', 'SASRec']:
            if model in sensitivity_results[dataset]:
                data = sensitivity_results[dataset][model]
                
                # Extraer dimensiones y scores
                embedding_dims = data.get('embedding_dimensions', [])
                recall_scores = data.get('recall_20_scores', {})
                
                # Convertir a listas ordenadas
                dims = []
                scores = []
                for dim in embedding_dims:
                    dim_str = str(dim)
                    if dim_str in recall_scores:
                        dims.append(dim)
                        scores.append(recall_scores[dim_str])
                
                if dims and scores:
                    # Plotear la línea
                    ax.plot(dims, scores, 
                           color=colors[model], 
                           marker=markers[model], 
                           linewidth=3, 
                           markersize=10, 
                           label=model,
                           markerfacecolor='white',
                           markeredgewidth=2)
        
        # Configurar leyenda
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Configurar límites del eje Y para mejor visualización
        ax.set_ylim(0.45, None)
        
        # Configurar ticks del eje X
        if sensitivity_results[dataset]:
            # Usar las dimensiones del primer modelo disponible
            first_model = list(sensitivity_results[dataset].keys())[0]
            dims = sensitivity_results[dataset][first_model].get('embedding_dimensions', [32, 64, 128, 256])
            ax.set_xticks(dims)
            ax.set_xticklabels([str(d) for d in dims])
    
    # Ajustar layout
    plt.tight_layout(pad=3.0)
    
    # Título general
    # fig.suptitle('Análisis de Sensibilidad de Embedding Dimensions', 
    #             fontsize=16, fontweight='bold', y=1.02)
    
    # Crear directorio si no existe
    os.makedirs('figures', exist_ok=True)
    
    # Guardar el gráfico
    plt.savefig('figures/embedding_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Gráficos de sensibilidad de embeddings generados exitosamente!")
    print("📊 Archivo guardado: figures/embedding_sensitivity_analysis.png")

def main():
    """
    Función principal
    """
    print("🚀 Iniciando generación de gráficos comparativos...")
    print("📂 Cargando resultados desde directorio 'outputs'...")
    
    # Cargar resultados
    results = load_results()
    
    # Imprimir resumen
    print_summary(results)
    
    # Verificar que tengamos datos
    total_models = sum(len(results[dataset]) for dataset in results)
    if total_models == 0:
        print("❌ No se encontraron archivos de resultados!")
        print("💡 Asegúrate de que los archivos JSON estén en el directorio 'outputs'")
        return
    
    print(f"\n📊 Generando gráficos comparativos para {total_models} modelos...")
    
    # Crear directorio de figuras si no existe
    os.makedirs('figures', exist_ok=True)
    
    try:
        # Crear gráficos de Recall y MRR
        print("📈 Generando gráficos de Recall@K y MRR@K...")
        create_comparison_plots(results)
        
        # Limpiar memoria
        plt.close('all')
        
        print(f"\n🎨 Generando gráficos de diversidad y sesgo...")
        
        # Crear gráficos de ILD y Popularity Bias
        create_diversity_bias_plots(results)
        
        # Limpiar memoria
        plt.close('all')
        
        print(f"\n📈 Generando gráficos de sensibilidad de embeddings...")
        
        # Cargar y crear gráficos de sensibilidad
        sensitivity_results = load_sensitivity_results()
        create_embedding_sensitivity_plots(sensitivity_results)
        
        # Limpiar memoria final
        plt.close('all')
        
        print("\n🎉 ¡Proceso completado exitosamente!")
        print("📁 Todos los gráficos guardados en el directorio 'figures/'")
        
    except Exception as e:
        print(f"❌ Error durante la generación de gráficos: {e}")
        print("💡 Intenta ejecutar el script nuevamente o revisa los archivos de datos")

if __name__ == "__main__":
    main()
