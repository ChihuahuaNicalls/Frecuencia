import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

def generar_tabla_frecuencias_con_formulas_imagen(num_datos=60):
    # Generar datos diferentes y únicos entre 10 y 99, y los organiza ascendentemente
    datos = [10, 11, 13, 17, 21, 22, 23, 24, 27, 29, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 48,
             49, 50, 51, 52, 53, 55, 56, 57, 58, 60, 65, 66, 68, 70, 72, 74, 75, 76, 77, 78, 79, 80, 82, 83,
             86, 87, 88, 89, 90, 91, 92, 94, 96, 97, 98, 99]
    datos.sort()
    datos = np.array(datos)

    #Calcular la Amplitud Total (A)
    rango_total = datos.max() - datos.min()

    #Calcular Número de Clases (K) y lo redondea hacia arriba
    k_float = math.sqrt(num_datos)
    k = math.ceil(k_float)

    print(f"\nDatos generados: {datos[:60]}")
    print(f"Valor Mínimo: {datos.min()}")
    print(f"Valor Máximo: {datos.max()}")
    print(f"Amplitud Total (A): {rango_total}")
    print(f"Numero de clases (K): {k_float:.2f}, rendondeado hacia arriba {k} clases")

    #Calcular Amplitud del Intervalo (H) y redondear hacia arriba
    amplitud_intervalo = rango_total / k
    amplitud_intervalo = np.ceil(amplitud_intervalo)

    print(f"Amplitud del Intervalo (H): {amplitud_intervalo}")

    # Límites de los intervalos
    limites_inferiores = [datos.min()]
    for i in range(1, k):
        limites_inferiores.append(limites_inferiores[-1] + amplitud_intervalo)

    limites_superiores = [limite + amplitud_intervalo for limite in limites_inferiores]

    if limites_superiores[-1] < datos.max():
        limites_superiores[-1] = datos.max() + 0.001

    rangos_variables = []
    for i in range(k):
        if i == k - 1:
            rango_str = f"[{int(limites_inferiores[i])} - {int(datos.max())}]"
        else:
            rango_str = f"[{int(limites_inferiores[i])} - {int(limites_superiores[i])})"
        rangos_variables.append(rango_str)

    # Calcular frecuencias

    # Frecuencia Absoluta
    frecuencia_absoluta = []
    for i in range(k):
        if i == k - 1:
            count = np.sum((datos >= limites_inferiores[i]) & (datos <= datos.max()))
        else:
            count = np.sum((datos >= limites_inferiores[i]) & (datos < limites_superiores[i]))
        frecuencia_absoluta.append(count)

    # Frecuencia Absoluta Acumulada
    frecuencia_absoluta_acumulada = np.cumsum(frecuencia_absoluta).tolist()

    # Frecuencia Relativa
    frecuencia_relativa = [fa / num_datos for fa in frecuencia_absoluta]

    # Frecuencia Relativa Acumulada
    frecuencia_relativa_acumulada = np.cumsum(frecuencia_relativa).tolist()

    puntos_medios_clase = [(limites_inferiores[i] + limites_superiores[i]) / 2 for i in range(k)]

    tabla_frecuencias = pd.DataFrame({
        'Rango': rangos_variables,
        'fi': frecuencia_absoluta,
        'Fi': frecuencia_absoluta_acumulada,
        'hi': [f'{x:.4f}' for x in frecuencia_relativa],
        'Hi': [f'{x:.4f}' for x in frecuencia_relativa_acumulada]
    })

    # Generar graficos ORIGINALES
    plt.style.use('ggplot')
    plt.figure(figsize=(16, 12))
    bar_color = '#64A53D'
    line_color = '#007ACC'

    #Frecuencia Absoluta
    ax1 = plt.subplot(2, 2, 1)
    ax1.bar(x=rangos_variables, height=frecuencia_absoluta, width=1.0, edgecolor='black', color=bar_color, align='center')
    ax1.set_title('Histograma de Frecuencia Absoluta')
    ax1.set_xticks(range(len(rangos_variables)))
    ax1.set_xticklabels(rangos_variables, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_ylim(0, max(frecuencia_absoluta) * 1.1)

    #Frecuencia Absoluta Acumulada
    ax2 = plt.subplot(2, 2, 2)
    ix = [limites_inferiores[0]] + [ls for ls in limites_superiores]
    iy = [0] + frecuencia_absoluta_acumulada

    ax2.plot(ix, iy, marker='o', linestyle='-', color=line_color, linewidth=2)
    ax2.set_title('Frecuencia Absoluta Acumulada')
    ax2.set_xticks(ix)
    ax2.set_xticklabels([f'{int(b)}' for b in ix], rotation=45, ha='right')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(0, num_datos * 1.1)

    for i in range(len(ix)):
        ax2.vlines(x=ix[i], ymin=0, ymax=iy[i], color='gray', linestyle=':', linewidth=1)

    #Frecuencia Relativa
    ax3 = plt.subplot(2, 2, 3)
    ax3.bar(x=rangos_variables, height=frecuencia_relativa, width=1.0, edgecolor='black', color=bar_color, align='center')
    ax3.set_title('Frecuencia Relativa')
    ax3.set_xticks(range(len(rangos_variables)))
    ax3.set_xticklabels(rangos_variables, rotation=45, ha='right')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    ax3.set_ylim(0, max(frecuencia_relativa) * 1.1)

    #Frecuencia Relativa Acumulada
    ax4 = plt.subplot(2, 2, 4)
    irx = [limites_inferiores[0]] + [ls for ls in limites_superiores]
    iry = [0] + frecuencia_relativa_acumulada

    ax4.plot(irx, iry, marker='o', linestyle='-', color=line_color, linewidth=2)
    ax4.set_title('Frecuencia Relativa Acumulada')
    ax4.set_xticks(irx)
    ax4.set_xticklabels([f'{int(b)}' for b in irx], rotation=45, ha='right')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.set_ylim(0, 1.1)

    for i in range(len(irx)):
        ax4.vlines(x=irx[i], ymin=0, ymax=iry[i], color='gray', linestyle=':', linewidth=1)

    plt.tight_layout(pad=10.0)
    plt.show()

    # Media para datos agrupados
    sum_ci_ni = sum(np.array(puntos_medios_clase) * np.array(frecuencia_absoluta))
    media_agrupada = (1 / num_datos) * sum_ci_ni
    print(f"Media para datos agrupados: {media_agrupada:.2f}")

    # Mediana para datos agrupados
    n_div_2 = num_datos / 2
    clase_mediana_idx = -1
    for i, Fa in enumerate(frecuencia_absoluta_acumulada):
        if Fa >= n_div_2:
            clase_mediana_idx = i
            break

    if clase_mediana_idx != -1:
        Lme = limites_inferiores[clase_mediana_idx]
        A = amplitud_intervalo
        f_mediana = frecuencia_absoluta[clase_mediana_idx]
        Fa_anterior = frecuencia_absoluta_acumulada[clase_mediana_idx - 1] if clase_mediana_idx > 0 else 0

        if f_mediana != 0:
            mediana_agrupada = Lme + A * ((n_div_2 - Fa_anterior) / f_mediana)
            print(f"Mediana para datos agrupados: {mediana_agrupada:.2f}")
        else:
            print("No se pudo calcular la mediana agrupada (frecuencia absoluta de la clase de la mediana es cero).")
            mediana_agrupada = np.nan # Asignar NaN si no se puede calcular
    else:
        print("No se encontró la clase de la mediana.")
        mediana_agrupada = np.nan # Asignar NaN si no se puede calcular

    # Moda para datos agrupados
    max_freq = 0
    moda_clases = []
    for freq in frecuencia_absoluta:
        if freq > max_freq:
            max_freq = freq
    for i, freq in enumerate(frecuencia_absoluta):
        if freq == max_freq:
            moda_clases.append(puntos_medios_clase[i])
            
    if len(moda_clases) == 1:
        moda_agrupada = moda_clases[0]
        print(f"Moda para datos agrupados: {moda_agrupada:.2f}")
    elif len(moda_clases) > 1:
        moda_agrupada = moda_clases # Es una lista de modas
        print(f"Datos bimodales o multimodales. Modas: {[f'{m:.2f}' for m in moda_clases]}")
    else:
        moda_agrupada = np.nan # Asignar NaN si no se encuentra
        print("No se encontró una moda (todos los valores tienen la misma frecuencia o no hay datos).")

    # Calcular asimetría y curtosis de los datos originales
    coef_asimetria = skew(datos)
    coef_kurtosis = kurtosis(datos) # Curtosis de Fisher (exceso de curtosis)

    print(f"\nCoeficiente de Asimetría (Skewness): {coef_asimetria:.4f}")
    print(f"Coeficiente de Curtosis (Kurtosis): {coef_kurtosis:.4f}")

    # --- Nuevo Gráfico para Medidas de Tendencia Central y Análisis de Distribución (copia del anterior) ---
    plt.figure(figsize=(10, 6))
    ax = plt.gca() # Get current axes

    # Dibujar el histograma
    # Asegúrate de que `bins_edges` tenga un elemento más que `limites_inferiores`
    bins_edges = np.array(limites_inferiores + [limites_superiores[-1]])
    ax.hist(datos, bins=bins_edges, edgecolor='black', alpha=0.7, color=bar_color) # La imagen image_4cab63.png muestra el histograma de distribución de datos con las medidas de tendencia central.
    
    # Marcar medidas de tendencia central en el histograma
    if not np.isnan(media_agrupada):
        ax.axvline(x=media_agrupada, color='red', linestyle='--', linewidth=2, label=f'Media: {media_agrupada:.2f}')
    if not np.isnan(mediana_agrupada):
        ax.axvline(x=mediana_agrupada, color='blue', linestyle='-.', linewidth=2, label=f'Mediana: {mediana_agrupada:.2f}')
    
    if isinstance(moda_agrupada, list):
        for m in moda_agrupada:
            ax.axvline(x=m, color='purple', linestyle=':', linewidth=2, label=f'Moda: {m:.2f}' if m == moda_agrupada[0] else '')
        handles, labels = ax.get_legend_handles_labels()
        unique_labels_dict = dict()
        for h, l in zip(handles, labels):
            if "Moda:" in l and l not in unique_labels_dict:
                unique_labels_dict[l] = h
            elif "Moda:" not in l:
                unique_labels_dict[l] = h
        
        unique_handles = list(unique_labels_dict.values())
        unique_labels = list(unique_labels_dict.keys())
        ax.legend(unique_handles, unique_labels)
    elif not np.isnan(moda_agrupada):
        ax.axvline(x=moda_agrupada, color='purple', linestyle=':', linewidth=2, label=f'Moda: {moda_agrupada:.2f}')
        ax.legend()
    else:
        ax.legend()
    
    ax.set_title(f'Distribución de Datos con Medidas de Tendencia Central\nAsimetría: {coef_asimetria:.2f}, Kurtosis: {coef_kurtosis:.2f}')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Frecuencia')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    rango_calculado = np.ptp(datos)
    print(f"Rango: {rango_calculado}")

    # Varianza
    varianza_muestral_datos_originales = np.var(datos, ddof=1)
    print(f"\nVarianza (Datos Originales - Muestral): {varianza_muestral_datos_originales:.2f}")
    sum_fi_diff_sq = sum(np.array(frecuencia_absoluta) * (np.array(puntos_medios_clase) - media_agrupada)**2)
    varianza_agrupada_calculada = sum_fi_diff_sq / (num_datos - 1)
    print(f"Varianza (Datos Agrupados - Calculada): {varianza_agrupada_calculada:.2f}")


    # Desviación estándar para datos agrupados
    desviacion_estandar_muestral_datos_originales = np.std(datos, ddof=1)
    print(f"\nDesviación Estándar (Datos Originales - Muestral): {desviacion_estandar_muestral_datos_originales:.2f}")
    desviacion_estandar_agrupada_calculada = math.sqrt(varianza_agrupada_calculada)
    print(f"Desviación Estándar (Datos Agrupados - Calculada): {desviacion_estandar_agrupada_calculada:.2f}")


    # Coeficiente de Variación
    if media_agrupada != 0:
        coef_variacion_agrupado_calculado = (desviacion_estandar_agrupada_calculada / media_agrupada) * 100
        print(f"\nCoeficiente de Variación (Datos Agrupados): {coef_variacion_agrupado_calculado:.2f}%")
    else:
        print("\nCoeficiente de Variación (Datos Agrupados): No se puede calcular (media es cero)")

    if np.mean(datos) != 0:
        coef_variacion_datos_originales = (desviacion_estandar_muestral_datos_originales / np.mean(datos)) * 100
        print(f"Coeficiente de Variación (Datos Originales): {coef_variacion_datos_originales:.2f}%")
    else:
        print("Coeficiente de Variación (Datos Originales): No se puede calcular (media es cero)")


    # Grafico de coeficiente de variacion con muestras
    num_muestras = 40
    tamano_muestra_simulada = num_datos
    coefs_variacion_muestras = []

    for _ in range(num_muestras):
        muestra_simulada = np.random.choice(datos, size=tamano_muestra_simulada, replace=True)
        
        mean_sample = np.mean(muestra_simulada)
        std_sample = np.std(muestra_simulada, ddof=1)
        
        if mean_sample != 0:
            cv_sample = (std_sample / mean_sample) * 100
            coefs_variacion_muestras.append(cv_sample)
        else:
            coefs_variacion_muestras.append(np.nan)

    plt.figure(figsize=(10, 6))
    x_indices = np.arange(1, num_muestras + 1)
    
    valid_cvs = [cv for cv in coefs_variacion_muestras if not np.isnan(cv)]
    valid_indices = [idx for idx, cv in zip(x_indices, coefs_variacion_muestras) if not np.isnan(cv)]

    plt.scatter(valid_indices, valid_cvs, color='grey', alpha=0.7, s=30)
    
    mean_cv_simulated = np.nanmean(coefs_variacion_muestras)
    if not np.isnan(mean_cv_simulated):
        plt.axhline(mean_cv_simulated, color='black', linestyle='-', linewidth=1)
        plt.text(x_indices[-1] + 0.5, mean_cv_simulated, f'Media CV: {mean_cv_simulated:.2f}%', va='center', color='black', fontsize=10)

    plt.title(f'{num_muestras} muestras simuladas de tamaño {tamano_muestra_simulada}\nCoeficiente de Variación')
    plt.xlabel('Muestra')
    plt.ylabel('Coeficiente de Variación (%)')
    
    if len(valid_cvs) > 0:
        min_cv = min(valid_cvs)
        max_cv = max(valid_cvs)
        plt.ylim(min_cv * 0.9, max_cv * 1.1 + (max_cv * 0.05 if max_cv > 0 else 1))
    else:
        plt.ylim(0, 10)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


    return tabla_frecuencias

# Generar y mostrar la tabla de frecuencias y los gráficos
tabla = generar_tabla_frecuencias_con_formulas_imagen(num_datos=60)
print("\nTabla de Frecuencias:")
print(tabla)