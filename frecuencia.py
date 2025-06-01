import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def generar_tabla_frecuencias_con_formulas_imagen(num_datos=60):
    # Generar datos diferentes y únicos entre 10 y 99, y los organiza ascendentemente
    datos = np.random.choice(range(10, 100), size=num_datos, replace=False)
    datos.sort()

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

    # Generar graficos
    plt.style.use('ggplot')
    plt.figure(figsize=(16, 12))
    bar_color = '#64A53D'
    line_color = '#007ACC'

    bins_edges = np.array(limites_inferiores + [limites_superiores[-1]])
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
        hme = frecuencia_relativa[clase_mediana_idx]
        Hme_prev = frecuencia_relativa_acumulada[clase_mediana_idx - 1] if clase_mediana_idx > 0 else 0.0

        if hme != 0:
            mediana_agrupada = Lme + (A / hme) * (0.5 - Hme_prev)
            print(f"Mediana para datos agrupados: {mediana_agrupada:.2f}")
        else:
            print("No se pudo calcular la mediana agrupada (frecuencia relativa de la clase de la mediana es cero).")
    else:
        print("No se encontró la clase de la mediana.")

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
        print(f"Moda para datos agrupados: {moda_clases[0]:.2f}")
    elif len(moda_clases) > 1:
        print(f"Datos bimodales o multimodales. Modas: {[f'{m:.2f}' for m in moda_clases]}")
    else:
        print("No se encontró una moda (todos los valores tienen la misma frecuencia o no hay datos).")


    return tabla_frecuencias

# Generar y mostrar la tabla de frecuencias
tabla = generar_tabla_frecuencias_con_formulas_imagen(num_datos=60)
print("\nTabla de Frecuencias:")
print(tabla)