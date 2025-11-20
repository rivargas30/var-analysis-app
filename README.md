# Análisis de Riesgo de Mercado: Metodologías VaR y Backtesting

Este proyecto es un Dashboard interactivo desarrollado en Python y Streamlit diseñado para calcular, comparar y validar el Valor en Riesgo (VaR) de un portafolio de activos bancarios (Colombia y EE.UU.).

La herramienta permite realizar un análisis profundo sobre la viabilidad de tres metodologías de VaR (Delta-Normal, Simulación Histórica y Monte Carlo) mediante pruebas estadísticas de Backtesting (Kupiec y López).

## Características Principales

* Carga y Normalización de Datos:

    * Extracción automática de precios vía yfinance.
    * Conversión de moneda (COP a USD) para comparabilidad entre activos (Banco de Bogotá, Davivienda, Bancolombia vs. JPMorgan, Bank of America, Wells Fargo).
    Análisis 
* Exploratorio de Datos (EDA):
    * Visualización de series de tiempo.
    * Estadísticas descriptivas (Asimetría, Curtosis).
    * Pruebas de Normalidad: Test de Jarque-Bera y gráficos QQ-Plots para detectar colas pesadas.
* Estimación de VaR (3 Enfoques):
    * Delta-Normal: Paramétrico (asume normalidad).
    * Simulación Histórica: No paramétrico (basado en cuantiles empíricos).
    * Simulación Monte Carlo: Estocástico (Movimiento Browniano Geométrico).
* Backtesting Riguroso:
    * Prueba de Kupiec ($LR_{uc}$): Valida la frecuencia de las excepciones (cobertura no condicional).
    * Función de Pérdida de López: Ranking de modelos basado en una función de pérdida para determinar la precisión relativa.
* Interfaz Interactiva: Control de parámetros (Nivel de Confianza $\alpha$) y navegación por secciones educativas.

## Tecnologías Utilizadas

El proyecto está construido con el siguiente stack tecnológico:

* Core: Python 3.x

* Frontend: Streamlit

* Datos: yfinance, Pandas, NumPy

* Estadística/Matemáticas: SciPy, Statsmodels

* Visualización: Matplotlib, Seaborn

## Uso

El proyecto es reproducible desde ![Stremalit VaR](https://var-analysis-46.streamlit.app/)

## Metodologías Detalladas1. 

1. Estimación del VaR
El dashboard calcula el VaR diario basado en un nivel de confianza ajustable (por defecto 99%):
* Delta-Normal: $VaR = -(\mu + z_{\alpha} \cdot \sigma)$. Rápido, pero sensible a la no normalidad.
* Simulación Histórica: Toma el percentil $\alpha$ de la distribución histórica de rendimientos. Captura colas pesadas reales.
* Monte Carlo: Genera 1000 caminos simulados de precios asumiendo un proceso estocástico para estimar el riesgo.

2. Backtesting (Validación)
* Test de Kupiec: Evalúa si el número de fallos (pérdidas > VaR) es estadísticamente igual al esperado.
    * Resultado: "Pasó" o "Falló" (basado en Chi-cuadrado).
* Prueba de López: Asigna un puntaje de pérdida.
    * Objetivo: Minimizar la función de pérdida. El modelo con el menor valor obtiene el Ranking #1.

## Resultados y Conclusiones del Estudio
Basado en los datos analizados (2020-2023), el dashboard arroja los siguientes hallazgos generales (visibles en la sección 7 de la app):

1. Rechazo de Normalidad: Las pruebas de Jarque-Bera indican que los rendimientos de los activos analizados no siguen una distribución normal (colas pesadas y asimetría).

2. Fallo del Modelo Paramétrico: El VaR Delta-Normal tiende a subestimar el riesgo y falla frecuentemente la prueba de Kupiec.

3. Modelo Recomendado: La Simulación Histórica demuestra ser la metodología más robusta, pasando las pruebas de cobertura y obteniendo los mejores puntajes en la función de pérdida de López.

## Referencias

El marco teórico y las pruebas implementadas se basan en la literatura financiera estándar y regulatoria:

Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models.

López, J. A. (1998). Methods for evaluating value-at-risk estimates.


## Licencia

Este es un proyecto construído por Alejandro Chavarro, Santiago Pinto y Ricardo Vargas en el marco de la clase de Teoría del Riesgo. Facultad de Estadística. Universidad Santo Tomás. 






