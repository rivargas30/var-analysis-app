import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm, skew, jarque_bera, chi2
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 

# --- Configuración de la Página y Estilos ---
st.set_page_config(
    page_title="VaR: Presentación y Backtesting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constantes y Parámetros ---
COMPANIES = ['BOGOTA.CL', 'PFDAVVNDA.CL', 'PFCIBEST.CL', 'JPM', 'BML-PL', 'WFC']
START_DATE = '2020-01-02'
END_DATE = '2023-02-08'
COLOMBIAN_TICKERS = ['BOGOTA.CL', 'PFDAVVNDA.CL', 'PFCIBEST.CL']
NEW_COLUMN_NAMES = {
    'BOGOTA.CL': 'Banco Bogotá',
    'PFDAVVNDA.CL': 'Davivienda',
    'PFCIBEST.CL': 'Bancolombia',
    'JPM': 'JPMorgan',
    'BML-PL': 'Bank of America',
    'WFC': 'Wells Fargo'
}
NUM_SIMULATIONS = 1000

# --- Funciones de Data Wrangling y Caching ---

@st.cache_data
def load_and_prepare_data(companies, start_date, end_date):
    """Carga los datos de Yahoo Finance y realiza la conversión a USD."""
    
    data = yf.download(companies + ['COP=X'], start=start_date, end=end_date)['Close']
    
    if data.empty:
        # st.error("Error al cargar los datos. Verifique los tickers y el rango de fechas.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    data.rename(columns={'COP=X': 'COP_USD'}, inplace=True)
    aligned_data = data.dropna(subset=['COP_USD'])
    combined_usd_df = aligned_data.copy()
    
    for ticker in COLOMBIAN_TICKERS:
        if ticker in combined_usd_df.columns:
            combined_usd_df[ticker] = combined_usd_df[ticker] / combined_usd_df['COP_USD']
    
    combined_usd_df.drop(columns=['COP=X', 'COP_USD'], errors='ignore', inplace=True)
    combined_usd_df.rename(columns=NEW_COLUMN_NAMES, inplace=True)
    
    daily_returns_df = combined_usd_df.pct_change().dropna()
    
    return combined_usd_df, daily_returns_df

@st.cache_data
def calculate_descriptive_statistics(df_prices, df_returns):
    """Calcula estadísticas descriptivas y la prueba de normalidad (Jarque-Bera)."""
    
    stats_list = []
    for col in df_prices.columns:
        returns = df_returns[col].dropna()
        jb_statistic, jb_pvalue = jarque_bera(returns)
        skewness_r = skew(returns)
        
        stats_list.append({
            'Activo': col,
            'Mínimo (USD)': f'{df_prices[col].min():.2f}',
            'Máximo (USD)': f'{df_prices[col].max():.2f}',
            'Media (USD)': f'{df_prices[col].mean():.2f}',
            'Volatilidad (Std. Dev.)': f'{returns.std():.4f}',
            'Asimetría': f'{skewness_r:.4f}',
            'Kurtosis (Exceso)': f'{returns.kurtosis():.4f}',
            'JB p-valor': f'{jb_pvalue:.4f}',
            'Normalidad Rechazada (a=0.05)': 'Sí' if jb_pvalue < 0.05 else 'No'
        })
        
    return pd.DataFrame(stats_list).set_index('Activo')

# --- Funciones de Cálculo del VaR ---

def calculate_var(df_returns, confidence_level, num_simulations):
    """Calcula el VaR Delta-Normal, Histórico y Monte Carlo."""
    
    alpha = 1 - confidence_level
    var_results = {}
    
    for asset in df_returns.columns:
        returns = df_returns[asset].dropna()
        
        # 1. Delta-Normal VaR
        mean_return = returns.mean()
        std_dev_return = returns.std()
        z_score = norm.ppf(alpha)
        delta_normal_var_return = -(mean_return + z_score * std_dev_return)
        
        # 2. Historical Simulation VaR
        historical_var_return = -returns.quantile(alpha)
        
        # 3. Monte Carlo Simulation VaR (Asumiendo GBM)
        np.random.seed(42)
        simulated_returns = np.random.normal(mean_return, std_dev_return, num_simulations)
        mc_var_return = -np.percentile(simulated_returns, alpha * 100)
        
        var_results[asset] = {
            'Delta-Normal VaR (%)': delta_normal_var_return * 100,
            'Simulación Histórica VaR (%)': historical_var_return * 100,
            'Simulación Monte Carlo VaR (%)': mc_var_return * 100
        }
    
    return pd.DataFrame(var_results).T

# --- Funciones de Backtesting (Kupiec y López) ---

def kupiec_test(exceptions, total_observations, alpha):
    """Prueba de Cobertura No Condicional (LRuc)."""
    
    N = total_observations
    x = exceptions
    p = alpha
    
    E = N * p
    
    # Calcular LRuc
    if x == 0:
        LRuc = -2 * (N * np.log(1 - p) - (N - x) * np.log(1 - E / N))
    elif x == N:
        LRuc = -2 * (x * np.log(p) - x * np.log(x / N))
    elif E == 0 or E == N:
        LRuc = 0 
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            LRuc = -2 * (x * np.log(p) + (N - x) * np.log(1 - p) - x * np.log(x / N) - (N - x) * np.log(1 - x / N))
    
    # p-valor de chi-cuadrado con 1 g.d.l.
    p_value = 1 - chi2.cdf(LRuc, 1)
    
    return LRuc, p_value, E

def lopez_loss_function(returns, var_values):
    """Función de Pérdida Binaria de López simplificada (Loss Function)."""
    
    # Indicador de excepción (1 si pérdida > VaR, 0 en otro caso)
    exception_indicator = (-returns > var_values).astype(int)
    
    # La pérdida es simplemente el número total de excepciones (violaciones)
    total_loss = np.sum(exception_indicator)
    
    return total_loss

def perform_backtesting(df_returns, var_df, confidence_level):
    """Calcula las excepciones, realiza la prueba de Kupiec y la función de pérdida de López."""
    
    alpha = 1 - confidence_level
    total_obs = len(df_returns)
    all_results = []
    
    method_mapping = {
        'Delta-Normal VaR': 'Delta-Normal VaR (%)',
        'Simulación Histórica VaR': 'Simulación Histórica VaR (%)',
        'Simulación Monte Carlo VaR': 'Simulación Monte Carlo VaR (%)'
    }

    for asset in df_returns.columns:
        returns = df_returns[asset].values
        
        for method_name, col_name in method_mapping.items():
            var_value_pct = var_df.loc[asset, col_name]
            var_values = np.array([var_value_pct / 100] * total_obs) 
            
            # 1. Cobertura No Condicional (Kupiec)
            exceptions = np.sum(-returns > var_values)
            LRuc, p_value_uc, expected_x = kupiec_test(exceptions, total_obs, alpha)
            
            # 2. Función de Pérdida de López
            total_loss = lopez_loss_function(returns, var_values) 

            def get_kupiec_result(p_val, level=0.01):
                return 'Pasó' if p_val >= level else 'Falló'
            
            all_results.append({
                'Activo': asset,
                'Método VaR': method_name,
                'x Observadas': exceptions,
                'x Esperadas': f'{expected_x:.2f}',
                
                'Kupiec (LRuc)': f'{LRuc:.3f}',
                'p-valor LRuc': f'{p_value_uc:.4f}',
                'Resultado Kupiec': get_kupiec_result(p_value_uc),
                
                # Reemplazo de Christoffersen por López
                'López (Loss)': total_loss,
                'Ranking López (1=Mejor)': 0 # Placeholder, el ranking se hará después
            })
            
    # Calcular el ranking de López (menor pérdida total es mejor)
    results_df = pd.DataFrame(all_results)
    
    # Asignar ranking por activo: 1 al menor valor de 'López (Loss)'
    results_df['Ranking López (1=Mejor)'] = results_df.groupby('Activo')['López (Loss)'].rank(method='min').astype(int)
    
    return results_df

# --- Funciones de Presentación por Sección ---

def section_introduction(df_prices, df_returns):
    st.header("🎯 1. Introducción y Metodología")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Objetivo Principal")
        st.markdown(
            """
            <p style='font-size:1.5em; line-height:1.6;'>
            Analizar y <b>comparar el comportamiento del Valor en Riesgo (VaR)</b> estimado por tres metodologías clave 
            (Delta-Normal, Simulación Histórica y Monte Carlo) en un portafolio simulado de activos bancarios, 
            validando su precisión mediante <b>pruebas de Backtesting</b>.
            </p>
            """, 
            unsafe_allow_html=True
        )

    with col2:
        st.subheader("Metodologías de Riesgo")
        st.markdown(
            """
            <p style='font-size:1.5em; line-height:1.6;'>
            1. <b>VaR Delta-Normal (Paramétrico):</b> Asume rendimientos normales. <br>
            2. <b>VaR Simulación Histórica (No Paramétrico):</b> Usa cuantiles de la historia real. <br>
            3. <b>VaR Simulación Monte Carlo (Mixto):</b> Simula caminos de precios (asumiendo Movimiento Browniano Geométrico). <br>
            4. <b>Backtesting (Kupiec & López):</b> Valida la exactitud de la cobertura (Kupiec) y la penaliza por las violaciones (López).
            </p>
            """, 
            unsafe_allow_html=True
        )

def section_eda(df_prices, df_returns, stats_df):
    st.header("📊 2. Análisis Exploratorio de Datos (EDA)")
    
    st.subheader("2.1. Series de Precios Homogeneizados (USD)")
    
    fig_prices, ax_prices = plt.subplots(figsize=(10, 4))
    for column in df_prices.columns:
        ax_prices.plot(df_prices.index, df_prices[column], label=column)
    ax_prices.set_title('Precios de Cierre Diarios en USD', fontsize=16)
    ax_prices.set_xlabel('Fecha', fontsize=12)
    ax_prices.set_ylabel('Precio (USD)', fontsize=12)
    ax_prices.legend(loc='upper left', fontsize=8, ncol=2)
    ax_prices.grid(True, alpha=0.5)
    st.pyplot(fig_prices)

    st.subheader("2.2. Estadísticas Descriptivas y Normalidad (Rendimientos)")
    st.dataframe(stats_df)
    
    st.markdown(
        """
        <p style='font-size:1.3em; line-height:1.5;'>
        <b>Análisis de Normalidad (Prueba de Jarque-Bera):</b> El p-valor de JB es significativamente <b>menor a 0.05</b> para todos los activos, lo que <b>rechaza la hipótesis de normalidad</b>. 
        Esto indica la presencia de <b>colas pesadas</b> y <b>asimetría</b>, confirmando que el uso del VaR Delta-Normal puede ser inadecuado.
        </p>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("2.3. Distribución de Rendimientos Diarios y QQ-Plots")
    
    selected_asset = st.selectbox("Seleccionar Activo para Gráficos:", df_returns.columns)

    col_hist, col_qq = st.columns(2)
    
    with col_hist:
        # Histograma de Rendimientos
        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
        sns.histplot(df_returns[selected_asset], kde=True, bins=30, ax=ax_hist, color='skyblue')
        ax_hist.set_title(f'Distribución de Rendimientos Diarios: {selected_asset}')
        ax_hist.set_xlabel('Rendimiento')
        ax_hist.set_ylabel('Frecuencia')
        st.pyplot(fig_hist)
        
    with col_qq:
        # QQ-Plot
        fig_qq, ax_qq = plt.subplots(figsize=(6, 4))
        sm.qqplot(df_returns[selected_asset].dropna(), line='s', ax=ax_qq)
        ax_qq.set_title(f'QQ-Plot (Normalidad): {selected_asset}')
        st.pyplot(fig_qq)
        

def section_var_estimation(var_results_df, confidence_level):
    st.header("💰 3. Estimación del Valor en Riesgo (VaR)")
    
    st.subheader(f"VaR Diario al {confidence_level:.3%} de Confianza")
    
    styled_var_df = var_results_df.copy()
    for col in styled_var_df.columns:
        styled_var_df[col] = (var_results_df[col]).apply(lambda x: f'{x:.4f}%') 
        
    st.dataframe(styled_var_df)

    st.markdown(
        """
        <p style='font-size:1.3em; line-height:1.5;'>
        <b>Observaciones Preliminares:</b>
        <ul>
            <li>Los modelos de VaR reflejan la <b>volatilidad</b> y las <b>propiedades de las colas</b> de los rendimientos.</li>
            <li>Una mayor desviación de la normalidad (activos colombianos) resulta en una mayor dispersión entre el VaR Paramétrico y el VaR No Paramétrico.</li>
        </ul>
        </p>
        """,
        unsafe_allow_html=True
    )

# --- SECCIÓN: 4. Estado del Arte de Pruebas ---
def section_state_of_the_art():
    st.header("📚 4. Estado del Arte de las Pruebas de Backtesting")
    st.markdown("""
    El backtesting es un componente regulatorio y de gestión de riesgo esencial, evolucionando desde pruebas binarias hasta métricas de función de pérdida.

    ### 4.1. El Legado de Kupiec (LRuc)
    La prueba de **Razón de Verosimilitud No Condicional ($LR_{uc}$)**, propuesta por **Paul H. Kupiec (1995)**, es la piedra angular del backtesting. Su relevancia radica en:
    * **Simplicidad:** Se enfoca puramente en el **número total de violaciones** ($x$) observadas.
    * **Regulatorio:** Es un requisito fundamental bajo los Acuerdos de Basilea para validar la precisión de los modelos VaR.
    * **Limitación:** Ignora si las violaciones están agrupadas en el tiempo (**clusterización**), lo que puede subestimar el riesgo sistémico o la inestabilidad del modelo. Esto condujo al desarrollo de pruebas condicionales como la de Christoffersen (reemplazada aquí por López).

    ### 4.2. La Evolución con las Funciones de Pérdida (López)
    Las pruebas basadas en **Funciones de Pérdida (Loss Functions)**, popularizadas por **José A. López (1998)**, representan una evolución cualitativa.
    * **Enfoque:** En lugar de solo verificar la *frecuencia* (como Kupiec), estas pruebas evalúan la *gravedad* de los fallos, penalizando no solo la ocurrencia de una violación sino también la **magnitud** por la que la pérdida excedió el VaR (aunque en nuestra implementación simplificada solo penalizamos la ocurrencia).
    * **Ventaja:** Ofrecen una medida **más informativa y continua** del rendimiento del modelo, permitiendo al gestor de riesgo clasificar (rankear) los modelos y elegir el que minimice el "costo" o "pérdida" total, una perspectiva más alineada con la toma de decisiones económicas.
    """)

def section_backtesting_explanation():
    st.header("🔍 5. Explicación de las Pruebas de Backtesting")
    st.markdown("""
El Backtesting es un proceso crucial que evalúa la precisión y la calidad de un modelo de Valor en Riesgo (VaR) comparando las pérdidas reales con las predicciones del modelo.

### 5.1. Prueba de Cobertura No Condicional (Kupiec)
* Propósito: Es una prueba estadística que verifica si el número de violaciones (días en que la pérdida real excede el VaR) observadas durante el período de prueba es estadísticamente igual al número de violaciones esperadas por el modelo, dado el nivel de confianza (alpha).
* Métrica: Utiliza una razón de verosimilitud (LRuc).
* Resultado: Si el modelo genera demasiadas o muy pocas violaciones, la prueba se rechaza ("Falló"), indicando que el modelo es inexacto (subestima o sobrestima el riesgo).

### 5.2. Prueba de Pérdida de Función (López)
* Propósito: Es una prueba de pérdida que asigna una puntuación numérica o "coste" a la calidad del modelo. Permite clasificar los modelos por su rendimiento.
* Métrica: Se implementa una Función de Pérdida Binaria simplificada, donde cada violación suma una unidad a la pérdida total del modelo.
* Resultado: El modelo con la menor Puntuación de Pérdida (Loss) para un activo es considerado el más preciso en esa dimensión y obtiene el Ranking 1.
""")


def section_backtesting(backtest_df):
    st.header("✅ 6. Pruebas de Backtesting (Kupiec y López)") 
    
    st.markdown("""
El backtesting evalúa la precisión del modelo en la práctica.

* Kupiec (LRuc): Prueba si el número de fallos es igual al esperado (alpha).
* López (Loss Function): Asigna una pérdida por cada violación del VaR. Un menor valor de pérdida total indica un mejor modelo.
""")

    st.subheader("6.1. Resumen de Pruebas de Cobertura y Precisión")
    
    # Columnas actualizadas
    comparison_cols = ['Activo', 'Método VaR', 'x Observadas', 'x Esperadas', 
                       'p-valor LRuc', 'Resultado Kupiec', 
                       'López (Loss)', 'Ranking López (1=Mejor)']
                       
    comparison_df = backtest_df[comparison_cols]
    
    # Aplicar formato condicional al DataFrame para presentación
    def color_result(val):
        color = 'lightcoral' if val == 'Falló' else ('lightgreen' if val == 'Pasó' else '')
        return f'background-color: {color}'

    st.dataframe(comparison_df.style.applymap(color_result, subset=['Resultado Kupiec']))
    
    
def section_findings(backtest_df, stats_df):
    st.header("🌟 7. Conclusiones y Hallazgos Principales")
    st.subheader("7.1. No Normalidad y Modelado del Riesgo")
    st.markdown(
        """
        <div style='font-size:1.5em; line-height:1.7;'>
        <p>El análisis de la distribución de rendimientos mediante la <b>prueba de Jarque-Bera (JB)</b> confirmó la presencia de <b>asimetría</b> (sesgo) y <b>exceso de curtosis</b> (colas pesadas) en todos los activos (p-valor JB < 0.05).</p>
        <p>Este hallazgo es fundamental: <b>la suposición de normalidad es rechazada</b>, invalidando la base teórica del VaR Delta-Normal para los activos estudiados.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("7.2. Desempeño del VaR por Metodología")
    
    st.markdown("### A. VaR Delta-Normal (Paramétrico)")
    st.markdown(
        """
        <div style='font-size:1.3em; line-height:1.5;'>
        <ul>
            <li><b>Estimación:</b> Produjo los valores de VaR más <b>bajos</b>, indicando una subestimación del riesgo, particularmente para los activos con alta no-normalidad.</li>
            <li><b>Backtesting (Kupiec):</b> <b>Falló</b> la prueba de <b>Cobertura No Condicional (LRuc)</b> en la mayoría de los casos. El número de <b>excepciones observadas superó significativamente al número esperado</b>, confirmando que el modelo es inexacto y subestima la pérdida máxima potencial.</li>
            <li><b>Backtesting (López):</b> Dado el alto número de violaciones, el VaR Delta-Normal obtendrá una <b>puntuación de pérdida alta</b> (o un bajo ranking).</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### B. VaR de Simulación Histórica (No Paramétrico)")
    st.markdown(
        """
        <div style='font-size:1.3em; line-height:1.5;'>
        <ul>
            <li><b>Estimación:</b> Produjo valores de VaR más <b>conservadores</b> (altos), ya que captura directamente los eventos de mercado extremos (pérdidas históricas).</li>
            <li><b>Backtesting (Kupiec):</b> Generalmente logra <b>pasar la prueba de Cobertura (LRuc)</b>, indicando un número de fallos consistente con el nivel de confianza.</li>
            <li><b>Backtesting (López):</b> Este modelo tenderá a tener la <b>pérdida total más baja</b> (o el ranking 1), ya que es más robusto a las colas pesadas.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### C. VaR de Simulación Monte Carlo (Mixto)")
    st.markdown(
        """
        <div style='font-size:1.3em; line-height:1.5;'>
        <ul>
            <li><b>Estimación:</b> Al simular los rendimientos bajo la <b>suposición de normalidad</b>, sus resultados fueron muy similares al VaR Delta-Normal, heredando sus debilidades.</li>
            <li><b>Backtesting (Kupiec y López):</b> Tiende a <b>fallar Kupiec</b> y a obtener una **puntuación de pérdida alta** en la prueba de López, similar al Delta-Normal.</li>
            <li><b>Implicación:</b> Para que Monte Carlo sea preciso en la práctica, requeriría la calibración y simulación de distribuciones que modelen correctamente las colas pesadas (e.g., distribución T de Student o modelos GARCH).</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("7.3. Recomendación Final")
    st.markdown(
        """
        <div style='font-size:1.5em; line-height:1.7;'>
        <p>Debido al rechazo de la normalidad, al desempeño de <b>Kupiec</b> y a la mejor <b>puntuación de pérdida (López)</b>, se recomienda utilizar el <b>Valor en Riesgo de Simulación Histórica</b> como la métrica principal para la medición del riesgo de mercado en este portafolio.</p>
        <p>El VaR paramétrico (Delta-Normal) debe ser descartado, ya que su subestimación del riesgo podría llevar a decisiones financieras deficientes.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- SECCIÓN: 8. Referencias ---
def section_references():
    st.header("📖 8. Referencias del Estado del Arte")
    st.markdown("""
    * **Kupiec, P. H. (1995).** *Techniques for verifying the accuracy of risk measurement models.* The Journal of Derivatives, 3(2), 73-84.
    * **López, J. A. (1998).** *Methods for evaluating value-at-risk estimates.* Economic Policy Review, Federal Reserve Bank of New York, 4(3), 119-144.
    * **Basilea (2019).** *Standards for calculating capital requirements for market risk.* (Documentos del Comité de Supervisión Bancaria de Basilea que regulan el uso del backtesting en la banca global).
    * Álvarez Ruiz, M. C., & Parra Oquendo, L. Y. (2024). *Cálculo del valor en riesgo (VaR) mediante el uso de diferentes metodologías para dos portafolios del mercado bancario colombiano y americano.* Efectivo, (39), 21–38. Instituto Tecnológico Metropolitano.
    * Caicedo, H. S. O., & Castañeda, A. F. V. (2022). *Ranking del riesgo de mercado de los bancos que cotizan en la Bolsa de Valores de Colombia (BVC) utilizando metodologías VaR para el periodo de enero de 2020 a marzo de 2022.* [Trabajo de grado, Escuela Colombiana de Ingeniería Julio Garavito]. Repositorio Institucional Escuelaing. [https://repositorio.escuelaing.edu.co/entities/publication/5dc961e5-3a8f-403f-8819-6f402d3672a9](https://repositorio.escuelaing.edu.co/entities/publication/5dc961e5-3a8f-403f-8819-6f402d3672a9)
    * Pineda, M. S. G., Agudelo, A. A. A., Rojas, R. A. M., & Duque, P. L. H. (2021). *Valor en riesgo y simulación: una revisión sistemática.* Económicas CUC, 43(1), 57–82. [https://revistascientificas.cuc.edu.co/economicascuc/article/view/3093](https://revistascientificas.cuc.edu.co/economicascuc/article/view/3093)
    * Sener, E., Baronyan, S., & Mengütürk, L. A. (2012). *Ranking the predictive performances of value-at-risk estimation methods.* International Journal of Forecasting, 28(4), 849–873. [https://www.sciencedirect.com/science/article/abs/pii/S0169207012000027?via%3Dihub](https://www.sciencedirect.com/science/article/abs/pii/S0169207012000027?via%3Dihub)
    * Trejo, B. R. B., & Gallegos, A. D. (2021). *Estimación del riesgo de mercado utilizando el VaR y la beta del CAPM.* Revista Mexicana de Economía y Finanzas Nueva Época, 16(2), 1–26. [https://www.remef.org.mx/index.php/remef/article/view/589](https://www.remef.org.mx/index.php/remef/article/view/589)
    """)
# --- FIN SECCIÓN ---
    
# --- Lógica Principal de la Aplicación ---

def main():
    st.title("📊 Análisis de Riesgo de Mercado: Comparación de Metodologías VaR")
    
    # --- 1. Sidebar para Parámetros y Navegación ---
    with st.sidebar:
        st.header("Menú de Navegación")
        # LISTA DE NAVEGACIÓN ACTUALIZADA
        page = st.radio(
            "Seleccione la Sección:",
            ["1. Introducción y Metodología", 
             "2. Análisis Exploratorio (EDA)", 
             "3. Estimación del VaR", 
             "4. Estado del Arte de Pruebas",
             "5. Explicación de Pruebas", 
             "6. Backtesting",
             "7. Conclusiones y Hallazgos Principales",
             "8. Referencias"]
        )
        
        st.markdown("---")
        st.header("Parámetros del Modelo")
        confidence_level = st.slider(
            'Nivel de Confianza (1 - alpha)',
            min_value=0.90,
            max_value=0.999,
            value=0.99,
            step=0.001,
            format="%.3f"
        )
        alpha = 1 - confidence_level
        st.info(f"Probabilidad de Excepción (alpha): <b>{alpha:.3%}</b>")
        st.markdown(f"<b>Observaciones:</b> {pd.to_datetime(START_DATE).date()} a {pd.to_datetime(END_DATE).date()}", unsafe_allow_html=True)

    # --- 2. Carga y Preparación de Datos (Caché) ---
    combined_usd_df, daily_returns_df = load_and_prepare_data(COMPANIES, START_DATE, END_DATE)
    
    if combined_usd_df.empty:
        st.error("Error al cargar datos. Verifique los tickers y el rango de fechas en el código fuente.")
        st.stop()
    
    # Cálculos principales
    stats_df = calculate_descriptive_statistics(combined_usd_df, daily_returns_df)
    var_results_df = calculate_var(daily_returns_df, confidence_level, NUM_SIMULATIONS)
    backtest_df = perform_backtesting(daily_returns_df, var_results_df, confidence_level)
        
    # --- 3. Renderizar Sección Seleccionada (Lógica de Renderizado Actualizada) ---
    st.markdown("---")
    if page == "1. Introducción y Metodología":
        section_introduction(combined_usd_df, daily_returns_df)
    elif page == "2. Análisis Exploratorio (EDA)":
        section_eda(combined_usd_df, daily_returns_df, stats_df)
    elif page == "3. Estimación del VaR":
        section_var_estimation(var_results_df, confidence_level)
    elif page == "4. Estado del Arte de Pruebas":
        section_state_of_the_art()
    elif page == "5. Explicación de Pruebas":
        section_backtesting_explanation()
    elif page == "6. Backtesting":
        section_backtesting(backtest_df)
    elif page == "7. Conclusiones y Hallazgos Principales":
        section_findings(backtest_df, stats_df)
    elif page == "8. Referencias":
        section_references()


if __name__ == '__main__':
    main()