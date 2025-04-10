# Código para el widget interactivo (usar componente iframe en Gamma.app)
import streamlit as st
import pandas as pd
import numpy as np

# Datos simulados basados en estudios reales (NOAA & GEOMAR)
pressure = st.slider("Presión (atm)", 100, 500, 300)
current_speed = st.slider("Velocidad corriente (m/s)", 0.1, 5.0, 1.0)
mineral_type = st.selectbox("Mineral", ["Nódulo Mn-Co", "Sulfuro Polimetálico"])

# Simular predicción de IA + Active Learning
def predict_erosion(p, v, mineral):
    base_rate = 0.05 if mineral == "Nódulo Mn-Co" else 0.12
    uncertainty = np.random.uniform(0.01, 0.15)
    use_dft = uncertainty > 0.1
    erosion_rate = base_rate * (p / 300) * (v / 1.0) + np.random.normal(0, 0.01)
    return erosion_rate, uncertainty, use_dft

erosion, uncert, use_dft = predict_erosion(pressure, current_speed, mineral_type)

# Mostrar resultados
col1, col2 = st.columns(2)
col1.metric("Tasa de erosión (Å/ps)", f"{erosion:.2f}", help="Valor predicho por la IA")
col2.metric("Incertidumbre", f"{uncert:.2f}", 
            "✅ DFT no requerido" if not use_dft else "⚠️ Ejecutando DFT...")

# Gráfico de clusters (simulado)
st.plotly_chart(cluster_plot(mineral_type))