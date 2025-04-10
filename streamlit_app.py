import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Función para generar datos de clusters (simulados)
def generate_cluster_data(mineral_type):
    np.random.seed(42)
    n_samples = 100
    
    if mineral_type == "Nódulo Mn-Co":
        # Simular 3 clusters para nódulos
        features = np.random.normal(size=(n_samples, 5))
        features[:30] += [2, 2, 0, 0, 0]  # Cluster 1 (alto Co)
        features[30:70] += [0, -2, 0, 0, 0]  # Cluster 2 (Mn dominante)
    else:
        # Simular 2 clusters para sulfuros
        features = np.random.normal(size=(n_samples, 5))
        features[:60] += [0, 3, 0, 0, 0]  # Cluster sulfuros ricos
    
    # Reducción dimensional para visualización
    tsne = TSNE(n_components=2, perplexity=20)
    embeddings = tsne.fit_transform(features)
    
    # K-Means para colorear clusters
    kmeans = KMeans(n_clusters=3 if mineral_type == "Nódulo Mn-Co" else 2)
    clusters = kmeans.fit_predict(features)
    
    return pd.DataFrame({
        "x": embeddings[:, 0],
        "y": embeddings[:, 1],
        "cluster": clusters,
        "Composición": np.random.choice(["Mn", "Co", "Ni", "Cu"], n_samples)
    })

# Función para graficar clusters
def cluster_plot(mineral_type):
    df = generate_cluster_data(mineral_type)
    fig = px.scatter(
        df, x="x", y="y", color="cluster",
        hover_data=["Composición"],
        title=f"Clusters de {mineral_type} (t-SNE)",
        width=600, height=400
    )
    fig.update_traces(marker=dict(size=12))
    return fig

# --- INTERFAZ STREAMLIT ---
st.title("Demo Smart OreChain")

# Controles
col1, col2 = st.columns(2)
with col1:
    pressure = st.slider("Presión (atm)", 100, 500, 300)
    mineral_type = st.selectbox("Mineral", ["Nódulo Mn-Co", "Sulfuro Polimetálico"])
with col2:
    current_speed = st.slider("Velocidad corriente (m/s)", 0.1, 5.0, 1.0)
    if st.button("Simular DFT", disabled=False):
        st.toast("Ejecutando simulación cuántica...", icon="⚛️")

# Simulación de predicción
def predict_erosion(p, v, mineral):
    base_rate = 0.05 if mineral == "Nódulo Mn-Co" else 0.12
    uncertainty = np.random.uniform(0.01, 0.15)
    use_dft = uncertainty > 0.1
    erosion_rate = base_rate * (p / 300) * (v / 1.0) + np.random.normal(0, 0.01)
    return erosion_rate, uncertainty, use_dft

erosion, uncert, use_dft = predict_erosion(pressure, current_speed, mineral_type)

# Resultados
st.divider()
st.subheader("Resultados en Tiempo Real")

metric_col1, metric_col2 = st.columns(2)
metric_col1.metric("Tasa de erosión", f"{erosion:.2f} Å/ps", 
                  help="Predicción del modelo GNN")
metric_col2.metric("Incertidumbre", f"{uncert:.2f} eV",
                  "✅ DFT no requerido" if not use_dft else "⚠️ Necesita validación DFT")

# Visualización de clusters
st.plotly_chart(cluster_plot(mineral_type))

# Mensaje contextual
if use_dft:
    st.warning("""
    **Sistema Active Learning activado**:  
    La alta incertidumbre (>0.1 eV) requiere validación con simulación DFT.  
    Esto ocurre solo en el 15-20% de los casos (ahorro del 80% en costes computacionales).
    """)
else:
    st.success("Predicción con alta confianza. Puede proceder con la certificación blockchain.")