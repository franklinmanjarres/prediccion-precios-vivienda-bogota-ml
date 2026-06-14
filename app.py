import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os

# ============================================================
# Configuración de la página
# ============================================================
st.set_page_config(
    page_title="Predicción de Precios — Bogotá",
    page_icon="🏠",
    layout="wide"
)

# ============================================================
# Título principal
# ============================================================
st.title("Predicción de Precios de Vivienda — Bogotá, Colombia")
st.markdown(
    "Modelo de regresión entrenado con datos del mercado inmobiliario de Bogotá. "
    "Ingresa las características del apartamento para estimar su precio."
)
st.divider()

# ============================================================
# Columnas OHE
# ============================================================
COLS_OHE_TIPO = [
    "tipo_de_inmueble_Apartamento",
    "tipo_de_inmueble_Casa",
    "tipo_de_inmueble_casa con conjunto cerrado"
]
COLS_OHE_PORTERIA = ["porteria_24 hrs", "porteria_No"]
COLS_OHE_BARRIO = [
    "barrio_BOSA OCCIDENTAL", "barrio_BRITALIA", "barrio_CALANDAIMA",
    "barrio_CASTILLA", "barrio_CIUDAD USME", "barrio_EL PORVENIR",
    "barrio_EL PRADO", "barrio_ENGATIVA", "barrio_GARCES NAVAS",
    "barrio_ISMAEL PERDOMO", "barrio_LA ALHAMBRA", "barrio_LA URIBE",
    "barrio_LOS CEDROS", "barrio_Otro", "barrio_SANTA BARBARA",
    "barrio_SUBA", "barrio_TIBABUYES", "barrio_TINTAL SUR", "barrio_USAQUEN"
]

ALL_COLS = (
    ["administración", "estrato", "antiguedad", "remodelado", "área",
     "habitaciones", "baños", "garajes", "elevadores", "deposito",
     "gas", "parqueadero"] +
    COLS_OHE_TIPO + COLS_OHE_PORTERIA + COLS_OHE_BARRIO
)

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Características del apartamento")

area           = st.sidebar.slider("Área (m²)", 20, 400, 80)
habitaciones   = st.sidebar.slider("Habitaciones", 1, 6, 3)
banos          = st.sidebar.slider("Baños", 1, 5, 2)
garajes        = st.sidebar.slider("Garajes", 0, 4, 1)
elevadores     = st.sidebar.slider("Elevadores", 0, 5, 1)
estrato        = st.sidebar.selectbox("Estrato", [1, 2, 3, 4, 5, 6], index=2)
administracion = st.sidebar.number_input("Administración (COP)", 0, 2_000_000, 300_000, step=50_000)
antiguedad     = st.sidebar.number_input("Antigüedad (años)", 0, 50, 10)

st.sidebar.markdown("---")
remodelado  = st.sidebar.selectbox("Remodelado", ["Si", "No"])
deposito    = st.sidebar.selectbox("Depósito", ["Si", "No"])
gas         = st.sidebar.selectbox("Gas", ["Si", "No"])
parqueadero = st.sidebar.selectbox("Parqueadero", ["Si", "No"])

st.sidebar.markdown("---")
tipo    = st.sidebar.selectbox("Tipo de inmueble", ["Apartamento", "Casa", "casa con conjunto cerrado"])
porteria = st.sidebar.selectbox("Portería", ["24 hrs", "No", "Otro"])
barrio  = st.sidebar.selectbox(
    "Barrio",
    ["BOSA OCCIDENTAL", "BRITALIA", "CALANDAIMA", "CASTILLA",
     "CIUDAD USME", "EL PORVENIR", "EL PRADO", "ENGATIVA",
     "GARCES NAVAS", "ISMAEL PERDOMO", "LA ALHAMBRA", "LA URIBE",
     "LOS CEDROS", "Otro", "SANTA BARBARA", "SUBA",
     "TIBABUYES", "TINTAL SUR", "USAQUEN"]
)

# ============================================================
# Construir features
# ============================================================
def construir_features():
    row = {col: 0 for col in ALL_COLS}
    row["administración"] = administracion
    row["estrato"]        = estrato
    row["antiguedad"]     = antiguedad
    row["remodelado"]     = 1 if remodelado == "Si" else 0
    row["área"]           = area
    row["habitaciones"]   = habitaciones
    row["baños"]          = banos
    row["garajes"]        = garajes
    row["elevadores"]     = elevadores
    row["deposito"]       = 1 if deposito == "Si" else 0
    row["gas"]            = 1 if gas == "Si" else 0
    row["parqueadero"]    = 1 if parqueadero == "Si" else 0
    col_tipo = f"tipo_de_inmueble_{tipo}"
    if col_tipo in row:
        row[col_tipo] = 1
    col_porteria = f"porteria_{porteria}"
    if col_porteria in row:
        row[col_porteria] = 1
    col_barrio = f"barrio_{barrio}"
    if col_barrio in row:
        row[col_barrio] = 1
    return pd.DataFrame([row])[ALL_COLS]

X_input = construir_features()
modelo_path = "modelo_huber.pkl"

col1, col2 = st.columns([1, 1])

# ============================================================
# Columna 1 — Precio + comparación de modelos
# ============================================================
with col1:
    st.subheader("Precio estimado")

    if os.path.exists(modelo_path):
        with open(modelo_path, "rb") as f:
            pipe_huber_std = pickle.load(f)
        log_pred    = pipe_huber_std.predict(X_input)[0]
        precio_pred = np.expm1(log_pred)
        st.metric(label="Precio predicho", value=f"${precio_pred / 1e6:,.1f}M COP")
        st.caption(f"Equivalente a ${precio_pred:,.0f} pesos colombianos")
    else:
        st.warning(
            "Modelo no encontrado. Guarda el modelo desde el notebook con:\n\n"
            "```python\nimport pickle\nwith open('modelo_huber.pkl', 'wb') as f:\n"
            "    pickle.dump(pipe_huber_std, f)\n```"
        )

    st.divider()
    st.subheader("Comparación de modelos")

    modelos = ["Baseline (mediana)", "Reg. Lineal (std)", "HuberRegressor (std)"]
    maes    = [105.2, 33.0, 34.2]
    colores = ["#c0c0c0", "#4C72B0", "#DD8452"]

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=modelos,
        y=maes,
        marker_color=colores,
        text=[f"${m}M" for m in maes],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>MAE: $%{y}M COP<extra></extra>"
    ))
    fig1.add_hline(
        y=105.2, line_dash="dash", line_color="red",
        annotation_text="Baseline", annotation_position="top right"
    )
    fig1.update_layout(
        title="MAE por modelo (Millones COP)",
        yaxis_title="MAE (M COP)",
        yaxis_range=[0, 130],
        plot_bgcolor="white",
        height=400
    )
    st.plotly_chart(fig1, use_container_width=True)

# ============================================================
# Columna 2 — Variables más influyentes
# ============================================================
with col2:
    st.subheader("Variables más influyentes")

    variables = [
        "área", "estrato", "garajes", "administración",
        "barrio_CIUDAD USME", "baños", "barrio_Otro",
        "barrio_SANTA BARBARA", "barrio_EL PRADO", "barrio_CASTILLA",
        "barrio_EL PORVENIR", "barrio_TINTAL SUR", "elevadores",
        "antiguedad", "barrio_LA URIBE"
    ]
    coeficientes = [
        0.1688, 0.1600, 0.0783, 0.0702,
        -0.0533, 0.0441, 0.0336,
        -0.0302, 0.0291, 0.0269,
        -0.0268, -0.0260, 0.0243,
        -0.0230, 0.0214
    ]

    colores_coef = ["#FFD700" if v > 0 else "#1F77B4" for v in coeficientes]
    direccion    = ["Sube el precio" if v > 0 else "Baja el precio" for v in coeficientes]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=coeficientes[::-1],
        y=variables[::-1],
        orientation="h",
        marker_color=colores_coef[::-1],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Coeficiente: %{x:.4f}<br>"
            "%{customdata}<extra></extra>"
        ),
        customdata=direccion[::-1]
    ))
    fig2.add_vline(x=0, line_color="black", line_width=1, line_dash="dash")
    fig2.update_layout(
        title="Top 15 variables — HuberRegressor",
        xaxis_title="Coeficiente (escala log-precio estandarizado)",
        plot_bgcolor="white",
        height=500,
        legend=dict(
            itemsizing="constant",
            title="Efecto sobre el precio"
        )
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# Footer
# ============================================================
st.divider()
st.markdown(
    "**Franklin Manuel Manjarres**  \n"
    "Cali, Colombia  \n"
    "[github.com/franklinmanjarres](https://github.com/franklinmanjarres/-Predicci-n-de-Precios-de-Vivienda-en-Bogot-Colombia)  \n"
    "[fw2z7ovawqtfw6gwnytndv.streamlit.app](https://fw2z7ovawqtfw6gwnytndv.streamlit.app)"
)
