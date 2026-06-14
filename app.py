import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# Columnas de variables OHE
# ============================================================
COLS_OHE_TIPO = [
    "tipo_de_inmueble_Apartamento",
    "tipo_de_inmueble_Casa",
    "tipo_de_inmueble_casa con conjunto cerrado"
]
COLS_OHE_PORTERIA = [
    "porteria_24 hrs",
    "porteria_No"
]
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
# Sidebar — inputs del usuario
# ============================================================
st.sidebar.header("Características del apartamento")

area         = st.sidebar.slider("Área (m²)", 20, 400, 80)
habitaciones = st.sidebar.slider("Habitaciones", 1, 6, 3)
banos        = st.sidebar.slider("Baños", 1, 5, 2)
garajes      = st.sidebar.slider("Garajes", 0, 4, 1)
elevadores   = st.sidebar.slider("Elevadores", 0, 5, 1)
estrato      = st.sidebar.selectbox("Estrato", [1, 2, 3, 4, 5, 6], index=2)
administracion = st.sidebar.number_input("Administración (COP)", 0, 2_000_000, 300_000, step=50_000)
antiguedad   = st.sidebar.number_input("Antigüedad (años)", 0, 50, 10)

st.sidebar.markdown("---")
remodelado   = st.sidebar.selectbox("Remodelado", ["Si", "No"])
deposito     = st.sidebar.selectbox("Depósito", ["Si", "No"])
gas          = st.sidebar.selectbox("Gas", ["Si", "No"])
parqueadero  = st.sidebar.selectbox("Parqueadero", ["Si", "No"])

st.sidebar.markdown("---")
tipo         = st.sidebar.selectbox(
    "Tipo de inmueble",
    ["Apartamento", "Casa", "casa con conjunto cerrado"]
)
porteria     = st.sidebar.selectbox("Portería", ["24 hrs", "No", "Otro"])
barrio       = st.sidebar.selectbox(
    "Barrio",
    ["BOSA OCCIDENTAL", "BRITALIA", "CALANDAIMA", "CASTILLA",
     "CIUDAD USME", "EL PORVENIR", "EL PRADO", "ENGATIVA",
     "GARCES NAVAS", "ISMAEL PERDOMO", "LA ALHAMBRA", "LA URIBE",
     "LOS CEDROS", "Otro", "SANTA BARBARA", "SUBA",
     "TIBABUYES", "TINTAL SUR", "USAQUEN"]
)

# ============================================================
# Construir vector de features
# ============================================================
def construir_features():
    row = {col: 0 for col in ALL_COLS}

    # Numéricas
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

    # OHE tipo
    col_tipo = f"tipo_de_inmueble_{tipo}"
    if col_tipo in row:
        row[col_tipo] = 1

    # OHE portería
    col_porteria = f"porteria_{porteria}"
    if col_porteria in row:
        row[col_porteria] = 1

    # OHE barrio
    col_barrio = f"barrio_{barrio}"
    if col_barrio in row:
        row[col_barrio] = 1

    return pd.DataFrame([row])[ALL_COLS]

# ============================================================
# Predicción
# ============================================================
X_input = construir_features()

# Intentar cargar modelo guardado
modelo_path = "modelo_huber.pkl"

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Precio estimado")

    if os.path.exists(modelo_path):
        with open(modelo_path, "rb") as f:
            pipe_huber_std = pickle.load(f)
        log_pred = pipe_huber_std.predict(X_input)[0]
        precio_pred = np.expm1(log_pred)
        st.metric(
            label="Precio predicho",
            value=f"${precio_pred / 1e6:,.1f}M COP"
        )
        st.caption(f"Equivalente a ${precio_pred:,.0f} pesos colombianos")
    else:
        st.warning(
            "Modelo no encontrado. Guarda el modelo desde el notebook con:\n\n"
            "```python\nimport pickle\nwith open('modelo_huber.pkl', 'wb') as f:\n"
            "    pickle.dump(pipe_huber_std, f)\n```\n\n"
            "Luego coloca el archivo `modelo_huber.pkl` en la misma carpeta que `app.py`."
        )

    st.divider()

    # Comparación de modelos
    st.subheader("Comparación de modelos")
    modelos  = ["Baseline\n(mediana)", "Reg. Lineal\n(std)", "HuberRegressor\n(std)"]
    maes     = [105.2, 33.0, 34.2]
    colores  = ["#c0c0c0", "#4C72B0", "#DD8452"]

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    bars = ax1.bar(modelos, maes, color=colores, edgecolor="white", width=0.5)
    for bar, mae in zip(bars, maes):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"${mae}M", ha="center", fontsize=10, fontweight="bold"
        )
    ax1.axhline(105.2, color="red", linestyle="--", linewidth=1, label="Baseline")
    ax1.set_title("MAE por modelo (Millones COP)", fontsize=11)
    ax1.set_ylabel("MAE (M COP)")
    ax1.set_ylim(0, 130)
    ax1.legend()
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    st.subheader("Variables más influyentes")

    variables  = [
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

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.barh(variables[::-1], coeficientes[::-1],
             color=colores_coef[::-1], edgecolor="white")
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("Top 15 variables — HuberRegressor", fontsize=11)
    ax2.set_xlabel("Coeficiente (escala log-precio estandarizado)")

    leyenda = [
        mpatches.Patch(color="#FFD700", label="Sube el precio"),
        mpatches.Patch(color="#1F77B4", label="Baja el precio")
    ]
    ax2.legend(handles=leyenda, loc="lower right")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ============================================================
# Footer
# ============================================================
st.divider()
st.markdown(
    "**Autor:** Franklin Manuel Manjarres · "
    "st.markdown(
    "**Repositorio:** [github.com/franklinmanjarres](https://github.com/franklinmanjarres/-Predicci-n-de-Precios-de-Vivienda-en-Bogot-Colombia)"
)"
)
