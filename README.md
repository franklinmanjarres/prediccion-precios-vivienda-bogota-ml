# Predicción de Precios de Vivienda en Bogotá | Machine Learning

Modelo de regresión para estimar el precio de apartamentos en Bogotá a partir de características físicas, socioeconómicas y de ubicación. El proyecto cubre el pipeline completo de machine learning: limpieza de datos, análisis exploratorio, selección de variables, modelado con validación cruzada y análisis de residuos.

---

## Descripción del problema

El mercado inmobiliario en Colombia presenta una variable única a nivel mundial: el **estrato socioeconómico** (escala 1–6), que no solo refleja el nivel económico del sector sino también el acceso a servicios, seguridad y potencial de valorización. Este proyecto incorpora el estrato como predictor y analiza su impacto real sobre el precio de la vivienda en Bogotá.

**Pregunta central:** ¿Qué variables determinan el precio de un apartamento en Bogotá y con qué precisión puede estimarse?

---

## Dataset

| Característica | Detalle |
|---|---|
| Fuente | Mercado inmobiliario de Bogotá, Colombia |
| Observaciones | 585 apartamentos |
| Variables originales | 21 |
| Variables del modelo | 16 (tras limpieza) |
| Variable objetivo | Precio en pesos colombianos (COP) |

Variables principales: `área`, `estrato`, `habitaciones`, `baños`, `garajes`, `administración`, `antigüedad`, `barrio`, entre otras.

---

## Pipeline del proyecto

```
Carga de datos
     ↓
Limpieza y preprocesamiento
     ↓
Análisis exploratorio (EDA)
     ↓
Selección de variables (Árbol de Decisión)
     ↓
Ingeniería de características
     ↓
Codificación y partición 75/25
     ↓
Modelado con validación cruzada (KFold, 10 splits)
     ↓
Análisis de residuos
     ↓
Interpretación de coeficientes
```

---

## Resultados

| Modelo | MAE (M COP) | RMSE (M COP) |
|---|---|---|
| Baseline (mediana) | 105.2 | 140.5 |
| Regresión Lineal | 33.0 | 58.1 |
| HuberRegressor sin estandarizar | diverge | diverge |
| Regresión Lineal estandarizada | 33.0 | 58.1 |
| **HuberRegressor estandarizado** | **34.2** | **62.7** |

El modelo final reduce el error en un **67.5%** respecto al baseline. El error promedio de **$34.2M COP** sobre un precio mediano de **$218M COP** representa una precisión del 84%.

---

## Variables más influyentes

| Variable | Coeficiente | Interpretación |
|---|---|---|
| `área` | +0.1688 | A mayor área, mayor precio |
| `estrato` | +0.1600 | Variable exclusiva del sistema colombiano |
| `garajes` | +0.0783 | Asociado a conjuntos de mayor categoría |
| `administración` | +0.0702 | Refleja tamaño y amenidades del conjunto |
| `antigüedad` | -0.0230 | A mayor antigüedad, menor precio |
| `barrio_CIUDAD USME` | -0.0533 | Zona de menor valor en Bogotá |

**`área` y `estrato` dominan la predicción** — el estrato, exclusivo del contexto colombiano, es el segundo predictor más importante, superando incluso al número de habitaciones y baños.

---

## Decisiones metodológicas

**Transformación logarítmica del target**
El precio presenta distribución asimétrica. Aplicar `log1p` antes del modelado y revertir con `expm1` al predecir mejora el ajuste del modelo.

**Árbol de Decisión como herramienta de selección**
No es el modelo final — se usa con `max_depth=4` para identificar las variables numéricas más discriminantes respecto al precio antes del modelado de regresión.

**HuberRegressor como modelo final**
Se elige sobre la Regresión Lineal por su robustez ante valores atípicos, característica importante en datos inmobiliarios donde existen propiedades con precios genuinamente extremos.

**Estandarización obligatoria**
Sin `StandardScaler`, HuberRegressor presenta divergencia numérica (MAE > $400 billones COP). Este resultado queda documentado como lección metodológica.

---

## Limitaciones y trabajo futuro

- Incorporar más propiedades de estrato 5–6 para mejorar el desempeño 
  en el segmento premium.
- Agregar variables espaciales: distancia a estaciones de TransMilenio, 
  parques e instituciones educativas.
---

## Stack tecnológico

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

---

## Autor
st.divider()
st.markdown(
    "**Franklin Manuel Manjarres**  \n"
    "Cali, Colombia  \n"
    "[github.com/franklinmanjarres](https://github.com/franklinmanjarres/-Predicci-n-de-Precios-de-Vivienda-en-Bogot-Colombia)  \n"
    "[fw2z7ovawqtfw6gwnytndv.streamlit.app](https://fw2z7ovawqtfw6gwnytndv.streamlit.app)"
)
