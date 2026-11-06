import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predicción - Depósito a Plazo Bancario", page_icon="🏦")

st.title("🏦 Predicción de suscripción a depósito a plazo")
st.write(
    "Esta app usa un modelo de Machine Learning (Árbol de Decisión con criterio "
    "de entropía) entrenado sobre datos reales de campañas de marketing bancario, "
    "para predecir si un cliente suscribirá o no un depósito a plazo fijo."
)


@st.cache_resource
def load_model():
    return joblib.load("modelo_banco.pkl")


model = load_model()

NUMERIC_FEATURES = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CATEGORICAL_OPTIONS = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown"],
    "marital": ["married", "single", "divorced"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["failure", "other", "success", "unknown"],
}

st.header("📋 Datos del cliente")

col1, col2 = st.columns(2)
inputs = {}

with col1:
    inputs["age"] = st.number_input("Edad", min_value=18, max_value=100, value=40)
    inputs["balance"] = st.number_input("Saldo en cuenta (balance)", value=500)
    inputs["day"] = st.number_input("Día del mes del último contacto", min_value=1, max_value=31, value=15)
    inputs["duration"] = st.number_input("Duración del último llamado (segundos)", min_value=0, value=180)
    inputs["campaign"] = st.number_input("Cantidad de contactos en esta campaña", min_value=1, value=2)
    inputs["pdays"] = st.number_input("Días desde el último contacto previo (-1 = nunca contactado)", value=-1)
    inputs["previous"] = st.number_input("Cantidad de contactos previos a esta campaña", min_value=0, value=0)

with col2:
    inputs["job"] = st.selectbox("Trabajo", CATEGORICAL_OPTIONS["job"])
    inputs["marital"] = st.selectbox("Estado civil", CATEGORICAL_OPTIONS["marital"])
    inputs["education"] = st.selectbox("Educación", CATEGORICAL_OPTIONS["education"])
    inputs["default"] = st.selectbox("¿Tiene crédito en default?", CATEGORICAL_OPTIONS["default"])
    inputs["housing"] = st.selectbox("¿Tiene préstamo hipotecario?", CATEGORICAL_OPTIONS["housing"])
    inputs["loan"] = st.selectbox("¿Tiene préstamo personal?", CATEGORICAL_OPTIONS["loan"])
    inputs["contact"] = st.selectbox("Tipo de contacto", CATEGORICAL_OPTIONS["contact"])
    inputs["month"] = st.selectbox("Mes del último contacto", CATEGORICAL_OPTIONS["month"])
    inputs["poutcome"] = st.selectbox("Resultado de la campaña anterior", CATEGORICAL_OPTIONS["poutcome"])

st.header("📂 (Opcional) Cargar varios clientes desde un CSV")
st.caption(
    "El CSV debe tener las columnas: "
    + ", ".join(NUMERIC_FEATURES + list(CATEGORICAL_OPTIONS.keys()))
)
uploaded_file = st.file_uploader("Subí un archivo CSV", type=["csv"])

if st.button("🔍 Predecir cliente individual"):
    X_new = pd.DataFrame([inputs])
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0][1]

    if pred == 1:
        st.success(f"✅ El cliente probablemente **SÍ** suscribirá el depósito (probabilidad: {proba:.2%})")
    else:
        st.error(f"❌ El cliente probablemente **NO** suscribirá el depósito (probabilidad de sí: {proba:.2%})")

if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)
    columnas_necesarias = NUMERIC_FEATURES + list(CATEGORICAL_OPTIONS.keys())
    faltantes = set(columnas_necesarias) - set(df_new.columns)
    if faltantes:
        st.error(f"Faltan columnas en el CSV: {faltantes}")
    else:
        X_batch = df_new[columnas_necesarias]
        preds = model.predict(X_batch)
        probas = model.predict_proba(X_batch)[:, 1]
        df_resultado = df_new.copy()
        df_resultado["prediccion"] = preds
        df_resultado["probabilidad_si"] = probas
        st.subheader("Resultados")
        st.dataframe(df_resultado)
        st.download_button(
            "⬇️ Descargar resultados en CSV",
            df_resultado.to_csv(index=False).encode("utf-8"),
            "resultados_prediccion.csv",
            "text/csv",
        )

st.divider()
st.caption("TP Explotación de Datos y Descubrimiento de Conocimiento — Modelo: Árbol de Decisión (entropía)")
