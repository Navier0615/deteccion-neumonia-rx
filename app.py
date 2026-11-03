"""
Detector de Neumonía en Rayos X
Aplicación Streamlit que clasifica radiografías de tórax en NORMAL o PNEUMONIA
usando una CNN entrenada desde cero (etapa de Deployment de CRISP-DM).
"""

import os

# Silenciar los mensajes informativos de TensorFlow (debe ir antes de importarlo)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_cnn.keras")

IMAGE_SIZE = (150, 150)
DECISION_THRESHOLD = 0.5
# NEAREST es el valor por defecto de ImageDataGenerator.flow_from_dataframe,
# usado en el entrenamiento; se mantiene para que el preprocesamiento sea idéntico
RESAMPLING_METHOD = Image.Resampling.NEAREST
ALLOWED_EXTENSIONS = ["jpeg", "jpg", "png"]

# El ícono de advertencia del disclaimer se dibuja como SVG en el footer
DISCLAIMER_TEXT = (
    "Esta herramienta es con fines educativos. No reemplaza el diagnóstico "
    "de un profesional médico. Siempre consulte a un radiólogo o neumólogo certificado."
)
COPYRIGHT_TEXT = "© 2026 · Proyecto académico · Institución Universitaria Colegio Mayor del Cauca"

# Paleta "Clinical Dark"
COLOR_BG = "#0F172A"
COLOR_SURFACE = "#1E293B"
COLOR_BORDER = "#334155"
COLOR_ACCENT = "#06B6D4"
COLOR_TEXT = "#E2E8F0"
COLOR_TEXT_STRONG = "#F8FAFC"
COLOR_MUTED = "#94A3B8"
COLOR_SUBTLE = "#64748B"
COLOR_FAINT = "#475569"
COLOR_NORMAL = "#10B981"
COLOR_PNEUMONIA = "#DC2626"
COLOR_PNEUMONIA_ICON = "#F87171"

st.set_page_config(
    page_title="Detector de Neumonía",
    page_icon=":material/pulmonology:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Íconos SVG de trazo (estilo Lucide). Usan currentColor para tomar el color
# del contenedor; no se usan emojis porque no se ven igual en todos los sistemas.
# ---------------------------------------------------------------------------

ICON_LUNGS = (
    '<path d="M12 3.5v7"/>'
    '<path d="M12 10.5 9.5 13"/><path d="M12 10.5l2.5 2.5"/>'
    '<path d="M9.5 7.5C6.5 7.5 4 11.5 4 16c0 2.2 1 4 3 4 1.8 0 2.5-1.2 2.5-3V7.5z"/>'
    '<path d="M14.5 7.5c3 0 5.5 4 5.5 8.5 0 2.2-1 4-3 4-1.8 0-2.5-1.2-2.5-3V7.5z"/>'
)
ICON_UPLOAD = (
    '<path d="M4 14.9A7 7 0 1 1 15.7 8h1.8a4.5 4.5 0 0 1 2.5 8.2"/>'
    '<path d="M12 12v9"/><path d="m16 16-4-4-4 4"/>'
)
ICON_WAITING = (
    '<circle cx="12" cy="12" r="10"/>'
    '<line x1="8" y1="12" x2="8.01" y2="12"/><line x1="12" y1="12" x2="12.01" y2="12"/>'
    '<line x1="16" y1="12" x2="16.01" y2="12"/>'
)
ICON_CHECK = '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>'
ICON_ALERT = (
    '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>'
    '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
)


def svg_icon(paths: str, size: int, color: str = "currentColor", stroke_width: float = 2) -> str:
    """Construye un ícono SVG de trazo a partir de sus elementos internos."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="flex-shrink:0; display:block;">{paths}</svg>'
    )


# ---------------------------------------------------------------------------
# Estilos globales (CSS encapsulado en un solo bloque)
# ---------------------------------------------------------------------------

CUSTOM_CSS = f"""
<style>
/* --- Tipografía: fuentes del sistema con jerarquía editorial --- */
.stApp,
[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"] {{
    font-family: 'Inter', 'SF Pro Display', 'Segoe UI', system-ui, sans-serif;
}}
.ca-eyebrow {{
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {COLOR_ACCENT};
}}

/* --- Contenedor principal --- */
[data-testid="stMainBlockContainer"] {{
    padding-top: 2.2rem;
    max-width: 1240px;
}}

/* --- Sidebar: separadores, títulos de sección y stats cards --- */
.ca-sb-divider {{
    border: none;
    border-top: 1px solid {COLOR_BORDER};
    margin: 1.4rem 0 1.1rem;
}}
.ca-sb-title {{
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {COLOR_ACCENT};
    margin-bottom: 0.75rem;
}}
.ca-sb-text {{
    font-size: 0.88rem;
    line-height: 1.55;
    color: {COLOR_MUTED};
}}
.ca-row {{
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    padding: 0.42rem 0;
    border-bottom: 1px solid rgba(51, 65, 85, 0.6);
    font-size: 0.85rem;
}}
.ca-row:last-child {{ border-bottom: none; }}
.ca-row-label {{ color: {COLOR_MUTED}; }}
.ca-row-value {{ color: {COLOR_TEXT}; font-weight: 600; text-align: right; }}
.ca-stats {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
}}
.ca-stat {{
    background: {COLOR_BG};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 0.85rem 0.8rem 0.75rem;
}}
.ca-stat-value {{
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    color: {COLOR_TEXT_STRONG};
    font-variant-numeric: tabular-nums;
}}
.ca-stat-value small {{
    font-size: 0.55em;
    font-weight: 600;
    color: {COLOR_MUTED};
    margin-left: 1px;
}}
.ca-stat-label {{
    margin-top: 0.45rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {COLOR_MUTED};
}}
.ca-stat-delta {{
    margin-top: 0.3rem;
    font-size: 0.68rem;
    font-weight: 600;
    color: {COLOR_NORMAL};
}}
.ca-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
}}
.ca-table th {{
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.65rem;
    color: {COLOR_MUTED};
    padding: 0.35rem 0.3rem;
    border-bottom: 1px solid {COLOR_BORDER};
}}
.ca-table td {{
    padding: 0.4rem 0.3rem;
    color: {COLOR_TEXT};
    border-bottom: 1px solid rgba(51, 65, 85, 0.6);
}}
.ca-table tr.ca-highlight td {{
    color: {COLOR_ACCENT};
    font-weight: 700;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] details {{
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    background: {COLOR_BG};
}}
.ca-powered {{
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    color: {COLOR_SUBTLE};
}}

/* --- Títulos de sección del área principal --- */
.ca-section {{
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {COLOR_MUTED};
    padding-bottom: 0.6rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid {COLOR_BORDER};
}}

/* --- Zona de carga: borde sólido que se ilumina en cyan al pasar el mouse --- */
.st-key-upload_zone {{
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.95) 0%, rgba(30, 41, 59, 0.55) 100%);
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px -18px rgba(0, 0, 0, 0.8);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}}
.st-key-upload_zone:hover {{
    border-color: {COLOR_ACCENT};
    box-shadow: 0 0 0 1px rgba(6, 182, 212, 0.25), 0 10px 30px -18px rgba(0, 0, 0, 0.8);
}}
.st-key-upload_zone [data-testid="stFileUploaderDropzone"] {{
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
}}

/* --- Imagen cargada con marco fino --- */
[data-testid="stImage"] img {{
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
}}

/* --- Indicador de espera con pulso suave --- */
@keyframes ca-pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.35; }}
}}
.ca-waiting-icon {{ animation: ca-pulse 2.2s ease-in-out infinite; }}
@media (prefers-reduced-motion: reduce) {{
    .ca-waiting-icon {{ animation: none; }}
}}
</style>
"""


def inject_styles():
    """Aplica el CSS global de la aplicación."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_html(html: str):
    """
    Renderiza un bloque HTML. Se compacta en una sola línea para que Markdown
    no interprete la indentación como bloque de código.
    """
    compact_html = " ".join(line.strip() for line in html.strip().splitlines())
    st.markdown(compact_html, unsafe_allow_html=True)


def section_title(text: str):
    """Título de sección del área principal."""
    render_html(f'<div class="ca-section">{text}</div>')


# ---------------------------------------------------------------------------
# Modelo y preprocesamiento
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando el modelo...")
def load_model():
    """Carga el modelo CNN una sola vez y lo mantiene en caché."""
    return keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convierte la imagen al formato de entrada del modelo:
    escala de grises, 150x150, valores en [0, 1] y shape (1, 150, 150, 1).
    """
    grayscale_image = image.convert("L")
    resized_image = grayscale_image.resize(IMAGE_SIZE, RESAMPLING_METHOD)
    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    # Agregar dimensión de canal y de batch
    return image_array.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)


def predict_pneumonia(model, image_array: np.ndarray) -> float:
    """Devuelve la probabilidad de PNEUMONIA (salida sigmoide del modelo)."""
    prediction = model.predict(image_array, verbose=0)
    return float(prediction[0][0])


# ---------------------------------------------------------------------------
# Componentes de la interfaz
# ---------------------------------------------------------------------------

def render_header():
    """Encabezado principal: ícono de pulmones, título en mayúsculas y subtítulo."""
    render_html(f"""
        <div style="position:relative; overflow:hidden; border:1px solid {COLOR_BORDER};
                    border-radius:8px; padding:2.2rem 2.4rem; margin-bottom:1.75rem;
                    background:
                        linear-gradient(rgba(148, 163, 184, 0.045) 1px, transparent 1px) 0 0 / 28px 28px,
                        linear-gradient(90deg, rgba(148, 163, 184, 0.045) 1px, transparent 1px) 0 0 / 28px 28px,
                        linear-gradient(135deg, {COLOR_BG} 0%, {COLOR_SURFACE} 100%);
                    box-shadow: 0 18px 40px -24px rgba(0, 0, 0, 0.9);">
            <div style="display:flex; align-items:center; gap:1.6rem; flex-wrap:wrap;">
                <div style="color:{COLOR_ACCENT}; filter: drop-shadow(0 0 14px rgba(6, 182, 212, 0.35));">
                    {svg_icon(ICON_LUNGS, 66, stroke_width=1.6)}
                </div>
                <div style="flex:1; min-width:240px;">
                    <div class="ca-eyebrow">Apoyo al diagnóstico · Radiografía de tórax</div>
                    <div style="margin-top:0.4rem; font-size:clamp(1.7rem, 4vw, 2.5rem); font-weight:700;
                                letter-spacing:0.08em; line-height:1.1; color:{COLOR_TEXT_STRONG};">
                        DETECTOR DE NEUMONÍA
                    </div>
                    <div style="width:60px; border-bottom:2px solid {COLOR_ACCENT}; margin:0.95rem 0 0.9rem;"></div>
                    <div style="font-size:1.05rem; font-weight:400; line-height:1.55;
                                color:{COLOR_MUTED}; max-width:760px;">
                        Clasificación automática de radiografías de tórax en NORMAL o PNEUMONIA
                        mediante una red neuronal convolucional, con la probabilidad estimada para cada clase.
                    </div>
                </div>
            </div>
        </div>
    """)


def stat_card(value: str, label: str, delta: str = "") -> str:
    """Stats card de la sidebar; el signo % se muestra más pequeño."""
    number, percent = (value[:-1], "%") if value.endswith("%") else (value, "")
    percent_html = f"<small>{percent}</small>" if percent else ""
    delta_html = f'<div class="ca-stat-delta">{delta}</div>' if delta else ""
    return f"""
        <div class="ca-stat">
            <div class="ca-stat-value">{number}{percent_html}</div>
            <div class="ca-stat-label">{label}</div>
            {delta_html}
        </div>
    """


def data_rows(rows: list) -> str:
    """Lista de pares etiqueta / valor con separadores finos."""
    return "".join(
        f'<div class="ca-row"><span class="ca-row-label">{label}</span>'
        f'<span class="ca-row-value">{value}</span></div>'
        for label, value in rows
    )


def render_sidebar():
    """Muestra la información del proyecto, dataset, modelo y métricas."""
    divider = '<hr class="ca-sb-divider">'
    with st.sidebar:
        render_html(f"""
            <div style="display:flex; align-items:center; gap:0.75rem;">
                <div style="color:{COLOR_ACCENT};">{svg_icon(ICON_LUNGS, 30, stroke_width=1.8)}</div>
                <div>
                    <div style="font-size:0.95rem; font-weight:700; letter-spacing:0.1em; color:{COLOR_TEXT_STRONG};">
                        NEUMONÍA · CNN
                    </div>
                    <div style="font-size:0.72rem; letter-spacing:0.08em; color:{COLOR_SUBTLE};">
                        CLASIFICADOR DE RAYOS X
                    </div>
                </div>
            </div>
            {divider}
            <div class="ca-sb-title">Proyecto</div>
            <div class="ca-sb-text">
                Etapa de despliegue de un proyecto desarrollado con la metodología CRISP-DM.
                Se evaluaron tres arquitecturas (ANN, CNN desde cero y Transfer Learning)
                y se seleccionó la de mayor sensibilidad clínica.
            </div>
            {divider}
            <div class="ca-sb-title">Dataset</div>
            {data_rows([
                ("Nombre", "Chest X-Ray Pneumonia"),
                ("Fuente", f'<a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia" '
                           f'target="_blank" style="color:{COLOR_ACCENT}; text-decoration:none;">Kaggle</a>'),
                ("Imágenes", "~5,840"),
                ("NORMAL", "26%"),
                ("PNEUMONIA", "74%"),
            ])}
            {divider}
            <div class="ca-sb-title">Modelo</div>
            {data_rows([
                ("Arquitectura", "CNN desde cero"),
                ("Bloques", "4 × Conv2D + MaxPool"),
                ("Clasificador", "Dense 128 + Dropout"),
                ("Entrada", "150 × 150 · 1 canal"),
                ("Salida", "Sigmoide · umbral 0.5"),
            ])}
            {divider}
            <div class="ca-sb-title">Rendimiento en test</div>
            <div class="ca-stats">
                {stat_card("88.94%", "Accuracy", "+9.77 pp vs ANN")}
                {stat_card("98.72%", "Recall", "+11.54 pp vs ANN")}
                {stat_card("85.75%", "Precision")}
                {stat_card("91.78%", "F1-Score")}
            </div>
            <div style="margin-top:0.8rem;">
                {data_rows([
                    ("Falsos negativos", "5 / 390"),
                    ("Falsos positivos", "64 / 234"),
                ])}
            </div>
            <div style="margin-top:0.5rem; font-size:0.72rem; color:{COLOR_SUBTLE};">
                624 imágenes de prueba. Precision, Recall y F1 sobre la clase PNEUMONIA.
            </div>
        """)

        with st.expander("Comparación de modelos"):
            render_html(f"""
                <table class="ca-table">
                    <tr><th>Modelo</th><th>Acc.</th><th>Recall</th><th>FN</th></tr>
                    <tr><td>ANN Base</td><td>79.17%</td><td>87.18%</td><td>50</td></tr>
                    <tr class="ca-highlight"><td>CNN</td><td>88.94%</td><td>98.72%</td><td>5</td></tr>
                    <tr><td>MobileNetV2</td><td>86.22%</td><td>98.21%</td><td>7</td></tr>
                </table>
            """)

        render_html(f"""
            {divider}
            <div class="ca-powered">Powered by TensorFlow + Streamlit</div>
        """)


def render_upload_zone():
    """Zona de carga con borde sólido; devuelve el archivo subido."""
    with st.container(key="upload_zone"):
        render_html(f"""
            <div style="display:flex; flex-direction:column; align-items:center; text-align:center;
                        padding:0.4rem 0 0.8rem;">
                <div style="color:{COLOR_ACCENT};">{svg_icon(ICON_UPLOAD, 46, stroke_width=1.6)}</div>
                <div style="margin-top:0.8rem; font-size:1.05rem; font-weight:700; letter-spacing:0.16em;
                            color:{COLOR_TEXT_STRONG};">
                    ARRASTRA UNA RADIOGRAFÍA
                </div>
                <div style="margin-top:0.35rem; font-size:0.9rem; color:{COLOR_MUTED};">
                    O haz clic para seleccionar · JPEG, JPG, PNG
                </div>
            </div>
        """)
        return st.file_uploader(
            "Cargar radiografía de tórax",
            type=ALLOWED_EXTENSIONS,
            label_visibility="collapsed",
        )


def render_waiting_state():
    """Indicador discreto mientras no se ha subido ninguna imagen."""
    render_html(f"""
        <div style="display:flex; align-items:center; justify-content:center; gap:0.6rem;
                    padding:0.9rem 0 0.4rem;">
            <span class="ca-waiting-icon" style="color:{COLOR_ACCENT};">{svg_icon(ICON_WAITING, 18)}</span>
            <span style="font-size:0.9rem; letter-spacing:0.04em; color:{COLOR_MUTED};">
                Esperando radiografía para análisis
            </span>
        </div>
    """)


def probability_row(label: str, value: float, color: str) -> str:
    """Fila de probabilidad con barra del color de la clase."""
    return f"""
        <div style="margin-top:0.9rem;">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;
                        letter-spacing:0.1em; margin-bottom:0.4rem;">
                <span style="color:{COLOR_MUTED}; font-weight:600;">{label}</span>
                <span style="color:{COLOR_TEXT}; font-weight:700; font-variant-numeric:tabular-nums;">{value:.2%}</span>
            </div>
            <div style="background:{COLOR_BG}; border:1px solid {COLOR_BORDER}; border-radius:4px;
                        height:8px; overflow:hidden;">
                <div style="width:{value * 100:.2f}%; height:100%; background:{color};"></div>
            </div>
        </div>
    """


def render_result(pneumonia_probability: float):
    """Tarjeta de diagnóstico: verde esmeralda para NORMAL, rojo profundo para PNEUMONIA."""
    is_pneumonia = pneumonia_probability > DECISION_THRESHOLD
    normal_probability = 1 - pneumonia_probability

    if is_pneumonia:
        label = "PNEUMONIA"
        message = "Se identifican patrones radiológicos compatibles con neumonía."
        icon = svg_icon(ICON_ALERT, 56, COLOR_PNEUMONIA_ICON, stroke_width=1.8)
        accent = COLOR_PNEUMONIA
        gradient = "linear-gradient(135deg, #7F1D1D 0%, #450A0A 100%)"
        message_color = "#FECACA"
        track_color = "rgba(69, 10, 10, 0.8)"
        confidence = pneumonia_probability
    else:
        label = "NORMAL"
        message = "No se identifican patrones radiológicos compatibles con neumonía."
        icon = svg_icon(ICON_CHECK, 56, COLOR_NORMAL, stroke_width=1.8)
        accent = COLOR_NORMAL
        gradient = "linear-gradient(135deg, #064E3B 0%, #022C22 100%)"
        message_color = "#A7F3D0"
        track_color = "rgba(2, 44, 34, 0.8)"
        confidence = normal_probability

    render_html(f"""
        <div style="background:{gradient}; border-left:4px solid {accent}; border-radius:8px;
                    padding:24px; margin-bottom:1.1rem;
                    box-shadow: 0 18px 40px -22px rgba(0, 0, 0, 0.95);">
            <div style="display:flex; align-items:center; gap:1.1rem;">
                {icon}
                <div>
                    <div style="font-size:0.72rem; font-weight:600; letter-spacing:0.16em;
                                text-transform:uppercase; color:{message_color}; opacity:0.85;">
                        Clasificación del modelo
                    </div>
                    <div style="margin-top:0.25rem; font-size:1.5rem; font-weight:700;
                                letter-spacing:0.1em; line-height:1.2; color:#FFFFFF;">
                        DIAGNÓSTICO: {label}
                    </div>
                </div>
            </div>
            <div style="margin-top:0.9rem; font-size:0.95rem; color:{message_color};">{message}</div>
            <div style="margin-top:1.3rem; font-size:0.72rem; font-weight:600; letter-spacing:0.16em;
                        text-transform:uppercase; color:{message_color}; opacity:0.85;">
                Nivel de confianza
            </div>
            <div style="display:flex; align-items:center; gap:1rem; margin-top:0.45rem;">
                <div style="flex:1; background:{track_color}; border:1px solid rgba(255, 255, 255, 0.08);
                            border-radius:4px; height:10px; overflow:hidden;">
                    <div style="width:{confidence * 100:.2f}%; height:100%; background:{accent};"></div>
                </div>
                <div style="font-size:1.9rem; font-weight:700; color:#FFFFFF;
                            font-variant-numeric:tabular-nums; line-height:1;">
                    {confidence:.1%}
                </div>
            </div>
        </div>
    """)

    if is_pneumonia:
        render_html(f"""
            <div style="display:flex; gap:0.85rem; align-items:flex-start; background:{COLOR_SURFACE};
                        border:1px solid #7F1D1D; border-left:4px solid {COLOR_PNEUMONIA};
                        border-radius:8px; padding:16px 18px; margin-bottom:1.1rem;">
                <span style="padding-top:2px;">{svg_icon(ICON_ALERT, 18, COLOR_PNEUMONIA_ICON)}</span>
                <div>
                    <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.16em;
                                text-transform:uppercase; color:{COLOR_PNEUMONIA_ICON};">
                        Recomendación clínica
                    </div>
                    <div style="margin-top:0.3rem; font-size:0.95rem; color:{COLOR_TEXT};">
                        Se recomienda la consulta urgente con un radiólogo o neumólogo certificado.
                    </div>
                </div>
            </div>
        """)

    render_html(f"""
        <div style="background:{COLOR_SURFACE}; border:1px solid {COLOR_BORDER}; border-radius:8px;
                    padding:20px 22px;">
            <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.16em;
                        text-transform:uppercase; color:{COLOR_ACCENT};">
                Distribución de probabilidad
            </div>
            {probability_row("NORMAL", normal_probability, COLOR_NORMAL)}
            {probability_row("PNEUMONIA", pneumonia_probability, COLOR_PNEUMONIA)}
            <div style="margin-top:1rem; font-size:0.78rem; color:{COLOR_SUBTLE};">
                Umbral de decisión: {DECISION_THRESHOLD}. Una probabilidad de PNEUMONIA superior
                al umbral clasifica la imagen como PNEUMONIA.
            </div>
        </div>
    """)


def render_footer():
    """Disclaimer médico obligatorio y créditos académicos."""
    render_html(f"""
        <hr style="border:none; border-top:1px solid {COLOR_BORDER}; margin:2.8rem 0 1.3rem;">
        <div style="background:transparent; color:{COLOR_SUBTLE};">
            <div style="display:flex; align-items:center; gap:0.45rem; color:{COLOR_MUTED};">
                {svg_icon(ICON_ALERT, 12)}
                <span style="font-size:0.72rem; font-weight:700; letter-spacing:0.16em;">AVISO MÉDICO</span>
            </div>
            <div style="margin-top:0.45rem; font-size:0.85rem; line-height:1.55; max-width:860px;">
                {DISCLAIMER_TEXT}
            </div>
            <div style="margin-top:1.1rem; font-size:0.7rem; letter-spacing:0.04em; color:{COLOR_FAINT};">
                {COPYRIGHT_TEXT}
            </div>
        </div>
    """)


# ---------------------------------------------------------------------------
# Aplicación principal
# ---------------------------------------------------------------------------

def main():
    inject_styles()
    render_sidebar()
    render_header()

    try:
        model = load_model()
    except Exception as error:
        st.error(f"No se pudo cargar el modelo desde `{MODEL_PATH}`.", icon=":material/error:")
        st.exception(error)
        render_footer()
        st.stop()

    uploaded_file = render_upload_zone()

    if uploaded_file is None:
        render_waiting_state()
        render_footer()
        return

    try:
        image = Image.open(uploaded_file)
        image.load()
    except (UnidentifiedImageError, OSError):
        st.error(
            "El archivo no es una imagen válida o está dañado. Intente con otro archivo.",
            icon=":material/error:",
        )
        render_footer()
        return

    col_image, col_result = st.columns(2, gap="large")

    with col_image:
        section_title("Imagen de entrada")
        st.image(image, caption=uploaded_file.name, width="stretch")

    with col_result:
        section_title("Análisis del modelo")
        try:
            with st.spinner("Analizando la radiografía..."):
                image_array = preprocess_image(image)
                pneumonia_probability = predict_pneumonia(model, image_array)
        except Exception as error:
            st.error("Ocurrió un error al procesar la imagen.", icon=":material/error:")
            st.exception(error)
        else:
            render_result(pneumonia_probability)

    render_footer()


if __name__ == "__main__":
    main()
