# 🫁 Detector de Neumonía en Rayos X

Aplicación web que clasifica radiografías de tórax en **NORMAL** o **PNEUMONIA** usando una Red Neuronal Convolucional (CNN) entrenada desde cero.

🔗 **Demo en línea:** _(URL de Streamlit Community Cloud, se agrega tras el despliegue)_

## 📋 Descripción

Proyecto desarrollado siguiendo la metodología **CRISP-DM**. Se entrenaron y compararon tres modelos (ANN, CNN desde cero y Transfer Learning con MobileNetV2). La CNN desde cero obtuvo el mejor desempeño y es el modelo desplegado en esta aplicación.

## 📊 Dataset

- **Nombre:** Chest X-Ray Pneumonia
- **Fuente:** [Kaggle — paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total:** ~5,840 imágenes
- **Clases:** NORMAL (26%) y PNEUMONIA (74%)

## 🧠 Modelo

Arquitectura de la CNN:

```
Input (150, 150, 1)
→ Conv2D(32, 3x3, relu) + MaxPooling2D(2x2)
→ Conv2D(64, 3x3, relu) + MaxPooling2D(2x2)
→ Conv2D(128, 3x3, relu) + MaxPooling2D(2x2)
→ Conv2D(128, 3x3, relu) + MaxPooling2D(2x2)
→ Flatten
→ Dense(128, relu) + Dropout(0.5)
→ Dense(1, sigmoid)
```

- **Entrada:** imagen en escala de grises de 150 × 150 píxeles, normalizada a [0, 1]
- **Salida:** probabilidad de PNEUMONIA (umbral de decisión: 0.5)

## 📈 Resultados en test (624 imágenes)

| Métrica | Valor |
|---|---|
| Accuracy | 88.94% |
| Precision (PNEUMONIA) | 85.75% |
| Recall (PNEUMONIA) | 98.72% |
| F1-Score (PNEUMONIA) | 91.78% |
| Falsos Negativos | 5 de 390 |
| Falsos Positivos | 64 de 234 |

### Comparación de modelos

| Modelo | Accuracy | Recall PNEUMONIA | FN |
|---|---|---|---|
| ANN Base | 79.17% | 87.18% | 50 |
| **CNN desde Cero** | **88.94%** | **98.72%** | **5** |
| Transfer Learning (MobileNetV2) | 86.22% | 98.21% | 7 |

## 🛠️ Tecnologías

- Python 3.11
- Streamlit
- TensorFlow / Keras
- Pillow
- NumPy

## 🚀 Ejecución local

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd parcial-neumonia-app

# 2. Crear y activar el entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

La aplicación queda disponible en `http://localhost:8501`.

## 📁 Estructura del proyecto

```
parcial-neumonia-app/
├── app.py                 # Aplicación Streamlit
├── requirements.txt       # Dependencias
├── README.md              # Este archivo
├── .gitignore
├── models/
│   └── model_cnn.keras    # Modelo entrenado
└── assets/
    └── sample_images/     # Imágenes de prueba
```

## ⚠️ Disclaimer médico

> Esta herramienta es con fines educativos. No reemplaza el diagnóstico de un profesional médico. Siempre consulte a un radiólogo o neumólogo certificado.
