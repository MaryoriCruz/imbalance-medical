🏥 Imbalance Medical — Detección de Enfermedades Raras

Proyecto educativo que demuestra cómo los datos desbalanceados afectan a los modelos de clasificación en contextos médicos.

Simulación:

95% enfermedad común

5% enfermedad rara

Comparación:

Regresión Logística sin balanceo

Regresión Logística con SMOTE

🚀 Instalación Paso a Paso
1️⃣ Clonar el repositorio
git clone https://github.com/TU-USUARIO/imbalance-medical.git
cd imbalance-medical

2️⃣ Crear entorno virtual
uv venv

3️⃣ Instalar dependencias
uv sync

4️⃣ Ejecutar el proyecto
uv run python main.py

📊 ¿Qué demuestra este proyecto?

Sin balanceo:

Alta accuracy

Recall bajo en la clase rara

Con SMOTE:

Mejor recall

Mejor F1-score

Modelo más útil clínicamente

🧠 Conceptos cubiertos

Datos desbalanceados

Accuracy vs Recall

Matriz de Confusión

F1-Score

SMOTE

Clasificación binaria

⚠️ Problema común en VS Code

Si aparece:

Import "sklearn" could not be resolved


Solución:

Ctrl + Shift + P

Python: Select Interpreter

Seleccionar .venv

🎯 Objetivo Académico

Mostrar que una alta accuracy no garantiza un modelo útil en problemas médicos donde detectar casos raros es crítico.

🔥 Ahora sí…

Cuando alguien clone el repo solo necesita:

uv sync
uv run python main.py


Y funciona.

Si quieres, el siguiente nivel sería:

Agregar visualización gráfica

Agregar comparación con class_weight

Convertirlo en demo interactiva con Streamlit

O dejarlo estilo paper reproducible

¿Lo dejamos académico… o lo subimos a modo portafolio senior? 😏🚀

no, empieza desde desde como lo clonas, luego lo que tienes que activar y luego el codigo y asi sucesivamente

Perfecto.
Vamos a armar un README ultra claro, paso a paso desde cero, como si tu compa nunca hubiera usado uv en su vida 😄🚀

Te lo dejo listo para copiar y pegar en tu README.md.

🏥 Imbalance Medical — Clasificación con Datos Desbalanceados

Este proyecto demuestra cómo los datos desbalanceados afectan a los modelos de Machine Learning en un contexto médico.

Simulamos:

95% pacientes con enfermedad común

5% pacientes con enfermedad rara

Y comparamos:

Regresión Logística sin balanceo

Regresión Logística con SMOTE

🚀 PASO A PASO DESDE CERO
1️⃣ Clonar el repositorio

Abrir terminal y ejecutar:

git clone https://github.com/TU-USUARIO/imbalance-medical.git


Entrar en la carpeta:

cd imbalance-medical

2️⃣ Crear el entorno virtual con uv

Si no tienes uv:

pip install uv


Crear entorno virtual:

uv venv


Esto generará:

.venv/

3️⃣ Activar el entorno virtual
Windows PowerShell:
.\.venv\Scripts\Activate.ps1

Git Bash:
source .venv/Scripts/activate


Si todo está correcto, verás (.venv) al inicio de la línea.

4️⃣ Instalar dependencias

Ejecutar:

uv sync


Esto instalará automáticamente:

numpy

scikit-learn

imbalanced-learn

5️⃣ Estructura del Proyecto

El proyecto debe verse así:

imbalance-medical/
│
├── main.py
├── pyproject.toml
├── uv.lock
├── README.md
└── .venv/

📦 Código del Proyecto
main.py

Crear un archivo llamado main.py y pegar lo siguiente:

# ==========================================
# PROYECTO: Datos Desbalanceados en Medicina
# ==========================================

# 1️⃣ Importamos librerías
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


# 2️⃣ Creamos dataset desbalanceado
X, y = make_classification(
    n_samples=1000,        # Total pacientes
    n_features=2,          # Biomarcadores
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.95],        # 95% enfermedad común
    flip_y=0,
    random_state=42
)

print("Distribución original:")
print("Clase 0 (común):", sum(y == 0))
print("Clase 1 (rara):", sum(y == 1))


# 3️⃣ Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)


# 4️⃣ Modelo sin balanceo
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n===== SIN BALANCEO =====")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))


# 5️⃣ Aplicar SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nDistribución después de SMOTE:")
print("Clase 0:", sum(y_train_res == 0))
print("Clase 1:", sum(y_train_res == 1))


# 6️⃣ Modelo con SMOTE
model_smote = LogisticRegression()
model_smote.fit(X_train_res, y_train_res)

y_pred_smote = model_smote.predict(X_test)

print("\n===== CON SMOTE =====")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred_smote))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_smote))

pyproject.toml

Asegúrate de que contenga:

[project]
name = "imbalance-medical"
version = "0.1.0"
description = "Imbalanced medical classification demo using Logistic Regression and SMOTE"
requires-python = ">=3.10"

dependencies = [
    "numpy",
    "scikit-learn",
    "imbalanced-learn"
]

▶️ Ejecutar el Proyecto

Con el entorno activado:

python main.py


O sin activarlo:

uv run python main.py

📊 ¿Qué observarás?
Sin SMOTE:

Alta accuracy

Bajo recall en enfermedad rara

Con SMOTE:

Mejor recall

Mejor F1-score

Mejor detección de la clase minoritaria

🧠 ¿Qué demuestra esto?

En problemas médicos:

Una accuracy alta NO significa que el modelo sea útil.

Si no detectas la enfermedad rara, el modelo no sirve clínicamente.