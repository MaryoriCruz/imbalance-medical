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
bashgit clone https://github.com/TU-USUARIO/imbalance-medical.git
Entrar en la carpeta:
bashcd imbalance-medical

2️⃣ Crear el entorno virtual con uv
Si no tienes uv:
bashpip install uv
Crear entorno virtual:
bashuv venv
Esto generará la carpeta .venv/

3️⃣ Activar el entorno virtual
Windows PowerShell:
powershell.\.venv\Scripts\Activate.ps1
Git Bash:
bashsource .venv/Scripts/activate

Si todo está correcto, verás (.venv) al inicio de la línea.


4️⃣ Instalar dependencias
bashuv sync
Esto instalará automáticamente:

numpy
scikit-learn
imbalanced-learn


5️⃣ Estructura del proyecto
imbalance-medical/
│
├── main.py
├── pyproject.toml
├── uv.lock
├── README.md
└── .venv/

📦 Código del Proyecto
main.py
python# ==========================================
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
toml[project]
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
bashpython main.py
O sin activarlo:
bashuv run python main.py

📊 ¿Qué observarás?
Sin SMOTECon SMOTEAccuracyAltaLigeramente menorRecall (enf. rara)Bajo ❌Alto ✅F1-Score (enf. rara)Bajo ❌Alto ✅Utilidad clínicaPobreBuena

⚠️ Problema común en VS Code
Si aparece este error:
Import "sklearn" could not be resolved
Solución:

Ctrl + Shift + P
Buscar Python: Select Interpreter
Seleccionar .venv


🧠 ¿Qué demuestra esto?
En problemas médicos:

Una accuracy alta NO significa que el modelo sea útil.
Si no detectas la enfermedad rara, el modelo no sirve clínicamente.

Conceptos cubiertos:

Datos desbalanceados
Accuracy vs Recall
Matriz de Confusión
F1-Score
SMOTE
Clasificación binaria