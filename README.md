# 🏥 Imbalance Medical
### Clasificación con Datos Desbalanceados en Medicina

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=for-the-badge&logo=scikit-learn)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.11+-green?style=for-the-badge)
![uv](https://img.shields.io/badge/uv-package%20manager-purple?style=for-the-badge)

> Proyecto educativo que demuestra cómo los **datos desbalanceados** afectan a los modelos de clasificación en contextos médicos, comparando un modelo sin balanceo vs uno con **SMOTE**.

---

## 🔬 ¿Qué simula este proyecto?

| Clase | Descripción | Proporción |
|-------|-------------|------------|
| `0` | Enfermedad común | 95% |
| `1` | Enfermedad rara | 5% |

---

## 🚀 Instalación y Ejecución Paso a Paso

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU-USUARIO/imbalance-medical.git
cd imbalance-medical
```

---

### 2. Crear el entorno virtual con `uv`

> Si no tienes `uv` instalado:

```bash
pip install uv
```

> Crear el entorno virtual:

```bash
uv venv
```

---

### 3. Activar el entorno virtual

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Git Bash / Mac / Linux:**
```bash
source .venv/Scripts/activate
```

> ✅ Si está activo verás `(.venv)` al inicio de la terminal.

---

### 4. Instalar dependencias

```bash
uv sync
```

---

### 5. Ejecutar el proyecto

```bash
uv run python main.py
```

---

## 📁 Estructura del Proyecto

```
imbalance-medical/
│
├── main.py          ← código principal
├── pyproject.toml   ← dependencias del proyecto
├── uv.lock          ← versiones exactas instaladas
└── README.md
```

---

## 🧠 Código — `main.py`

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ── Crear dataset desbalanceado ───────────────────────────────────
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.95],       # 95% enfermedad común, 5% rara
    flip_y=0,
    random_state=42
)

print("Distribución original:")
print(f"  Clase 0 (común): {sum(y == 0)}")
print(f"  Clase 1 (rara):  {sum(y == 1)}")

# ── Dividir en entrenamiento y prueba ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ── Modelo SIN balanceo ──────────────────────────────────────────
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n========== SIN BALANCEO ==========")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# ── Aplicar SMOTE ────────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Distribución después de SMOTE:")
print(f"  Clase 0: {sum(y_train_res == 0)}")
print(f"  Clase 1: {sum(y_train_res == 1)}")

# ── Modelo CON SMOTE ─────────────────────────────────────────────
model_smote = LogisticRegression()
model_smote.fit(X_train_res, y_train_res)
y_pred_smote = model_smote.predict(X_test)

print("\n========== CON SMOTE ==========")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred_smote))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_smote))
```

---

## ⚙️ Configuración — `pyproject.toml`

```toml
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
```

---

## 📊 ¿Qué resultados verás?

| Métrica | Sin SMOTE | Con SMOTE |
|---------|-----------|-----------|
| Accuracy | Alta ✅ | Ligeramente menor |
| Recall clase rara | Bajo ❌ | Alto ✅ |
| F1-Score clase rara | Bajo ❌ | Alto ✅ |
| Utilidad clínica | ❌ Pobre | ✅ Buena |

---

## ⚠️ Problema común en VS Code

Si aparece el error:
```
Import "sklearn" could not be resolved
```

**Solución:**
1. Presiona `Ctrl + Shift + P`
2. Escribe **Python: Select Interpreter**
3. Selecciona el intérprete de `.venv`

---

## 💡 Conclusión

> **Una accuracy alta NO significa que el modelo sea útil.**
> En medicina, si no detectas la enfermedad rara, el modelo falla clínicamente sin importar su porcentaje de aciertos.

### Conceptos cubiertos
- Datos desbalanceados
- Accuracy vs Recall
- Matriz de Confusión
- F1-Score
- SMOTE
- Clasificación binaria
