# ==========================================
# PROYECTO: Datos Desbalanceados en Medicina
# ==========================================

# -------------------------------------------------
# 1️⃣ IMPORTACIÓN DE LIBRERÍAS
# -------------------------------------------------

# numpy es la librería base para trabajar con arrays numéricos.
# Aunque aquí no la usamos de forma avanzada, es estándar en ML.
import numpy as np  

# make_classification nos permite crear un dataset sintético
# para simular un problema real de clasificación.
from sklearn.datasets import make_classification  

# train_test_split divide nuestros datos en entrenamiento y prueba.
from sklearn.model_selection import train_test_split  

# LogisticRegression es el modelo de clasificación que usaremos.
from sklearn.linear_model import LogisticRegression  

# classification_report nos da métricas como precision, recall y f1-score.
# confusion_matrix nos muestra la matriz de confusión.
from sklearn.metrics import classification_report, confusion_matrix  

# SMOTE es la técnica que usaremos para balancear la clase minoritaria.
from imblearn.over_sampling import SMOTE  


# -------------------------------------------------
# 2️⃣ CREACIÓN DEL DATASET DESBALANCEADO
# -------------------------------------------------

# Creamos un dataset artificial que simula pacientes.
# X serán las características (biomarcadores)
# y será la etiqueta (0 = común, 1 = rara)
X, y = make_classification(
    
    n_samples=1000,        # Creamos 1000 pacientes
    
    n_features=2,          # Cada paciente tiene 2 biomarcadores
    
    n_redundant=0,         # No queremos variables redundantes
    
    n_clusters_per_class=1,# Cada clase tendrá un solo grupo
    
    weights=[0.95],        # 95% clase 0 (enfermedad común)
                            # 5% clase 1 (enfermedad rara)
    
    flip_y=0,              # No agregamos ruido en las etiquetas
    
    random_state=42        # Fijamos semilla para reproducibilidad
)

# Mostramos cuántos pacientes hay por clase
print("Distribución original:")

# sum(y == 0) cuenta cuántos valores en y son iguales a 0
print("Clase 0 (común):", sum(y == 0))

# sum(y == 1) cuenta cuántos valores en y son iguales a 1
print("Clase 1 (rara):", sum(y == 1))


# -------------------------------------------------
# 3️⃣ DIVISIÓN EN ENTRENAMIENTO Y PRUEBA
# -------------------------------------------------

# Dividimos el dataset:
# 70% entrenamiento
# 30% prueba
# Esto es fundamental para evaluar el modelo correctamente
X_train, X_test, y_train, y_test = train_test_split(
    
    X, y,
    
    test_size=0.3,         # 30% para evaluación
    
    random_state=42        # Reproducibilidad
)


# -------------------------------------------------
# 4️⃣ MODELO SIN BALANCEO
# -------------------------------------------------

# Creamos el modelo de Regresión Logística
model = LogisticRegression()

# Entrenamos el modelo usando los datos desbalanceados
model.fit(X_train, y_train)

# El modelo hace predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Mostramos resultados
print("\n===== SIN BALANCEO =====")

# Matriz de confusión:
# [[TN, FP],
#  [FN, TP]]
print(confusion_matrix(y_test, y_pred))

# Reporte completo:
# precision, recall, f1-score, soporte
print(classification_report(y_test, y_pred))


# -------------------------------------------------
# 5️⃣ APLICAMOS SMOTE
# -------------------------------------------------

# Creamos el objeto SMOTE
smote = SMOTE(random_state=42)

# fit_resample hace dos cosas:
# 1. Aprende la distribución de la clase minoritaria
# 2. Genera ejemplos sintéticos para balancear
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Mostramos la nueva distribución
print("\nDistribución después de SMOTE:")

print("Clase 0:", sum(y_train_res == 0))
print("Clase 1:", sum(y_train_res == 1))


# -------------------------------------------------
# 6️⃣ MODELO CON SMOTE
# -------------------------------------------------

# Creamos un nuevo modelo
model_smote = LogisticRegression()

# Lo entrenamos ahora con datos balanceados
model_smote.fit(X_train_res, y_train_res)

# Hacemos predicciones sobre el mismo conjunto de prueba original
y_pred_smote = model_smote.predict(X_test)

print("\n===== CON SMOTE =====")

# Mostramos matriz de confusión
print(confusion_matrix(y_test, y_pred_smote))

# Mostramos métricas
print(classification_report(y_test, y_pred_smote))

#Aquí ocurre algo clave: no estamos duplicando pacientes.
#Estamos generando pacientes sintéticos matemáticamente coherentes interpolando entre casos reales de la clase minoritaria.

# Y cuando compares resultados: Observen cómo la accuracy puede bajar ligeramente,pero el recall de la enfermedad rara aumenta significativamente.Eso en medicina salva vidas