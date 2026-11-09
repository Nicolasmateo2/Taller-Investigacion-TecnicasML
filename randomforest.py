import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------
# 1. Configuración de Archivos y Carga del Dataset (Breast Cancer Wisconsin)
# -------------------------------------------------------------------------
os.makedirs("resultados", exist_ok=True)

# Carga del dataset oficial desde scikit-learn
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("============================================================")
print("INFORMACIÓN GENERAL DEL DATASET")
print("============================================================")
print(f"Dimensiones de X: {X.shape[0]} muestras, {X.shape[1]} variables (M)")
print(f"Clases a predecir: {data.target_names.tolist()} (0 = Maligno, 1 = Benigno)")
print(f"Distribución de clases:\n{y.value_counts()}")

# Partición estratificada (80% Entrenamiento, 20% Prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nPartición final:")
print(f" - Conjunto de Entrenamiento (Train): {X_train.shape[0]} registros")
print(f" - Conjunto de Prueba (Test):        {X_test.shape[0]} registros")


# -------------------------------------------------------------------------
# 2. Instanciación y Entrenamiento de Random Forest
# -------------------------------------------------------------------------
# Configuración según fundamentos teóricos del artículo:
# B = 100 árboles (n_estimators)
# m = sqrt(M) características por nodo (max_features='sqrt')
# Impureza de Gini (criterion='gini')
# Score Out-of-Bag (oob_score=True)
rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_features="sqrt",
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)

# Entrenamiento del ensamble
rf_model.fit(X_train, y_train)

print("\n============================================================")
print("ENTRENAMIENTO Y ESTIMACIÓN OUT-OF-BAG (OOB)")
print("============================================================")
print(f"Estimación de Exactitud OOB (OOB Score): {rf_model.oob_score_:.4f}")
print(f"Error de Generalización OOB:             {1 - rf_model.oob_score_:.4f}")


# -------------------------------------------------------------------------
# 3. Evaluación de Predicciones en el Conjunto de Prueba
# -------------------------------------------------------------------------
y_pred = rf_model.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
test_prec = precision_score(y_test, y_pred)
test_rec = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print("\n============================================================")
print("EVALUACIÓN EN EL CONJUNTO DE PRUEBA (TEST)")
print("============================================================")
print(f"Exactitud (Accuracy):  {test_acc:.4f}")
print(f"Precisión (Precision): {test_prec:.4f}")
print(f"Sensibilidad (Recall): {test_rec:.4f}")
print(f"F1-Score:              {test_f1:.4f}")
print("\nReporte de Clasificación Detallado:")
print(classification_report(y_test, y_pred, target_names=data.target_names, digits=4))

# Generación y guardado de la Matriz de Confusión
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=data.target_names,
    cmap="Blues",
    ax=ax,
    colorbar=False,
)
plt.title("Matriz de Confusión - Random Forest Classifier", fontweight="bold")
plt.tight_layout()
plt.savefig("resultados/matriz_confusion_rf.png", dpi=300)
plt.close()  # Reemplazado plt.show() por plt.close() para no bloquear la ejecución


# -------------------------------------------------------------------------
# 4. Análisis de Importancia de Variables (Feature Importance)
# -------------------------------------------------------------------------
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_10_importances = importances.nlargest(10).sort_values(ascending=True)

print("\n============================================================")
print("TOP 10 VARIABLES MÁS IMPORTANTES (GINI IMPORTANCE)")
print("============================================================")
print(top_10_importances.sort_values(ascending=False).round(4))

# Visualización gráfica del Top 10 de variables predictoras
plt.figure(figsize=(9, 6))
top_10_importances.plot(kind="barh", color="#2b5c8f", edgecolor="black")
plt.title(
    "Top 10 Características según Reducción Media de Impureza (Gini)",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Importancia Relativa", fontsize=10)
plt.ylabel("Característica Diagnóstica", fontsize=10)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("resultados/feature_importance_rf.png", dpi=300)
plt.close()  