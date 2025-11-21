import pandas as pd
import numpy as np
import os

# Obtener la ruta del directorio actual del script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# --- 1. Definir Nombres de Archivos ---
archivo_leads = os.path.join(PROJECT_ROOT, 'data/raw/leads_historicos.csv')
archivo_comportamiento = os.path.join(PROJECT_ROOT, 'data/raw/clientes_comportamiento.csv')
archivo_transacciones = os.path.join(PROJECT_ROOT, 'data/raw/clientes_transacciones.csv')


# --- 2. Carga y Fusión de Datos ---
# Cargar DataFrames
df_leads = pd.read_csv(archivo_leads)
df_comp = pd.read_csv(archivo_comportamiento)
df_trans = pd.read_csv(archivo_transacciones)

# Renombrar columnas en df_trans para claridad antes de la fusión
df_trans.rename(columns={'presupuesto': 'presupuesto_empresa',
                              'industria': 'industria_empresa'}, inplace=True)

    # Realizar la fusión (JOIN) de las dos tablas de clientes
    # Clave: 'id_cliente'
df_clientes_full = pd.merge(
        df_comp,
        df_trans,
        on='id_cliente',
        how='inner'
    )
    
    # Crear la variable objetivo 'churn_riesgo'
MEDIANA_DIAS = df_clientes_full['dias_desde_ultima_compra'].median()
df_clientes_full['churn_riesgo'] = np.where(
        (df_clientes_full['categoria_cliente'] == 'Nuevo') & (df_clientes_full['dias_desde_ultima_compra'] > MEDIANA_DIAS),
        1, # Riesgo de Churn
        0  # No Riesgo
    )

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

# Ejecutar el código del Paso 1 para cargar y fusionar df_clientes_full y df_leads

# --- A. Preparación de Datos para el Modelo de CHURN ---
print("Iniciando preparación de datos para el modelo de CHURN...")

clientes_excluir = ['id_cliente', 'categoria_cliente']
df_churn = df_clientes_full.drop(columns=clientes_excluir)

Y_churn = df_churn['churn_riesgo']
X_churn = df_churn.drop(columns=['churn_riesgo'])

numeric_features_churn = X_churn.select_dtypes(include=np.number).columns.tolist()
categorical_features_churn = X_churn.select_dtypes(include='object').columns.tolist()

numeric_transformer_churn = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# CORRECCIÓN: Se añade sparse_output=False al OneHotEncoder para compatibilidad
categorical_transformer_churn = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Usamos sparse_output=False para versiones recientes (>=1.2)
    # Si sigue fallando, cambie sparse_output=False a sparse=False para versiones muy antiguas.
])

# Se quita sparse_output del ColumnTransformer
preprocessor_churn = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_churn, numeric_features_churn),
        ('cat', categorical_transformer_churn, categorical_features_churn)
    ],
    remainder='passthrough'
)

X_churn_processed = preprocessor_churn.fit_transform(X_churn)

feature_names_churn = (numeric_features_churn +
                      list(preprocessor_churn.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features_churn)))

X_churn_final = pd.DataFrame(X_churn_processed, columns=feature_names_churn)

print("\n--- Características de CHURN Procesadas con éxito ---")
print(f"Dimensiones de X_churn_final: {X_churn_final.shape}")


# -----------------------------------------------------------------------------

# --- B. Preparación de Datos para el Modelo de Calidad de LEADS ---

print("\nIniciando preparación de datos para el modelo de LEADS...")

Y_leads = df_leads['compro'].apply(lambda x: 1 if x == 'Sí' else 0)

# Variables de alta cardinalidad excluidas
leads_excluir = ['lead_id', 'fecha_lead', 'compro', 'status', 'observacion_asesor', 'empresa_lead', 'programa_producto_interes']
X_leads = df_leads.drop(columns=leads_excluir)

numeric_features_leads = X_leads.select_dtypes(include=np.number).columns.tolist()
categorical_features_leads = X_leads.select_dtypes(include='object').columns.tolist()

numeric_transformer_leads = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# CORRECCIÓN: Se añade sparse_output=False al OneHotEncoder para compatibilidad
categorical_transformer_leads = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='desconocido')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Usamos sparse_output=False
])

# Se quita sparse_output del ColumnTransformer
preprocessor_leads = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_leads, numeric_features_leads),
        ('cat', categorical_transformer_leads, categorical_features_leads)
    ],
    remainder='passthrough'
)

X_leads_processed = preprocessor_leads.fit_transform(X_leads)

feature_names_leads = (numeric_features_leads +
                       list(preprocessor_leads.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features_leads)))

X_leads_final = pd.DataFrame(X_leads_processed, columns=feature_names_leads)

print("\n--- Características de LEADS Procesadas con éxito ---")
print(f"Dimensiones de X_leads_final: {X_leads_final.shape}")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import numpy as np

# ASUMIMOS que X_churn_final y Y_churn están disponibles del Paso 2

print("\n--- PASO 3: Entrenamiento del Modelo de CHURN (Regresión Logística) ---")

# 1. Dividir los datos en conjuntos de entrenamiento y prueba
# Usamos 70% para entrenar y 30% para probar (test_size=0.3)
X_train_churn, X_test_churn, Y_train_churn, Y_test_churn = train_test_split(
    X_churn_final,
    Y_churn,
    test_size=0.3,
    random_state=42,
    stratify=Y_churn # Mantiene la proporción de la variable objetivo
)

print(f"Conjunto de entrenamiento (X_train_churn): {X_train_churn.shape}")
print(f"Conjunto de prueba (X_test_churn): {X_test_churn.shape}")


# 2. Inicializar y Entrenar el Modelo
# Un modelo de Regresión Logística es un buen punto de partida para clasificación binaria.
modelo_churn = LogisticRegression(
    solver='liblinear',  # Algoritmo eficiente para conjuntos pequeños
    class_weight='balanced', # Ajusta el modelo para clases desbalanceadas
    random_state=42
)

print("\nEntrenando el modelo...")
modelo_churn.fit(X_train_churn, Y_train_churn)
print("Entrenamiento completado.")


# 3. Predicción y Evaluación
Y_pred_churn = modelo_churn.predict(X_test_churn)

print("\n--- Resultados de la Evaluación del Modelo de CHURN ---")

# Métrica de Precisión (Accuracy)
accuracy = accuracy_score(Y_test_churn, Y_pred_churn)
print(f"Precisión del Modelo (Accuracy): {accuracy:.4f}")

# Reporte de Clasificación (Incluye Precision, Recall, F1-Score)
print("\nReporte de Clasificación:")
print(classification_report(Y_test_churn, Y_pred_churn))

# Matriz de Confusión
cm = confusion_matrix(Y_test_churn, Y_pred_churn)
print("\nMatriz de Confusión (CM):")
print("  Predicción: 0  | 1")
print(f"Real 0:    {cm[0][0]:<5} | {cm[0][1]}")
print(f"Real 1:    {cm[1][0]:<5} | {cm[1][1]}")

# 4. Interpretación Rápida
# Obtener los coeficientes para ver qué variables influyen más en el riesgo de Churn (1)
coeficientes = pd.Series(modelo_churn.coef_[0], index=X_churn_final.columns).sort_values(ascending=False)

print("\n--- Top 5 Variables más Positivamente Influyentes en CHURN (Riesgo Alto) ---")
print(coeficientes.head(5))

# Interpretación: Un coeficiente positivo fuerte indica que si esa variable aumenta, el riesgo de churn (1) aumenta.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import numpy as np

# ASUMIMOS que X_leads_final y Y_leads están disponibles del Paso 2

print("\n--- PASO 4: Entrenamiento del Modelo de LEAD SCORING ---")

# 1. Dividir los datos en conjuntos de entrenamiento y prueba
# Usamos 70% para entrenar y 30% para probar (test_size=0.3)
X_train_leads, X_test_leads, Y_train_leads, Y_test_leads = train_test_split(
    X_leads_final,
    Y_leads,
    test_size=0.3,
    random_state=42,
    stratify=Y_leads # Mantiene la proporción de la variable objetivo
)

print(f"Conjunto de entrenamiento (X_train_leads): {X_train_leads.shape}")
print(f"Conjunto de prueba (X_test_leads): {X_test_leads.shape}")


# 2. Inicializar y Entrenar el Modelo
# Regresión Logística para obtener probabilidades (Score) y simplicidad
modelo_leads = LogisticRegression(
    solver='liblinear',
    class_weight='balanced', # Crucial ya que la conversión suele ser baja
    random_state=42
)

print("\nEntrenando el modelo...")
modelo_leads.fit(X_train_leads, Y_train_leads)
print("Entrenamiento completado.")


# 3. Predicción y Evaluación
Y_pred_leads = modelo_leads.predict(X_test_leads)
Y_proba_leads = modelo_leads.predict_proba(X_test_leads)[:, 1] # Probabilidad de ser un lead 'comprador' (1)

print("\n--- Resultados de la Evaluación del Modelo de LEAD SCORING ---")

# Métrica de Precisión (Accuracy)
accuracy = accuracy_score(Y_test_leads, Y_pred_leads)
print(f"Precisión del Modelo (Accuracy): {accuracy:.4f}")

# Reporte de Clasificación (Precision, Recall, F1-Score)
print("\nReporte de Clasificación:")
print(classification_report(Y_test_leads, Y_pred_leads))

# Matriz de Confusión
cm = confusion_matrix(Y_test_leads, Y_pred_leads)
print("\nMatriz de Confusión (CM):")
print("  Predicción: 0 (No Compra) | 1 (Compra)")
print(f"Real 0:    {cm[0][0]:<5} | {cm[0][1]}")
print(f"Real 1:    {cm[1][0]:<5} | {cm[1][1]}")

# 4. Interpretación Rápida (Variables Clave)
# Obtener los coeficientes para ver qué variables influyen más en la conversión (Score Alto)
coeficientes = pd.Series(modelo_leads.coef_[0], index=X_leads_final.columns).sort_values(ascending=False)

print("\n--- Top 5 Variables más Positivamente Influyentes en la CONVERSIÓN (Score Alto) ---")
print(coeficientes.head(5))

# Guardar los modelos y preprocesadores (Opcional, pero útil para el Dashboard)
# Guardamos los modelos entrenados y los DataFrames finales para que los puedas usar en tu dashboard.
# Si quieres guardar esto para usarlo en otro archivo (como en Streamlit o Power BI),
# necesitarás la librería joblib (pip install joblib).

try:
    import joblib
    joblib.dump(modelo_churn, os.path.join(PROJECT_ROOT, 'models/modelo_churn.pkl'))
    joblib.dump(modelo_leads, os.path.join(PROJECT_ROOT, 'models/modelo_leads.pkl'))
    joblib.dump(preprocessor_churn, os.path.join(PROJECT_ROOT, 'models/preprocessor_churn.pkl'))
    joblib.dump(preprocessor_leads, os.path.join(PROJECT_ROOT, 'models/preprocessor_leads.pkl'))
    # También puedes guardar los DataFrames procesados como CSV
    X_churn_final.to_csv(os.path.join(PROJECT_ROOT, 'data/processed/X_churn_processed.csv'), index=False)
    X_leads_final.to_csv(os.path.join(PROJECT_ROOT, 'data/processed/X_leads_processed.csv'), index=False)
    
    print("\nModelos y DataFrames guardados (.pkl y .csv) para el dashboard.")
except ImportError:
    print("\nAdvertencia: La librería 'joblib' no está instalada. No se pudieron guardar los modelos para uso futuro en el dashboard.")