# Sistema de Inteligencia de Clientes

Este proyecto implementa un sistema integral para la gestión de clientes y leads, utilizando Machine Learning para predecir el riesgo de abandono (Churn) y calificar la calidad de los prospectos (Lead Scoring).

## 📂 Estructura del Proyecto

El proyecto se ha reorganizado para ser escalable y mantenible:

- **`src/`**: Código fuente (`Dashboard.py`, `Chatbot.py`, `train_models.py`).
- **`data/raw/`**: Archivos CSV originales (`leads_historicos.csv`, etc.) y nuevos leads capturados (`new_leads.csv`).
- **`data/processed/`**: Archivos CSV procesados listos para el dashboard.
- **`models/`**: Modelos entrenados (`.pkl`).

## 🚀 Instalación

Asegúrate de tener las dependencias instaladas:

```bash
pip install -r requirements.txt
```

## 🛠️ Cómo Ejecutar

### 1. Entrenar los Modelos
Si es la primera vez o quieres actualizar los modelos con nuevos datos:

```bash
python src/train_models.py
```
Esto generará los archivos `.pkl` en la carpeta `models/` y los datos procesados en `data/processed/`.

### 2. Ejecutar el Dashboard
El dashboard es el centro de control. Ejecútalo con:

```bash
streamlit run src/Dashboard.py
```

### 3. Ejecutar el Chatbot (Captura de Leads)
Para simular la captura de leads en tiempo real:

```bash
streamlit run src/Chatbot.py
```

## 🌟 Funcionalidades Principales

### 📊 Dashboard Interactivo
El dashboard ha sido mejorado con una nueva interfaz de pestañas (**Tabs**) y filtros laterales:

1.  **🏠 Resumen**: Métricas clave e insights generados por IA.
2.  **🔴 Leads en Tiempo Real**:
    *   Muestra los leads capturados por el Chatbot al instante.
    *   Usa el botón **"🔄 Actualizar Datos en Tiempo Real"** para refrescar la tabla sin reiniciar.
3.  **⚠️ Alertas de Churn**: Identifica clientes en riesgo alto (>60%) y sugiere acciones.
4.  **📈 Análisis Estratégico**:
    *   Gráficos de **Plotly** para analizar conversiones por industria y urgencia.
    *   **Filtro Global de Industria** en la barra lateral para profundizar en el análisis.

### 🤖 Chatbot de Captura
*   Interactúa con los usuarios para recolectar datos (Nombre, Industria, Urgencia).
*   Guarda automáticamente la información en `data/raw/new_leads.csv`.
*   Se integra con el Dashboard para mostrar los nuevos prospectos en tiempo real.

## 🧪 Verificación

Para probar el flujo completo:
1.  Abre el **Dashboard** y ve a la pestaña "Leads en Tiempo Real".
2.  Abre el **Chatbot** en otra ventana y completa una conversación.
3.  Vuelve al Dashboard y haz clic en **"🔄 Actualizar Datos en Tiempo Real"**.
4.  ¡Verás el nuevo lead en la tabla!
