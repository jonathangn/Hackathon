# Informe Técnico: Procesamiento de Datos y Modelado

Este documento detalla el flujo completo de los datos en el **Sistema de Inteligencia de Clientes**, desde la ingesta de datos crudos hasta la visualización en tiempo real, así como el historial de mejoras implementadas.

## 1. Fuentes de Datos (`data/raw/`)

El sistema se alimenta de tres fuentes de datos principales en formato CSV:

1.  **`leads_historicos.csv`**: Contiene información histórica de prospectos (leads).
    *   *Columnas Clave*: `industria`, `urgencia_compra`, `fuente_meta`, `compro` (Variable Objetivo para Lead Scoring).
2.  **`clientes_comportamiento.csv`**: Datos sobre el comportamiento de compra y satisfacción de clientes actuales.
    *   *Columnas Clave*: `id_cliente`, `frecuencia_compra`, `satisfaccion`, `dias_desde_ultima_compra`.
3.  **`clientes_transacciones.csv`**: Datos demográficos y transaccionales de las empresas clientes.
    *   *Columnas Clave*: `id_cliente`, `presupuesto`, `tamaño_empresa`.

## 2. Pipeline de Procesamiento (ETL)

El script `src/train_models.py` orquesta el proceso de Extracción, Transformación y Carga:

### A. Preparación para Modelo de Churn (Retención)
1.  **Fusión de Datos (Join)**: Se unen `clientes_comportamiento` y `clientes_transacciones` usando `id_cliente` como llave.
2.  **Ingeniería de Características**:
    *   Se calcula la mediana de `dias_desde_ultima_compra`.
    *   **Variable Objetivo (`churn_riesgo`)**: Se define como `1` (Riesgo) si el cliente es 'Nuevo' y lleva más tiempo sin comprar que la mediana. De lo contrario es `0`.
3.  **Preprocesamiento**:
    *   **Numérico**: Imputación de valores faltantes (mediana) y escalado estándar.
    *   **Categórico**: Codificación One-Hot (OneHotEncoder) para convertir variables de texto en números.

### B. Preparación para Modelo de Lead Scoring (Captación)
1.  **Limpieza**: Se eliminan columnas irrelevantes o con alta cardinalidad (ej. IDs, fechas exactas).
2.  **Variable Objetivo (`compro`)**: Se convierte la columna `compro` ('Sí'/'No') a binaria (1/0).
3.  **Preprocesamiento**: Similar al modelo de Churn, se aplica un pipeline de imputación y codificación a las variables del lead (industria, cargo, urgencia, etc.).

## 3. Modelado Predictivo

Se entrenan dos modelos de **Regresión Logística** debido a su interpretabilidad y eficiencia para este MVP:

| Modelo | Objetivo | Métrica Principal | Insight Clave |
| :--- | :--- | :--- | :--- |
| **Modelo de Churn** | Predecir probabilidad de abandono. | **Recall (85%)**: Priorizamos detectar a *todos* los posibles abandonos, aunque haya falsas alarmas. | Clientes con baja satisfacción son críticos. |
| **Modelo de Lead Scoring** | Predecir probabilidad de compra. | **Accuracy (~76%)**: Buscamos un balance general. | La `urgencia_compra` es el predictor más fuerte. |

Los modelos entrenados se guardan como archivos `.pkl` en la carpeta `models/`.

## 4. Arquitectura en Tiempo Real

El sistema integra un flujo de datos en vivo para operar con nuevos leads:

1.  **Captura (Chatbot)**:
    *   El script `src/Chatbot.py` interactúa con el usuario.
    *   Recopila datos: Nombre, Industria, Cargo, Ciudad, Urgencia.
    *   **Inferencia en Vivo**: Carga el modelo `modelo_leads.pkl` y calcula el *Score* en tiempo real.
2.  **Almacenamiento Intermedio**:
    *   Los datos del nuevo lead (incluyendo su Score) se anexan a `data/raw/new_leads.csv`.
3.  **Visualización (Dashboard)**:
    *   El script `src/Dashboard.py` lee `new_leads.csv`.
    *   Muestra los leads más recientes en la pestaña "🔴 Leads en Tiempo Real".
    *   Permite actualización manual mediante un botón, sin necesidad de reiniciar el servidor.

## 5. Historial de Implementación y Mejoras

A continuación se detallan las fases de desarrollo y las mejoras específicas realizadas en el proyecto.

### A. Reorganización del Proyecto
Para mejorar la escalabilidad y mantenibilidad, se reestructuró el proyecto:
-   **Estructura de Directorios**:
    -   `src/`: Código fuente.
    -   `data/raw/` y `data/processed/`: Gestión ordenada de datos.
    -   `models/`: Artefactos de ML.
-   **Actualización de Código**: Se modificaron `train_models.py` y `Dashboard.py` para usar rutas relativas y absolutas robustas, asegurando que funcionen independientemente del directorio de ejecución.

### B. Implementación de Tiempo Real
Se conectó el Chatbot con el Dashboard para permitir un flujo de trabajo dinámico:
-   **Chatbot (`src/Chatbot.py`)**: Se añadió la función `save_lead` para guardar cada interacción en `new_leads.csv`.
-   **Dashboard (`src/Dashboard.py`)**: Se añadió la lógica para leer este archivo y un botón de actualización ("🔄 Actualizar Datos en Tiempo Real").
-   **Verificación**: Se probó simulando un lead en el chatbot y verificando su aparición inmediata en el dashboard.

### C. Mejoras de UX y Visualización Estratégica
Se transformó el dashboard en una herramienta de gestión completa:
1.  **Diseño con Pestañas (Tabs)**:
    -   *Resumen*: Métricas clave.
    -   *Leads en Tiempo Real*: Operación diaria.
    -   *Alertas de Churn*: Gestión de riesgos.
    -   *Análisis Estratégico*: Inteligencia de negocios.
2.  **Filtros Laterales**: Se añadió un filtro global de "Industria" para segmentar los gráficos estratégicos.
3.  **Nuevas Visualizaciones (Plotly)**:
    -   *Conversión por Industria*: Gráfico de barras.
    -   *Conversión vs. Urgencia*: Gráfico de líneas para ver tendencias.
    -   *Leads por Fuente*: Gráfico de pastel.
    -   *Riesgo de Churn por Industria*: Comparativa visual.

## 6. Conclusión General

Esta arquitectura permite a la empresa:
1.  **Priorizar** esfuerzos de ventas en leads con alta probabilidad de compra (Hot Leads).
2.  **Prevenir** la fuga de clientes actuales mediante alertas tempranas de Churn.
3.  **Monitorear** la operación en tiempo real a través de un dashboard centralizado y fácil de usar.
