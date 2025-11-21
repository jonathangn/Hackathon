import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Asistente Virtual de Ventas", page_icon="🤖")

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models/modelo_leads.pkl')
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, 'models/preprocessor_leads.pkl')

# Cargar recursos
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except FileNotFoundError:
        return None, None

model, preprocessor = load_resources()

st.title("🤖 Chatbot de Captura de Leads")

if model is None or preprocessor is None:
    st.error("Error: No se encontraron los modelos. Por favor ejecuta 'python src/train_models.py' primero.")
    st.stop()

# Inicializar estado
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! 👋 Soy el asistente virtual. Estoy aquí para registrar tu interés en nuestros productos. Para comenzar, ¿cuál es tu nombre?"}
    ]
    st.session_state.step = 1 # 1: Nombre, 2: Industria, 3: Cargo, 4: Ciudad, 5: Urgencia, 6: Final
    st.session_state.user_data = {}

# Opciones válidas (basadas en el entrenamiento)
INDUSTRIAS = ['Educación', 'Ropa Deportiva', 'Seguros']
CARGOS = ['Contador', 'Director', 'Coordinador', 'Especialista', 'Supervisor', 'Profesional Independiente', 'Consultor', 'Gerente', 'Analista', 'Auxiliar', 'Asesor', 'Jefe', 'Asistente', 'Empresario', 'Administrador']
CIUDADES = ['Pasto', 'Armenia', 'Bogotá', 'Villavicencio', 'Cúcuta', 'Neiva', 'Ibagué', 'Medellín', 'Valledupar', 'Manizales', 'Pereira', 'Barranquilla', 'Cartagena', 'Cali', 'Bucaramanga']

# Mostrar historial de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Función para procesar la entrada del usuario
def process_input(user_input):
    step = st.session_state.step
    
    # Guardar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    response = ""
    
    if step == 1: # Nombre
        st.session_state.user_data['nombre'] = user_input
        response = f"¡Gracias {user_input}! ¿En qué industria trabajas? (Opciones: {', '.join(INDUSTRIAS)})"
        st.session_state.step = 2
        
    elif step == 2: # Industria
        # Validación simple
        matched = [i for i in INDUSTRIAS if i.lower() in user_input.lower()]
        if matched:
            st.session_state.user_data['industria'] = matched[0]
            response = f"Entendido, {matched[0]}. ¿Cuál es tu cargo actual? (Ej: {', '.join(CARGOS[:3])}...)"
            st.session_state.step = 3
        else:
            response = f"Por favor selecciona una de las siguientes industrias: {', '.join(INDUSTRIAS)}"
            # No avanzamos de paso
            
    elif step == 3: # Cargo
        # Validación laxa, intentamos buscar coincidencia, si no, guardamos lo que escribió
        matched = [c for c in CARGOS if c.lower() in user_input.lower()]
        val = matched[0] if matched else user_input
        st.session_state.user_data['cargo_lead'] = val
        response = f"Perfecto. ¿Desde qué ciudad nos escribes? (Ej: Bogotá, Medellín, Cali...)"
        st.session_state.step = 4
        
    elif step == 4: # Ciudad
        matched = [c for c in CIUDADES if c.lower() in user_input.lower()]
        val = matched[0] if matched else user_input
        st.session_state.user_data['ciudad'] = val
        response = "Ya casi terminamos. En una escala del 1 al 10, ¿qué tan urgente es tu necesidad de compra?"
        st.session_state.step = 5
        
    elif step == 5: # Urgencia
        try:
            urgencia = int(user_input)
            if 1 <= urgencia <= 10:
                st.session_state.user_data['urgencia_compra'] = urgencia
                
                # --- PREDICCIÓN ---
                score, proba = predict_lead_score(st.session_state.user_data)
                
                st.session_state.user_data['score'] = proba
                
                calidad = "🔥 CALIENTE" if proba > 0.7 else ("⚠️ TIBIO" if proba > 0.4 else "❄️ FRÍO")
                
                response = f"""¡Gracias por la información! Hemos registrado tus datos.
                
**Análisis de tu perfil:**
*   **Score de Lead:** {proba:.2f}
*   **Calidad:** {calidad}

Un asesor se pondrá en contacto contigo pronto."""
                st.session_state.step = 6 # Fin
            else:
                response = "Por favor ingresa un número entre 1 y 10."
        except ValueError:
            response = "Por favor ingresa un número válido."

    elif step == 6:
        response = "Ya tenemos tus datos. Si deseas iniciar de nuevo, recarga la página."

    # Guardar respuesta del bot
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Forzar recarga para mostrar mensajes
    st.rerun()

def predict_lead_score(data):
    # Construir DataFrame con una sola fila
    # Llenar con valores por defecto para las columnas que no pedimos pero el modelo necesita
    
    # Columnas esperadas por el modelo (basado en lo que vimos en train_models.py)
    # Necesitamos recrear la estructura exacta de X_leads antes del preprocesador
    
    # Valores por defecto para un lead web nuevo
    input_data = {
        'industria': [data.get('industria', 'Otros')],
        'tipo_campana': ['Orgánico'], # Asumimos orgánico
        'fuente_meta': ['Directo'],   # Asumimos directo
        'dispositivo': ['Mobile'],    # Asumimos mobile
        'hora_generacion': [12],      # Hora promedio
        'cargo_lead': [data.get('cargo_lead', 'Otros')],
        'ciudad': [data.get('ciudad', 'Bogotá')],
        'urgencia_compra': [data.get('urgencia_compra', 5)],
        'interaccion_previa': ['No'],
        'horas_hasta_contacto': [24], # Promedio
        'lead_respondio': ['Sí'],     # Asumimos que sí porque está chateando
        'intentos_contacto': [1],
        # Columnas que se eliminan en train_models.py pero que podrían estar en el raw df si el preprocessor las espera?
        # No, el preprocessor recibe X_leads que ya tiene las columnas eliminadas.
        # Pero espera, el preprocessor se ajustó sobre X_leads.
        # X_leads tiene: industria, tipo_campana, fuente_meta, dispositivo, hora_generacion, cargo_lead, ciudad, urgencia_compra, interaccion_previa, horas_hasta_contacto, lead_respondio, intentos_contacto
    }
    
    df_input = pd.DataFrame(input_data)
    
    # Transformar
    try:
        X_processed = preprocessor.transform(df_input)
        # Predecir
        proba = model.predict_proba(X_processed)[:, 1][0]
        return 0, proba # Retornamos clase 0 (dummy) y probabilidad
    except Exception as e:
        st.error(f"Error en predicción: {e}")
        return 0, 0.0

# Input de chat
if prompt := st.chat_input("Escribe tu respuesta aquí..."):
    process_input(prompt)
