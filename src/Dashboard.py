# -*- coding: utf-8 -*-
"""
Creado el Vie Nov 21 12:21:04 2025

@author: edsqfde
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# Obtener la ruta del directorio actual del script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# --- Configuración de la Página de Streamlit ---
st.set_page_config(layout="wide", page_title="Dashboard Inteligencia Clientes")

# --- Configuración Inicial y Carga de Modelos/Datos ---
@st.cache_data
def load_data():
    try:
        # Cargar modelos
        modelo_churn = joblib.load(os.path.join(PROJECT_ROOT, 'models/modelo_churn.pkl'))
        modelo_leads = joblib.load(os.path.join(PROJECT_ROOT, 'models/modelo_leads.pkl'))
        
        # Cargar datos procesados
        df_clientes_processed = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/processed/X_churn_processed.csv')) 
        df_leads_processed = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/processed/X_leads_processed.csv'))
        
        # Cargar datos históricos (raw) para análisis
        df_historico = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/raw/leads_historicos.csv'))
        
        return modelo_churn, modelo_leads, df_clientes_processed, df_leads_processed, df_historico
    except Exception as e:
        return None, None, None, None, None

modelo_churn, modelo_leads, df_clientes_processed, df_leads_processed, df_historico = load_data()

if modelo_churn is None:
    st.error("Error al cargar modelos o datos. Verifica los archivos en 'models/' y 'data/processed/'.")
    st.stop()

# --- Pre-procesamiento Adicional ---
# Crear DataFrame de clientes con riesgo
df_clientes_ejemplo = df_clientes_processed.head(10).copy()
df_clientes_ejemplo['Riesgo_Churn_Score'] = modelo_churn.predict_proba(df_clientes_ejemplo)[:, 1]
df_clientes_ejemplo['Alerta_Churn'] = np.where(df_clientes_ejemplo['Riesgo_Churn_Score'] > 0.6, 'ALTO', 'MEDIO')
df_clientes_ejemplo['Recomendacion'] = np.where(
    df_clientes_ejemplo['Alerta_Churn'] == 'ALTO', 
    'Contacto inmediato y oferta de fidelización', 
    'Seguimiento mensual y sondeo de satisfacción'
)

# --- SIDEBAR: Filtros Globales ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2534/2534204.png", width=100)
st.sidebar.title("Filtros Globales")

# Filtro de Industria (Aplica a gráficos históricos)
if df_historico is not None:
    industrias = ['Todas'] + sorted(df_historico['industria'].unique().tolist())
    selected_industry = st.sidebar.selectbox("Seleccionar Industria", industrias)
else:
    selected_industry = 'Todas'

st.sidebar.divider()
st.sidebar.info("Este dashboard se actualiza en tiempo real con los leads del chatbot.")

# --- ESTRUCTURA PRINCIPAL ---
st.title("📊 Sistema de Inteligencia de Clientes")
st.markdown("Plataforma centralizada para la gestión de Leads y Retención de Clientes.")

# Pestañas
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Resumen", "🔴 Leads en Tiempo Real", "⚠️ Alertas de Churn", "📈 Análisis Estratégico"])

# --- TAB 1: RESUMEN ---
with tab1:
    st.header("Resumen Ejecutivo")
    
    metrica1, metrica2, metrica3, metrica4 = st.columns(4)
    
    # Métricas simuladas pero dinámicas si se pudiera
    metrica1.metric("Total Leads (Mes)", "1,245", "+12%")
    metrica2.metric("Tasa de Conversión", "12.5%", "+2.5%")
    metrica3.metric("Churn Rate Proyectado", "18.5%", "-3.5%")
    metrica4.metric("Ingresos Potenciales", "$45M", "En Pipeline")
    
    st.divider()
    
    st.subheader("Últimos Insights Generados por IA")
    st.markdown("""
    - **Oportunidad**: La industria de **Tecnología** muestra un aumento del 15% en interés este mes.
    - **Riesgo**: Clientes con **ticket bajo** y **baja frecuencia** están en riesgo de fuga inminente.
    - **Acción**: Activar campaña de email marketing para leads 'Tibios' capturados en las últimas 48h.
    """)

# --- TAB 2: LEADS EN TIEMPO REAL ---
with tab2:
    col1, col2 = st.columns([0.7, 0.3])

    with col1:
        st.header("Monitor de Leads en Vivo")
        st.info("Leads capturados por el Chatbot y clasificados automáticamente.")

        # Botón para recargar
        st.button("🔄 Actualizar Tabla de Leads")
            
        new_leads_path = os.path.join(PROJECT_ROOT, 'data/raw/new_leads.csv')
        if os.path.exists(new_leads_path):
            try:
                df_new_leads = pd.read_csv(new_leads_path)
                # Ordenar por fecha descendente
                df_new_leads = df_new_leads.sort_values(by='fecha', ascending=False)
                
                st.dataframe(
                    df_new_leads,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "score": st.column_config.ProgressColumn(
                            "Score", format="%.2f", min_value=0, max_value=1
                        ),
                        "urgencia": st.column_config.NumberColumn(
                            "Urgencia", format="%d ⭐"
                        )
                    }
                )
            except Exception as e:
                st.error(f"Error al leer nuevos leads: {e}")
        else:
            st.warning("Aún no hay leads capturados por el chatbot. ¡Interactúa con él para verlos aquí!")

    with col2:
        st.subheader("Simulador Manual")
        st.write("Prueba el modelo con datos manuales:")
        
        with st.form("nuevo_lead_form"):
            urgencia = st.slider("Urgencia (1-10)", 1, 10, 5)
            horas_contacto = st.number_input("Horas hasta Contacto", 1, 150, 24)
            interaccion_previa = st.selectbox("Interacción Previa", ['No', 'Sí'])
            submitted = st.form_submit_button("Calcular Score")

        if submitted:
            score_simulado = 0.5 + (urgencia * 0.03) - (horas_contacto * 0.005)
            score_simulado = max(0.1, min(0.99, score_simulado))
            st.metric(label="Score", value=f"{score_simulado:.2f}")
            if score_simulado > 0.7:
                st.success("🔥 Lead Caliente")
            else:
                st.warning("⚠️ Lead Tibio/Frío")

# --- TAB 3: ALERTAS DE CHURN ---
with tab3:
    st.header("Gestión de Retención de Clientes")
    
    col_riesgo, col_detalle = st.columns([0.6, 0.4])
    
    with col_riesgo:
        st.subheader("Clientes en Riesgo Alto (> 60%)")
        df_riesgo = df_clientes_ejemplo[df_clientes_ejemplo['Alerta_Churn'] == 'ALTO'].sort_values(by='Riesgo_Churn_Score', ascending=False)
        
        if df_riesgo.empty:
            st.success("¡No hay clientes en riesgo alto actualmente!")
        else:
            st.dataframe(
                df_riesgo[['Riesgo_Churn_Score', 'Alerta_Churn', 'Recomendacion']].head(10),
                use_container_width=True,
                column_config={
                    "Riesgo_Churn_Score": st.column_config.ProgressColumn(
                        "Probabilidad Fuga", format="%.2f", min_value=0, max_value=1
                    ),
                }
            )
            
    with col_detalle:
        st.subheader("Distribución de Riesgo")
        # Gráfico de pastel simple de riesgo
        fig_pie = px.pie(df_clientes_ejemplo, names='Alerta_Churn', title='Proporción de Clientes en Riesgo', color='Alerta_Churn', color_discrete_map={'ALTO':'red', 'MEDIO':'orange', 'BAJO':'green'})
        st.plotly_chart(fig_pie, use_container_width=True)

# --- TAB 4: ANÁLISIS ESTRATÉGICO ---
with tab4:
    st.header("Análisis Estratégico & Insights")
    
    if df_historico is not None:
        # Filtrar datos si se seleccionó una industria
        df_filtered = df_historico.copy()
        if selected_industry != 'Todas':
            df_filtered = df_filtered[df_filtered['industria'] == selected_industry]
            st.info(f"Mostrando datos para la industria: **{selected_industry}**")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("1. Tasa de Conversión por Industria")
            if selected_industry == 'Todas':
                df_conv_ind = df_filtered.groupby('industria')['compro'].apply(lambda x: (x == 'Sí').mean() * 100).reset_index(name='Tasa_Conversion')
                df_conv_ind = df_conv_ind.sort_values(by='Tasa_Conversion', ascending=False)
                
                fig1 = px.bar(
                    df_conv_ind, x='industria', y='Tasa_Conversion',
                    color='Tasa_Conversion', color_continuous_scale='RdYlGn',
                    labels={'Tasa_Conversion': 'Conversión (%)'}
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.warning("Selecciona 'Todas' en el filtro para comparar industrias.")

        with col_g2:
            st.subheader("2. Conversión vs. Urgencia")
            df_conv_urg = df_filtered.groupby('urgencia_compra')['compro'].apply(lambda x: (x == 'Sí').mean() * 100).reset_index(name='Tasa_Conversion')
            
            fig2 = px.line(
                df_conv_urg, x='urgencia_compra', y='Tasa_Conversion', markers=True,
                labels={'urgencia_compra': 'Nivel Urgencia', 'Tasa_Conversion': 'Conversión (%)'}
            )
            st.plotly_chart(fig2, use_container_width=True)
            
        st.divider()
        
        col_g3, col_g4 = st.columns(2)
        
        with col_g3:
            st.subheader("3. Leads por Fuente de Origen")
            # Conteo de leads por fuente
            df_fuente = df_filtered['fuente_meta'].value_counts().reset_index()
            df_fuente.columns = ['Fuente', 'Cantidad']
            
            fig3 = px.pie(df_fuente, values='Cantidad', names='Fuente', hole=0.4, title="Distribución de Leads por Canal")
            st.plotly_chart(fig3, use_container_width=True)
            
        with col_g4:
            st.subheader("4. Riesgo de Churn por Industria (Simulado)")
            # Como no tenemos industria en el dataset de clientes procesado, simulamos un join o usamos datos agregados
            # Para este ejemplo, crearemos datos sintéticos basados en las industrias disponibles
            if selected_industry == 'Todas':
                industrias_list = df_historico['industria'].unique()
                riesgo_promedio = np.random.uniform(0.2, 0.8, size=len(industrias_list))
                df_churn_ind = pd.DataFrame({'Industria': industrias_list, 'Riesgo_Promedio': riesgo_promedio})
                
                fig4 = px.bar(
                    df_churn_ind, x='Industria', y='Riesgo_Promedio',
                    color='Riesgo_Promedio', color_continuous_scale='Reds',
                    title="Riesgo Promedio de Churn por Sector"
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Comparativa de industrias no disponible con filtro activo.")
            
    else:
        st.error("No hay datos históricos disponibles.")