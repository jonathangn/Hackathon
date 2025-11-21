# Enhanced Dashboard Walkthrough

I have significantly improved the **Dashboard UX** and added **Strategic Visualizations**.

## Key Improvements

### 1. Tabs Layout
The dashboard is now organized into 4 tabs for better navigation:
- **🏠 Resumen**: High-level metrics and AI insights.
- **🔴 Leads en Tiempo Real**: Live monitoring of chatbot leads.
- **⚠️ Alertas de Churn**: Risk management for existing customers.
- **📈 Análisis Estratégico**: Deep dive into historical data and trends.

### 2. Sidebar Filters
- **Global Industry Filter**: You can now filter the "Strategic Analysis" charts by **Industry** using the sidebar. This allows for more granular analysis.

### 3. New Visualizations
In the **Análisis Estratégico** tab, you will find:
- **Leads by Source**: A pie chart showing which channels (Instagram, Facebook, etc.) generate the most leads.
- **Churn Risk by Industry**: A bar chart comparing the average churn risk across different sectors.

## How to Use
1.  Run the dashboard:
    ```bash
    streamlit run src/Dashboard.py
    ```
2.  Navigate through the tabs to see different views.
3.  Use the **Sidebar** to filter charts by Industry.
